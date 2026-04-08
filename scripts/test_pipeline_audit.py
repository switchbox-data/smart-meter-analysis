#!/usr/bin/env python3
"""End-to-end audit: synthetic data → pipeline → hand-calculated verification.

Generates synthetic hourly loads with known kWh patterns, runs the actual
compute_delivery_deltas.py and package_dtou_results.py pipeline code on
them, and asserts that every output column matches hand-calculated expected
values within floating-point tolerance.

Usage::

    uv run python scripts/test_pipeline_audit.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = Path("/tmp/pipeline_audit")  # noqa: S108
COMPUTE_SCRIPT = REPO_ROOT / "scripts" / "pricing_pilot" / "compute_delivery_deltas.py"
PACKAGE_SCRIPT = REPO_ROOT / "scripts" / "pricing_pilot" / "package_dtou_results.py"

# ── Test configuration ────────────────────────────────────────────────────

ZIP_CODE = "60601"
TOLERANCE = 1e-6

DELIVERY_CLASSES = ["C23", "C24", "C26", "C28"]

# (month_str, season, days_in_month)
MONTHS_CFG = [
    ("202301", "nonsummer", 31),
    ("202307", "summer", 31),
]

# kWh per hour for each TOU period — chosen to exercise period weighting
KWH_PER_HOUR: dict[str, float] = {
    "morning": 2.0,
    "midday_peak": 3.0,
    "evening": 1.5,
    "overnight": 1.0,
}

# ── Period mapping (must match compute_delivery_deltas.py PERIOD_MAP) ─────

PERIOD_MAP: dict[int, str] = {
    0: "overnight",
    1: "overnight",
    2: "overnight",
    3: "overnight",
    4: "overnight",
    5: "overnight",
    6: "morning",
    7: "morning",
    8: "morning",
    9: "morning",
    10: "morning",
    11: "morning",
    12: "morning",
    13: "midday_peak",
    14: "midday_peak",
    15: "midday_peak",
    16: "midday_peak",
    17: "midday_peak",
    18: "midday_peak",
    19: "evening",
    20: "evening",
    21: "overnight",
    22: "overnight",
    23: "overnight",
}

# Hours per period per day (derived from PERIOD_MAP for explicitness)
HOURS_PER_PERIOD_PER_DAY: dict[str, int] = {
    "morning": 7,  # hours 6-12
    "midday_peak": 6,  # hours 13-18
    "evening": 2,  # hours 19-20
    "overnight": 9,  # hours 0-5, 21-23
}
assert sum(HOURS_PER_PERIOD_PER_DAY.values()) == 24  # noqa: S101

# ── Rate constants (must match pipeline code exactly) ─────────────────────

FLAT_PTCS: dict[str, float] = {"summer": 10.028, "nonsummer": 9.660}

STOU_SUPPLY: dict[str, dict[str, float]] = {
    "summer": {
        "morning": 4.279,
        "midday_peak": 19.485,
        "evening": 4.356,
        "overnight": 3.136,
    },
    "nonsummer": {
        "morning": 4.095,
        "midday_peak": 18.080,
        "evening": 4.352,
        "overnight": 3.278,
    },
}

FLAT_DFCS: dict[str, float] = {
    "C23": 6.228,
    "C24": 4.791,
    "C26": 3.165,
    "C28": 2.996,
}

TOU_DFCS: dict[str, dict[str, float]] = {
    "C23": {"morning": 4.009, "midday_peak": 10.712, "evening": 3.747, "overnight": 2.984},
    "C24": {"morning": 3.073, "midday_peak": 8.689, "evening": 2.856, "overnight": 2.251},
    "C26": {"morning": 1.999, "midday_peak": 5.329, "evening": 1.890, "overnight": 1.550},
    "C28": {"morning": 1.925, "midday_peak": 4.975, "evening": 1.823, "overnight": 1.512},
}


# ── Helpers ───────────────────────────────────────────────────────────────


def _acct_name(dc: str, month_str: str) -> str:
    """Generate a deterministic account name for a delivery class + month."""
    return f"ACCT_{dc}_{month_str}"


# ── Step 1: Generate synthetic data ──────────────────────────────────────


def generate_synthetic_data() -> None:
    """Create hourly_loads parquet files and delivery_class_lookup.parquet.

    Directory layout matches what compute_delivery_deltas.py expects:

        {AUDIT_DIR}/
            statewide_stou_{month}_v2/
                statewide_stou_{month}_v2/
                    _tmp/
                        month={month}/
                            hourly_loads.parquet
            delivery_class_lookup.parquet
    """
    print("\n" + "=" * 60)
    print("STEP 1: Generate synthetic data")
    print("=" * 60)

    if AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR)

    # --- Hourly loads (one file per month) ---
    for month_str, _season, days in MONTHS_CFG:
        run_name = f"statewide_stou_{month_str}_v2"
        hourly_dir = AUDIT_DIR / run_name / run_name / "_tmp" / f"month={month_str}"
        hourly_dir.mkdir(parents=True)

        year = int(month_str[:4])
        month_int = int(month_str[4:6])

        rows: list[dict] = []
        for dc in DELIVERY_CLASSES:
            acct = _acct_name(dc, month_str)
            for day in range(1, days + 1):
                for hour in range(24):
                    period = PERIOD_MAP[hour]
                    rows.append({
                        "account_identifier": acct,
                        "zip_code": ZIP_CODE,
                        "hour_chicago": datetime(year, month_int, day, hour, 0, 0),
                        "kwh_hour": KWH_PER_HOUR[period],
                    })

        df = pl.DataFrame(rows).with_columns(
            pl.col("hour_chicago").cast(pl.Datetime("us")),
            pl.col("kwh_hour").cast(pl.Float64),
        )

        total_hours = days * 24
        assert len(df) == len(DELIVERY_CLASSES) * total_hours, (  # noqa: S101
            f"Expected {len(DELIVERY_CLASSES) * total_hours} rows, got {len(df)}"
        )

        hourly_path = hourly_dir / "hourly_loads.parquet"
        df.write_parquet(hourly_path)
        print(f"  {month_str}: {len(df):,} rows → {hourly_path}")

    # --- Delivery class lookup (all 8 accounts) ---
    lookup_rows = []
    for month_str, _, _ in MONTHS_CFG:
        for dc in DELIVERY_CLASSES:
            lookup_rows.append({
                "account_identifier": _acct_name(dc, month_str),
                "delivery_service_class": dc,
            })

    lookup_df = pl.DataFrame(lookup_rows)
    lookup_path = AUDIT_DIR / "delivery_class_lookup.parquet"
    lookup_df.write_parquet(lookup_path)
    print(f"  Lookup: {len(lookup_df)} rows → {lookup_path}")


# ── Step 2: Hand-calculate expected values ───────────────────────────────


def hand_calculate_expected() -> dict[str, dict]:
    """Compute expected values with explicit arithmetic — no pipeline code.

    Returns {account_name: {column: value}} for all 8 test accounts.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Hand-calculate expected values")
    print("=" * 60)

    expected: dict[str, dict] = {}

    for month_str, season, days in MONTHS_CFG:
        # ── Period kWh totals (same for all accounts in this month) ──

        period_kwh: dict[str, float] = {}
        for period in ("morning", "midday_peak", "evening", "overnight"):
            hours_in_period = HOURS_PER_PERIOD_PER_DAY[period] * days
            period_kwh[period] = hours_in_period * KWH_PER_HOUR[period]

        total_kwh = sum(period_kwh.values())

        print(f"\n  {month_str} ({season}, {days} days):")
        print(
            f"    Hours per period: morning={HOURS_PER_PERIOD_PER_DAY['morning'] * days}, "
            f"midday_peak={HOURS_PER_PERIOD_PER_DAY['midday_peak'] * days}, "
            f"evening={HOURS_PER_PERIOD_PER_DAY['evening'] * days}, "
            f"overnight={HOURS_PER_PERIOD_PER_DAY['overnight'] * days}"
        )
        print(
            f"    kWh per period: morning={period_kwh['morning']:.1f}, "
            f"midday_peak={period_kwh['midday_peak']:.1f}, "
            f"evening={period_kwh['evening']:.1f}, "
            f"overnight={period_kwh['overnight']:.1f}"
        )
        print(f"    Total kWh: {total_kwh:.1f}")

        # ── Supply (same for all delivery classes in this month) ──

        flat_ptc = FLAT_PTCS[season]

        # flat_supply = total_kwh x flat_ptc / 100
        flat_supply = total_kwh * flat_ptc / 100.0

        # stou_supply = Σ(period_kwh x stou_rate) / 100
        stou_supply_cents = 0.0
        for p in ("morning", "midday_peak", "evening", "overnight"):
            contrib = period_kwh[p] * STOU_SUPPLY[season][p]
            stou_supply_cents += contrib
        stou_supply = stou_supply_cents / 100.0

        # supply_delta = flat - stou  (positive = customer saves under TOU)
        supply_delta = flat_supply - stou_supply

        print(f"    Flat supply: {total_kwh} x {flat_ptc}¢ / 100 = ${flat_supply:.6f}")
        print(f"    STOU supply: ${stou_supply:.6f}")
        print(f"    Supply delta: ${supply_delta:.6f}")

        # ── Per delivery class: delivery + combined totals ──

        for dc in DELIVERY_CLASSES:
            acct = _acct_name(dc, month_str)

            # flat_delivery = total_kwh x flat_dfc / 100
            flat_dfc = FLAT_DFCS[dc]
            flat_delivery = total_kwh * flat_dfc / 100.0

            # tou_delivery = Σ(period_kwh x tou_dfc) / 100
            tou_delivery_cents = 0.0
            for p in ("morning", "midday_peak", "evening", "overnight"):
                tou_delivery_cents += period_kwh[p] * TOU_DFCS[dc][p]
            tou_delivery = tou_delivery_cents / 100.0

            # delivery_delta = flat - tou  (positive = customer saves under TOU)
            delivery_delta = flat_delivery - tou_delivery

            # ── STOU combined ──
            total_bill_a = flat_supply + flat_delivery
            total_bill_b = stou_supply + tou_delivery
            total_delta = supply_delta + delivery_delta
            total_pct_savings = total_delta / total_bill_a * 100.0

            # ── DTOU (supply cancels: same flat PTC on both sides) ──
            dtou_total_bill_a = flat_supply + flat_delivery  # = total_bill_a
            dtou_total_bill_b = flat_supply + tou_delivery  # flat supply, not STOU
            dtou_total_delta = delivery_delta  # supply cancels
            dtou_total_pct_savings = delivery_delta / dtou_total_bill_a * 100.0

            expected[acct] = {
                "month": month_str,
                "season": season,
                "delivery_service_class": dc,
                "total_kwh": total_kwh,
                # Supply
                "bill_a_dollars": flat_supply,
                "bill_b_dollars": stou_supply,
                "supply_delta_dollars": supply_delta,
                # Delivery
                "flat_delivery_dollars": flat_delivery,
                "tou_delivery_dollars": tou_delivery,
                "delivery_delta_dollars": delivery_delta,
                # STOU combined
                "total_bill_a_dollars": total_bill_a,
                "total_bill_b_dollars": total_bill_b,
                "total_delta_dollars": total_delta,
                "total_pct_savings": total_pct_savings,
                # DTOU
                "flat_supply_dollars": flat_supply,  # = bill_a_dollars
                "dtou_total_bill_a_dollars": dtou_total_bill_a,
                "dtou_total_bill_b_dollars": dtou_total_bill_b,
                "dtou_total_delta_dollars": dtou_total_delta,
                "dtou_total_pct_savings": dtou_total_pct_savings,
            }

            print(f"\n    {acct} ({dc}):")
            print(f"      Flat delivery: {total_kwh} x {flat_dfc}¢ / 100 = ${flat_delivery:.6f}")
            print(
                f"      TOU delivery: "
                f"({period_kwh['morning']}x{TOU_DFCS[dc]['morning']} + "
                f"{period_kwh['midday_peak']}x{TOU_DFCS[dc]['midday_peak']} + "
                f"{period_kwh['evening']}x{TOU_DFCS[dc]['evening']} + "
                f"{period_kwh['overnight']}x{TOU_DFCS[dc]['overnight']}) / 100 "
                f"= ${tou_delivery:.6f}"
            )
            print(f"      Delivery delta: ${delivery_delta:.6f}")
            print(
                f"      STOU: bill_a=${total_bill_a:.6f}  bill_b=${total_bill_b:.6f}  "
                f"delta=${total_delta:.6f}  pct={total_pct_savings:.6f}%"
            )
            print(
                f"      DTOU: bill_a=${dtou_total_bill_a:.6f}  bill_b=${dtou_total_bill_b:.6f}  "
                f"delta=${dtou_total_delta:.6f}  pct={dtou_total_pct_savings:.6f}%"
            )

    return expected


# ── Step 3: Run the actual pipeline ──────────────────────────────────────


def run_pipeline() -> None:
    """Invoke compute_delivery_deltas.py and package_dtou_results.py."""
    print("\n" + "=" * 60)
    print("STEP 3: Run pipeline")
    print("=" * 60)

    # --- compute_delivery_deltas.py --both ---
    cmd_compute = [
        sys.executable,
        str(COMPUTE_SCRIPT),
        "--both",
        "--billing-output-dir",
        str(AUDIT_DIR),
    ]
    print(f"  Running: {' '.join(cmd_compute)}")
    result = subprocess.run(cmd_compute, capture_output=True, text=True)  # noqa: S603
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"compute_delivery_deltas.py failed with rc={result.returncode}")

    # --- package_dtou_results.py --both ---
    cmd_package = [
        sys.executable,
        str(PACKAGE_SCRIPT),
        "--both",
        "--billing-output-dir",
        str(AUDIT_DIR),
    ]
    print(f"  Running: {' '.join(cmd_package)}")
    result = subprocess.run(cmd_package, capture_output=True, text=True)  # noqa: S603
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"package_dtou_results.py failed with rc={result.returncode}")


# ── Step 4: Compare actual vs expected ───────────────────────────────────


def _compare_column(
    label: str,
    actual: float | None,
    expected_val: float,
    tol: float,
) -> bool:
    """Compare one value; return True if PASS."""
    if actual is None:
        diff = float("inf")
        status = "FAIL"
    else:
        diff = abs(actual - expected_val)
        status = "PASS" if diff < tol else "FAIL"
    print(f"      {label:30s}  expected={expected_val:>14.6f}  actual={actual!s:>14s}  diff={diff:.2e}  {status}")
    return status == "PASS"


def _compare_stou_account(
    df: pl.DataFrame,
    dc: str,
    month_str: str,
    expected: dict[str, dict],
    numeric_cols: list[str],
) -> int:
    """Compare one account's STOU row against expected values. Returns failure count."""
    acct = _acct_name(dc, month_str)
    exp = expected[acct]

    row = df.filter(pl.col("account_identifier") == acct)
    if len(row) == 0:
        print(f"\n    FAIL: {acct} not found in output")
        return 1
    if len(row) > 1:
        print(f"\n    FAIL: {acct} has {len(row)} rows (expected 1)")
        return 1

    row_dict = row.to_dicts()[0]
    print(f"\n    {acct} ({dc}, {month_str}):")

    failures = 0
    if row_dict["delivery_service_class"] != dc:
        print(f"      FAIL: delivery_service_class expected={dc} actual={row_dict['delivery_service_class']}")
        failures += 1

    for col in numeric_cols:
        if not _compare_column(col, row_dict[col], exp[col], TOLERANCE):
            failures += 1

    return failures


def compare_stou_output(expected: dict[str, dict]) -> int:
    """Read STOU combined parquets and compare against expected values.

    Returns number of failures.
    """
    print("\n" + "=" * 60)
    print("STEP 4a: Compare STOU combined output vs expected")
    print("=" * 60)

    # Expected STOU output schema (14 columns, exact order)
    stou_schema = [
        "account_identifier",
        "zip_code",
        "delivery_service_class",
        "total_kwh",
        "bill_a_dollars",
        "bill_b_dollars",
        "supply_delta_dollars",
        "flat_delivery_dollars",
        "tou_delivery_dollars",
        "delivery_delta_dollars",
        "total_bill_a_dollars",
        "total_bill_b_dollars",
        "total_delta_dollars",
        "total_pct_savings",
    ]

    numeric_cols = [
        "total_kwh",
        "bill_a_dollars",
        "bill_b_dollars",
        "supply_delta_dollars",
        "flat_delivery_dollars",
        "tou_delivery_dollars",
        "delivery_delta_dollars",
        "total_bill_a_dollars",
        "total_bill_b_dollars",
        "total_delta_dollars",
        "total_pct_savings",
    ]

    failures = 0

    for month_str, _, _ in MONTHS_CFG:
        path = AUDIT_DIR / "stou_combined" / f"stou_combined_{month_str}.parquet"
        if not path.exists():
            print(f"  FAIL: output file missing: {path}")
            failures += 1
            continue

        df = pl.read_parquet(path)

        # Verify schema
        if df.columns != stou_schema:
            print(f"  FAIL: schema mismatch for {month_str}")
            print(f"    expected: {stou_schema}")
            print(f"    actual:   {df.columns}")
            failures += 1
            continue
        print(f"\n  {month_str}: schema OK ({len(df.columns)} columns)")

        # Verify row count: 4 accounts per month
        if len(df) != len(DELIVERY_CLASSES):
            print(f"  FAIL: expected {len(DELIVERY_CLASSES)} rows, got {len(df)}")
            failures += 1
            continue
        print(f"  {month_str}: row count OK ({len(df)} rows)")

        for dc in DELIVERY_CLASSES:
            failures += _compare_stou_account(df, dc, month_str, expected, numeric_cols)

    return failures


def compare_dtou_output(expected: dict[str, dict]) -> int:
    """Read DTOU combined parquets and compare against expected values.

    Returns number of failures.
    """
    print("\n" + "=" * 60)
    print("STEP 4b: Compare DTOU combined output vs expected")
    print("=" * 60)

    # Expected DTOU output schema (12 columns)
    dtou_schema = [
        "account_identifier",
        "zip_code",
        "delivery_service_class",
        "total_kwh",
        "flat_supply_dollars",
        "flat_delivery_dollars",
        "tou_delivery_dollars",
        "delivery_delta_dollars",
        "dtou_total_bill_a_dollars",
        "dtou_total_bill_b_dollars",
        "dtou_total_delta_dollars",
        "dtou_total_pct_savings",
    ]

    failures = 0

    for month_str, _, _ in MONTHS_CFG:
        path = AUDIT_DIR / "dtou_combined" / f"dtou_combined_{month_str}.parquet"
        if not path.exists():
            print(f"  FAIL: output file missing: {path}")
            failures += 1
            continue

        df = pl.read_parquet(path)

        # Verify schema
        if df.columns != dtou_schema:
            print(f"  FAIL: schema mismatch for {month_str}")
            print(f"    expected: {dtou_schema}")
            print(f"    actual:   {df.columns}")
            failures += 1
            continue
        print(f"\n  {month_str}: schema OK ({len(df.columns)} columns)")

        # Row count
        if len(df) != len(DELIVERY_CLASSES):
            print(f"  FAIL: expected {len(DELIVERY_CLASSES)} rows, got {len(df)}")
            failures += 1
            continue
        print(f"  {month_str}: row count OK ({len(df)} rows)")

        # DTOU numeric columns to check
        dtou_numeric_cols = [
            "total_kwh",
            "flat_supply_dollars",
            "flat_delivery_dollars",
            "tou_delivery_dollars",
            "delivery_delta_dollars",
            "dtou_total_bill_a_dollars",
            "dtou_total_bill_b_dollars",
            "dtou_total_delta_dollars",
            "dtou_total_pct_savings",
        ]

        for dc in DELIVERY_CLASSES:
            acct = _acct_name(dc, month_str)
            exp = expected[acct]

            row = df.filter(pl.col("account_identifier") == acct)
            if len(row) != 1:
                print(f"\n    FAIL: {acct} has {len(row)} rows (expected 1)")
                failures += 1
                continue

            row_dict = row.to_dicts()[0]
            print(f"\n    {acct} ({dc}, {month_str}):")

            for col in dtou_numeric_cols:
                if not _compare_column(col, row_dict[col], exp[col], TOLERANCE):
                    failures += 1

            # Key DTOU invariant: dtou_total_delta == delivery_delta (supply cancels)
            supply_residual = abs(row_dict["dtou_total_delta_dollars"] - row_dict["delivery_delta_dollars"])
            if supply_residual > TOLERANCE:
                print(f"      FAIL: supply did not cancel — dtou_total_delta - delivery_delta = {supply_residual:.2e}")
                failures += 1
            else:
                print(f"      Supply cancellation check: residual={supply_residual:.2e}  PASS")

    return failures


# ── Step 5: Verify YAML rate loading ─────────────────────────────────────


def verify_yaml_rates() -> int:
    """Import _load_stou_supply_rates and verify against known constants."""
    print("\n" + "=" * 60)
    print("STEP 5: Verify YAML rate loading")
    print("=" * 60)

    # Import the function from the pipeline module
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "pricing_pilot"))
    from compute_delivery_deltas import STOU_YAML_PATH, _load_stou_supply_rates

    rates = _load_stou_supply_rates(STOU_YAML_PATH)
    print(f"  Loaded rates from: {STOU_YAML_PATH}")
    print(f"  Seasons: {list(rates.keys())}")

    failures = 0

    for season in ("summer", "nonsummer"):
        if season not in rates:
            print(f"  FAIL: missing season '{season}'")
            failures += 1
            continue

        print(f"\n  {season}:")
        for period in ("morning", "midday_peak", "evening", "overnight"):
            if period not in rates[season]:
                print(f"    FAIL: missing period '{period}'")
                failures += 1
                continue

            actual = rates[season][period]
            expected_val = STOU_SUPPLY[season][period]
            match = "PASS" if abs(actual - expected_val) < 1e-10 else "FAIL"
            print(f"    {period:15s}  expected={expected_val:.3f}  actual={actual:.3f}  {match}")
            if match == "FAIL":
                failures += 1

    # Also verify we get exactly the expected periods (no extras)
    for season in rates:
        extra = set(rates[season].keys()) - {"morning", "midday_peak", "evening", "overnight"}
        if extra:
            print(f"  FAIL: unexpected periods in {season}: {extra}")
            failures += 1

    return failures


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    """Run the full audit end-to-end."""
    print("=" * 60)
    print("PIPELINE AUDIT: Synthetic data end-to-end verification")
    print("=" * 60)

    # Step 1: Generate synthetic data
    generate_synthetic_data()

    # Step 2: Hand-calculate expected values
    expected = hand_calculate_expected()

    # Step 3: Run the actual pipeline
    run_pipeline()

    # Step 4: Compare actual vs expected
    stou_failures = compare_stou_output(expected)
    dtou_failures = compare_dtou_output(expected)

    # Step 5: Verify YAML rate loading
    yaml_failures = verify_yaml_rates()

    # ── Summary ──
    total_failures = stou_failures + dtou_failures + yaml_failures
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  STOU comparison failures: {stou_failures}")
    print(f"  DTOU comparison failures: {dtou_failures}")
    print(f"  YAML rate check failures: {yaml_failures}")
    print(f"  Total failures:           {total_failures}")

    if total_failures == 0:
        print("\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {total_failures} CHECK(S) FAILED")

    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
