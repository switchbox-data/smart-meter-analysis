#!/usr/bin/env python3
"""
End-to-end DST roll-in validation for the RTP billing pipeline.

Validates that the chain:

    interval data → compute_hourly_loads (hour_chicago)
    YAML/CSV      → build_tariff_hourly_prices / build_flat_hourly_prices (datetime_chicago)
    join          → compute_household_bills (hour_chicago = datetime_chicago)

produces correct, unique, fully-covered join keys—including across DST
spring-forward and fall-back transitions.

Timezone note
─────────────
"Chicago" throughout this codebase means the IANA timezone **America/Chicago**
(Central Time: CT = CDT in summer / CST in winter).  It does NOT refer to the
City of Chicago specifically.  The analysis covers the **full ComEd service
territory**, which spans most of northern Illinois and operates entirely within
the America/Chicago timezone.

Assertions
──────────
1. Hourly loads: ``(account_identifier, hour_chicago)`` is unique.
2. Tariffs A and B: ``datetime_chicago`` is unique.
3. Coverage: every ``hour_chicago`` in the loads joins to exactly one row in
   both tariff A and tariff B (no null prices after join).

On failure the script exits non-zero with a message identifying the offending
keys.

Usage as script (exit-code based):

    python tests/test_dst_rollin.py                   # synthetic + sample data
    python tests/test_dst_rollin.py --sample-only      # only real files

Usage as pytest:

    pytest tests/test_dst_rollin.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

# ── Paths to sample artefacts built by the existing pipeline ──────────────
SAMPLE_LOADS = Path("data/reference/hourly_loads_202308_50.parquet")
SAMPLE_TARIFF_STOU = Path("data/reference/comed_stou_hourly_prices_2023.parquet")
SAMPLE_TARIFF_FLAT = Path("data/reference/comed_flat_hourly_prices_2023.parquet")

log = logging.getLogger(__name__)

TZ = "America/Chicago"


# ═══════════════════════════════════════════════════════════════════════════
# Assertion helpers (shared by script mode and pytest)
# ═══════════════════════════════════════════════════════════════════════════


def assert_loads_unique(df: pl.DataFrame) -> None:
    """(account_identifier, hour_chicago) must be unique in hourly loads."""
    dupes = (
        df.group_by(["account_identifier", "hour_chicago"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["account_identifier", "hour_chicago"])
    )
    if dupes.height > 0:
        sample = dupes.head(20)
        raise AssertionError(
            f"Hourly loads: {dupes.height} duplicate (account_identifier, hour_chicago) keys.\n"
            f"First offenders:\n{sample}"
        )
    log.info("  PASS  loads uniqueness: %d rows, 0 duplicates.", df.height)


def assert_tariff_unique(df: pl.DataFrame, label: str) -> None:
    """datetime_chicago must be unique in a tariff price calendar."""
    dupes = df.group_by("datetime_chicago").len().filter(pl.col("len") > 1).sort("datetime_chicago")
    if dupes.height > 0:
        sample = dupes.head(20)
        raise AssertionError(
            f"Tariff {label}: {dupes.height} duplicate datetime_chicago values.\nFirst offenders:\n{sample}"
        )
    log.info(
        "  PASS  tariff %s uniqueness: %d rows, 0 duplicates.",
        label,
        df.height,
    )


def assert_coverage(
    loads: pl.DataFrame,
    tariff: pl.DataFrame,
    label: str,
) -> None:
    """Every hour_chicago in loads must match exactly one datetime_chicago in tariff."""
    load_hours = loads.select("hour_chicago").unique()
    tariff_hours = tariff.select("datetime_chicago").unique()

    # Hours present in loads but missing from tariff
    missing = load_hours.join(
        tariff_hours,
        left_on="hour_chicago",
        right_on="datetime_chicago",
        how="anti",
    )
    if missing.height > 0:
        sample = missing.sort("hour_chicago").head(20)
        raise AssertionError(
            f"Coverage gap: {missing.height} hour(s) in loads have no price in tariff {label}.\n"
            f"Missing hours:\n{sample}"
        )

    # Simulate the actual left join and check for nulls
    joined = loads.join(
        tariff.select("datetime_chicago", "price_cents_per_kwh"),
        left_on="hour_chicago",
        right_on="datetime_chicago",
        how="left",
    )
    null_prices = joined.filter(pl.col("price_cents_per_kwh").is_null()).height
    if null_prices > 0:
        unmatched = (
            joined.filter(pl.col("price_cents_per_kwh").is_null())
            .select("hour_chicago")
            .unique()
            .sort("hour_chicago")
            .head(20)
        )
        raise AssertionError(
            f"Tariff {label}: {null_prices} load rows produced null prices after join.\nUnmatched hours:\n{unmatched}"
        )

    # Check that no row was duplicated (tariff uniqueness should prevent this,
    # but belt-and-suspenders)
    if joined.height != loads.height:
        raise AssertionError(
            f"Tariff {label}: join changed row count {loads.height} → {joined.height}. "
            "Tariff may have duplicate datetime_chicago values."
        )

    log.info(
        "  PASS  coverage tariff %s: %d unique load hours all matched.",
        label,
        load_hours.height,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data builders (for DST edge-case testing)
# ═══════════════════════════════════════════════════════════════════════════


def _make_tariff_for_month(year: int, month: int) -> pl.DataFrame:
    """Build a minimal hourly tariff calendar for a single month.

    Handles DST exactly the same way as build_tariff_hourly_prices.py:
    tz-aware range → strip to naive → deduplicate fall-back duplicates.
    """
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    dt_range = pl.datetime_range(
        start,
        end,
        interval="1h",
        time_zone=TZ,
        eager=True,
        closed="left",
    )
    df = pl.DataFrame({"datetime_aware": dt_range})

    df = df.with_columns(
        pl.col("datetime_aware").dt.replace_time_zone(None).alias("datetime_chicago"),
    )

    # Deduplicate fall-back (same logic as build_tariff_hourly_prices.py)
    df = df.sort("datetime_aware").unique(subset=["datetime_chicago"], keep="first")
    df = df.drop("datetime_aware")

    # Assign a dummy price so coverage checks can verify non-null join
    df = df.with_columns(pl.lit(5.0).alias("price_cents_per_kwh"))
    return df.sort("datetime_chicago")


def _make_interval_data(year: int, month: int, n_accounts: int = 3) -> pl.DataFrame:
    """Build synthetic 30-minute interval data for *n_accounts* over one month.

    Mimics the naive-local datetime column that compute_hourly_loads expects.
    Includes every half-hour in the month using the tz-aware → naive approach
    so that DST transitions are correctly represented.
    """
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    dt_range = pl.datetime_range(
        start,
        end,
        interval="30m",
        time_zone=TZ,
        eager=True,
        closed="left",
    )
    df_times = pl.DataFrame({"datetime_aware": dt_range})
    df_times = df_times.with_columns(
        pl.col("datetime_aware").dt.replace_time_zone(None).alias("datetime"),
    )
    # During fall-back, two aware instants map to the same naive value.
    # Real meter data would show two intervals at e.g. 01:00 and 01:30 (CDT)
    # and then *again* at 01:00 and 01:30 (CST) — all naive-identical.
    # We keep all rows here to let compute_hourly_loads aggregate them.
    df_times = df_times.drop("datetime_aware")

    accounts = [f"ACCT_{i:04d}" for i in range(n_accounts)]
    df_accts = pl.DataFrame({
        "account_identifier": accounts,
        "zip_code": [f"6000{i}" for i in range(n_accounts)],
    })

    # Cross join: every account x every interval
    df = df_times.join(df_accts, how="cross")
    df = df.with_columns(pl.lit(0.5).alias("kwh"))
    return df


def _aggregate_hourly(df: pl.DataFrame) -> pl.DataFrame:
    """Replicate the aggregation logic from compute_hourly_loads.py."""
    return (
        df.with_columns(
            pl.col("datetime").dt.truncate("1h").alias("hour_chicago"),
        )
        .group_by(["account_identifier", "zip_code", "hour_chicago"])
        .agg(pl.col("kwh").sum().alias("kwh_hour"))
        .sort(["account_identifier", "hour_chicago"])
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pytest tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDSTRollInSynthetic:
    """Synthetic end-to-end checks for DST edge-case months."""

    # -- November 2023: fall-back (2 AM → 1 AM on Nov 5) --------------------

    def test_november_loads_unique(self) -> None:
        intervals = _make_interval_data(2023, 11)
        loads = _aggregate_hourly(intervals)
        assert_loads_unique(loads)

    def test_november_tariff_unique(self) -> None:
        tariff = _make_tariff_for_month(2023, 11)
        assert_tariff_unique(tariff, "synthetic_nov")

    def test_november_coverage(self) -> None:
        intervals = _make_interval_data(2023, 11)
        loads = _aggregate_hourly(intervals)
        tariff = _make_tariff_for_month(2023, 11)
        assert_coverage(loads, tariff, "synthetic_nov")

    def test_november_fall_back_hour_count(self) -> None:
        """November tariff should have 720 hours (30 days x 24), NOT 721.

        The fall-back duplicate (1 AM CST) is dropped so that
        datetime_chicago stays unique.
        """
        tariff = _make_tariff_for_month(2023, 11)
        assert tariff.height == 30 * 24, f"Expected 720 hours for Nov 2023 (fall-back deduped), got {tariff.height}"

    # -- March 2023: spring-forward (2 AM → 3 AM on Mar 12) ─────────────────

    def test_march_loads_unique(self) -> None:
        intervals = _make_interval_data(2023, 3)
        loads = _aggregate_hourly(intervals)
        assert_loads_unique(loads)

    def test_march_tariff_unique(self) -> None:
        tariff = _make_tariff_for_month(2023, 3)
        assert_tariff_unique(tariff, "synthetic_mar")

    def test_march_coverage(self) -> None:
        intervals = _make_interval_data(2023, 3)
        loads = _aggregate_hourly(intervals)
        tariff = _make_tariff_for_month(2023, 3)
        assert_coverage(loads, tariff, "synthetic_mar")

    def test_march_spring_forward_hour_count(self) -> None:
        """March tariff should have 743 hours (31x24 - 1 skipped)."""
        tariff = _make_tariff_for_month(2023, 3)
        assert tariff.height == 31 * 24 - 1, f"Expected 743 hours for Mar 2023 (spring-forward), got {tariff.height}"

    # -- August 2023: no DST transition (baseline sanity) ───────────────────

    def test_august_loads_unique(self) -> None:
        intervals = _make_interval_data(2023, 8)
        loads = _aggregate_hourly(intervals)
        assert_loads_unique(loads)

    def test_august_tariff_unique(self) -> None:
        tariff = _make_tariff_for_month(2023, 8)
        assert_tariff_unique(tariff, "synthetic_aug")

    def test_august_coverage(self) -> None:
        intervals = _make_interval_data(2023, 8)
        loads = _aggregate_hourly(intervals)
        tariff = _make_tariff_for_month(2023, 8)
        assert_coverage(loads, tariff, "synthetic_aug")

    def test_august_hour_count(self) -> None:
        tariff = _make_tariff_for_month(2023, 8)
        assert tariff.height == 31 * 24


class TestDSTRollInSampleData:
    """Validate the real sample artefacts sitting in data/reference/."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self) -> None:
        for p in (SAMPLE_LOADS, SAMPLE_TARIFF_STOU, SAMPLE_TARIFF_FLAT):
            if not p.exists():
                pytest.skip(f"Sample file not found: {p}")

    @pytest.fixture()
    def loads(self) -> pl.DataFrame:
        return pl.read_parquet(SAMPLE_LOADS)

    @pytest.fixture()
    def tariff_stou(self) -> pl.DataFrame:
        return pl.read_parquet(SAMPLE_TARIFF_STOU)

    @pytest.fixture()
    def tariff_flat(self) -> pl.DataFrame:
        return pl.read_parquet(SAMPLE_TARIFF_FLAT)

    def test_loads_unique(self, loads: pl.DataFrame) -> None:
        assert_loads_unique(loads)

    def test_tariff_stou_unique(self, tariff_stou: pl.DataFrame) -> None:
        assert_tariff_unique(tariff_stou, "STOU")

    def test_tariff_flat_unique(self, tariff_flat: pl.DataFrame) -> None:
        assert_tariff_unique(tariff_flat, "flat")

    def test_coverage_stou(
        self,
        loads: pl.DataFrame,
        tariff_stou: pl.DataFrame,
    ) -> None:
        assert_coverage(loads, tariff_stou, "STOU")

    def test_coverage_flat(
        self,
        loads: pl.DataFrame,
        tariff_flat: pl.DataFrame,
    ) -> None:
        assert_coverage(loads, tariff_flat, "flat")

    def test_full_year_hour_count_stou(self, tariff_stou: pl.DataFrame) -> None:
        """2023 has spring-forward → 8760 - 1 = 8759 unique hours."""
        assert tariff_stou.height == 8759, f"Expected 8759 rows in STOU 2023 tariff, got {tariff_stou.height}"

    def test_full_year_hour_count_flat(self, tariff_flat: pl.DataFrame) -> None:
        assert tariff_flat.height == 8759, f"Expected 8759 rows in flat 2023 tariff, got {tariff_flat.height}"


# ═══════════════════════════════════════════════════════════════════════════
# Script-mode runner (exit non-zero on first failure)
# ═══════════════════════════════════════════════════════════════════════════


def _run_synthetic_checks() -> list[str]:
    """Run synthetic DST month checks, returning failure messages."""
    failures: list[str] = []
    for year, month, label in [
        (2023, 11, "Nov-2023-fall-back"),
        (2023, 3, "Mar-2023-spring-forward"),
        (2023, 8, "Aug-2023-no-DST"),
    ]:
        log.info("── Synthetic %s ──", label)
        try:
            intervals = _make_interval_data(year, month)
            loads = _aggregate_hourly(intervals)
            tariff = _make_tariff_for_month(year, month)
            assert_loads_unique(loads)
            assert_tariff_unique(tariff, label)
            assert_coverage(loads, tariff, label)
        except AssertionError as exc:
            failures.append(f"[{label}] {exc}")
    return failures


def _run_sample_checks() -> list[str]:
    """Validate real sample artefacts in data/reference/."""
    failures: list[str] = []
    log.info("── Sample artefacts (data/reference/) ──")
    for p in (SAMPLE_LOADS, SAMPLE_TARIFF_STOU, SAMPLE_TARIFF_FLAT):
        if not p.exists():
            failures.append(f"Sample file not found: {p}")
            return failures

    loads = pl.read_parquet(SAMPLE_LOADS)
    tariff_stou = pl.read_parquet(SAMPLE_TARIFF_STOU)
    tariff_flat = pl.read_parquet(SAMPLE_TARIFF_FLAT)

    try:
        assert_loads_unique(loads)
    except AssertionError as exc:
        failures.append(f"[sample-loads] {exc}")

    for df, label in [(tariff_stou, "STOU"), (tariff_flat, "flat")]:
        try:
            assert_tariff_unique(df, label)
        except AssertionError as exc:
            failures.append(f"[sample-tariff-{label}] {exc}")

    for df, label in [(tariff_stou, "STOU"), (tariff_flat, "flat")]:
        try:
            assert_coverage(loads, df, label)
        except AssertionError as exc:
            failures.append(f"[sample-coverage-{label}] {exc}")
    return failures


def _run_checks(sample_only: bool = False) -> list[str]:
    """Run all checks and return a list of failure messages (empty = success)."""
    failures: list[str] = []
    if not sample_only:
        failures.extend(_run_synthetic_checks())
    failures.extend(_run_sample_checks())
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate DST roll-in semantics end-to-end for the RTP billing pipeline.\n\n"
            "TIMEZONE NOTE: 'Chicago' means the IANA timezone America/Chicago "
            "(Central Time), NOT the City of Chicago.  The analysis covers the "
            "full ComEd service territory across northern Illinois."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Skip synthetic tests; only validate real sample artefacts.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    print(
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  DST roll-in validation — RTP billing pipeline             ║\n"
        "║                                                            ║\n"
        "║  'Chicago' = America/Chicago timezone (Central Time).      ║\n"
        "║  Analysis covers the full ComEd territory, not just the    ║\n"
        "║  City of Chicago.                                          ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )

    failures = _run_checks(sample_only=args.sample_only)

    if failures:
        print(f"\nFAILED — {len(failures)} check(s):\n", file=sys.stderr)
        for f in failures:
            print(f"  ✗ {f}\n", file=sys.stderr)
        return 1

    log.info("All DST roll-in checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
