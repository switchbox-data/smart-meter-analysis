#!/usr/bin/env python3
"""
Report for ~/pricing_pilot/bills_unscaled/:
1. Per file: row count, mean bill difference, median bill difference.
2. Join to account_bg_map, aggregate mean bill_diff_dollars by block group (geoid_bg).
3. For each (month, delivery_class): Pearson correlation between DTOU and Rate BEST (STOU)
   mean deltas at block group level — report all 8 in a table.
4. For each of the 16 files: share of block groups where mean delta is positive (savings)
   vs negative (losses).

Sign convention (canonical, matches compute_household_bills.py):
  delta = flat_rate - alternative_rate (bill_a - bill_b)
  Positive = customer SAVES under the alternative (TOU is cheaper)
  Negative = customer PAYS MORE under the alternative (TOU is worse)

Usage::

  uv run python scripts/pricing_pilot/bill_stats_and_bg_correlation.py

  # Override paths:
  uv run python scripts/pricing_pilot/bill_stats_and_bg_correlation.py \\
    --bills-dir ~/pricing_pilot/bills_unscaled \\
    --account-bg-map-pattern ~/pricing_pilot/account_bg_map_{yyyymm}.parquet
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import polars as pl

CLASSES_ORDER = ("sf_no_esh", "mf_no_esh", "sf_esh", "mf_esh")


def _add_delta_col(df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """Add a ``delta`` column using the same priority as export_delta_geojson_by_class.

    Priority: bill_diff_dollars → net_bill_diff_dollars → bill_a_dollars - bill_b_dollars.
    Resolved per-file so heterogeneous schemas across files are handled correctly.

    Sign convention: delta = flat - alt (bill_a - bill_b); positive = saves.

    Returns:
        Tuple of (DataFrame with ``delta`` column, source column label).
    """
    cols = df.columns
    if "bill_diff_dollars" in cols:
        return df.with_columns(pl.col("bill_diff_dollars").alias("delta")), "bill_diff_dollars"
    if "net_bill_diff_dollars" in cols:
        return df.with_columns(pl.col("net_bill_diff_dollars").alias("delta")), "net_bill_diff_dollars"
    if "bill_b_dollars" in cols and "bill_a_dollars" in cols:
        return df.with_columns(
            (pl.col("bill_a_dollars") - pl.col("bill_b_dollars")).alias("delta")
        ), "bill_a_dollars - bill_b_dollars"
    raise ValueError(
        f"No delta column in {cols}; expected bill_diff_dollars, "
        "net_bill_diff_dollars, or both of bill_b_dollars/bill_a_dollars"
    )


def parse_name(name: str) -> tuple[str, str, str] | None:
    """Parse {yyyymm}_flat_vs_{dtou|stou}_{delivery_class}.parquet -> (yyyymm, rate, dc) or None."""
    stem = name.replace(".parquet", "")
    month = stem[:6]
    if not month.isdigit() or len(month) != 6 or "_flat_vs_" not in stem:
        return None
    rate = "dtou" if "dtou" in stem else "stou"
    if "mf_esh" in stem:
        dc = "mf_esh"
    elif "mf_no_esh" in stem:
        dc = "mf_no_esh"
    elif "sf_esh" in stem:
        dc = "sf_esh"
    elif "sf_no_esh" in stem:
        dc = "sf_no_esh"
    else:
        return None
    return month, rate, dc


def main() -> int:
    """Run bill stats and block-group correlation report."""
    default_bills_dir = Path.home() / "pricing_pilot" / "bills_unscaled"
    default_map_pattern = str(Path.home() / "pricing_pilot" / "account_bg_map_{yyyymm}.parquet")

    parser = argparse.ArgumentParser(description="Bill stats and block-group correlation report.")
    parser.add_argument(
        "--bills-dir",
        type=Path,
        default=default_bills_dir,
        help=f"Directory containing *_flat_vs_*_*.parquet bill files (default: {default_bills_dir}).",
    )
    parser.add_argument(
        "--account-bg-map-pattern",
        type=str,
        default=default_map_pattern,
        help=(
            "Path with {yyyymm} placeholder for account->BG crosswalk "
            "(default: ~/pricing_pilot/account_bg_map_{yyyymm}.parquet)."
        ),
    )
    args = parser.parse_args()

    bills_dir: Path = args.bills_dir
    map_pattern: str = args.account_bg_map_pattern

    if not bills_dir.exists():
        print(f"Bills directory not found: {bills_dir}", file=sys.stderr)
        return 1

    # Same glob as export_delta_geojson_by_class to avoid ingesting unrelated parquets
    files = sorted(bills_dir.glob("*_flat_vs_*_*.parquet"))
    files = [f for f in files if parse_name(f.name) is not None]
    if not files:
        print(f"No bill files matching *_flat_vs_*_*.parquet in {bills_dir}", file=sys.stderr)
        return 1

    # Cache maps; skips months without a map file instead of raising KeyError
    map_cache: dict[str, pl.DataFrame | None] = {}

    def _load_map(yyyymm: str) -> pl.DataFrame | None:
        if yyyymm not in map_cache:
            p = Path(map_pattern.replace("{yyyymm}", yyyymm))
            if not p.exists():
                print(f"  WARNING: account-BG map not found for {yyyymm}: {p}", file=sys.stderr)
                map_cache[yyyymm] = None
            else:
                map_cache[yyyymm] = pl.read_parquet(p)
        return map_cache[yyyymm]

    # --- 1. Per-file stats ---
    print("=" * 70)
    print("1. PER-FILE STATS (row count, mean bill diff, median bill diff)")
    print("   Delta column resolved per file: bill_diff_dollars → net_bill_diff_dollars → bill_a - bill_b")
    print("=" * 70)

    for f in files:
        df = pl.read_parquet(f)
        n = df.height
        if n == 0:
            print(f"  {f.name}")
            print("    row_count=0, mean_bill_diff=NaN, median_bill_diff=NaN [col: N/A]")
        else:
            df, col_label = _add_delta_col(df)
            mean_d = df["delta"].mean()
            median_d = df["delta"].median()
            print(f"  {f.name}")
            print(f"    row_count={n}, mean_bill_diff={mean_d:.4f}, median_bill_diff={median_d:.4f} [col: {col_label}]")

    # --- 2. Aggregate to block group (mean delta per BG) ---
    print()
    print("=" * 70)
    print("2. BLOCK GROUP MEAN DELTAS (per file, joined with account_bg_map)")
    print("=" * 70)

    bg_means: dict[tuple[str, str, str], pl.DataFrame] = {}

    for f in files:
        parsed = parse_name(f.name)
        if not parsed:
            continue
        month, rate, dc = parsed

        df = pl.read_parquet(f)
        if df.height == 0:
            bg_means[(month, rate, dc)] = pl.DataFrame(schema={"geoid_bg": pl.Utf8, "mean_delta": pl.Float64})
            continue

        m = _load_map(month)
        if m is None:
            continue

        df, _ = _add_delta_col(df)
        bg = (
            df.select("account_identifier", "delta")
            .join(
                m.select("account_identifier", "geoid_bg"),
                on="account_identifier",
                how="inner",
            )
            .group_by("geoid_bg")
            .agg(pl.col("delta").mean().alias("mean_delta"))
        )
        bg_means[(month, rate, dc)] = bg
        print(f"  {f.name} -> {bg.height} block groups")

    # --- 3. Pearson correlation table: DTOU vs Rate BEST (STOU) per (month, delivery_class) ---
    print()
    print("=" * 70)
    print("3. PEARSON CORRELATION: DTOU vs Rate BEST (STOU) mean deltas by block group")
    print("   (2 months x 4 classes = 8 correlations)")
    print("=" * 70)

    corr_rows: list[dict[str, object]] = []
    for month in ("202301", "202307"):
        for dc in CLASSES_ORDER:
            bg_dtou = bg_means.get((month, "dtou", dc))
            bg_stou = bg_means.get((month, "stou", dc))
            if bg_dtou is None or bg_stou is None or bg_dtou.height == 0 or bg_stou.height == 0:
                r: float = float("nan")
                n_bg = 0
            else:
                combined = bg_dtou.rename({"mean_delta": "mean_delta_dtou"}).join(
                    bg_stou.rename({"mean_delta": "mean_delta_stou"}),
                    on="geoid_bg",
                    how="inner",
                )
                n_bg = combined.height
                if n_bg >= 2:
                    r = combined.select(pl.pearson_corr("mean_delta_dtou", "mean_delta_stou")).item()
                else:
                    r = float("nan")
            corr_rows.append({"month": month, "delivery_class": dc, "pearson_r": r, "n_bg": n_bg})

    # Print as pivoted table (month x delivery_class)
    header = f"{'month':<10}" + "".join(f"{dc:>14}" for dc in CLASSES_ORDER)
    print(header)
    for month_label in ("202301", "202307"):
        vals = []
        for dc in CLASSES_ORDER:
            r_val = next(
                row["pearson_r"] for row in corr_rows if row["month"] == month_label and row["delivery_class"] == dc
            )
            vals.append(f"{r_val:.4f}" if not math.isnan(r_val) else "N/A")
        print(f"{month_label:<10}" + "".join(f"{v:>14}" for v in vals))
    print()

    # --- 4. Share of block groups with savings (mean_delta > 0) vs losses (mean_delta < 0) ---
    print("=" * 70)
    print("4. SHARE OF BLOCK GROUPS: positive mean delta (savings) vs negative (losses)")
    print("   Per file (16 files). July all-losses check: DTOU and/or Rate BEST.")
    print("=" * 70)

    for f in files:
        parsed = parse_name(f.name)
        if not parsed:
            continue
        month, rate, dc = parsed
        bg = bg_means.get((month, rate, dc))
        if bg is None or bg.height == 0:
            n_bg = 0
            pct_s = pct_l = "N/A"
        else:
            n_bg = bg.height
            pos = bg.filter(pl.col("mean_delta") > 0).height
            neg = bg.filter(pl.col("mean_delta") < 0).height
            pct_s = f"{100 * pos / n_bg:.2f}%"
            pct_l = f"{100 * neg / n_bg:.2f}%"
        print(f"  {f.name}")
        print(f"    n_bg={n_bg}, share BGs with savings (Δ>0)={pct_s}, share BGs with losses (Δ<0)={pct_l}")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
