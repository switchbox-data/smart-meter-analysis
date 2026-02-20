#!/usr/bin/env python3
"""Audit: verify STOU and DTOU calendar parquets assign identical TOU periods.

STOU (Space-Time-of-Use) and DTOU (Dynamic Time-of-Use) use the same four
TOU period windows (morning, midday_peak, evening, overnight) — only prices
differ.  This script loads both calendar parquets and fails loudly if any
datetime_chicago maps to a different period label in the two files.

The tariff calendars are produced by scripts/build_tariff_hourly_prices.py;
each has a ``period`` column and a ``datetime_chicago`` join key.

Usage (EC2, pointing at the "allin" calendar parquets)::

    uv run python scripts/tmp/audit_stou_dtou_windows.py \\
        --stou ~/pricing_pilot/comed_stou_allin_202307_sf_no_esh.parquet \\
        --dtou ~/pricing_pilot/comed_dtou_allin_202307_sf_no_esh.parquet

    # Or compare entire directories of calendar parquets (one per file)
    uv run python scripts/tmp/audit_stou_dtou_windows.py \\
        --stou ~/pricing_pilot/stou_calendars/ \\
        --dtou ~/pricing_pilot/dtou_calendars/

Exit codes:
    0 — all period assignments match
    1 — at least one mismatch (details printed to stdout)
    2 — input error (file not found, missing column, row-count mismatch)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

PERIOD_COL = "period"
KEY_COL = "datetime_chicago"

REQUIRED_COLS = {KEY_COL, PERIOD_COL}


def _load(path: Path) -> pl.DataFrame:
    """Load a single parquet or scan a directory of parquets."""
    if path.is_dir():
        parts = sorted(path.glob("*.parquet"))
        if not parts:
            print(f"ERROR: no parquet files in directory: {path}", file=sys.stderr)
            sys.exit(2)
        df = pl.read_parquet(parts)
    elif path.is_file():
        df = pl.read_parquet(path)
    else:
        print(f"ERROR: path not found: {path}", file=sys.stderr)
        sys.exit(2)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(
            f"ERROR: {path} is missing required columns: {sorted(missing)}. Found: {sorted(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(2)

    return df.select([KEY_COL, PERIOD_COL]).sort(KEY_COL)


def audit(stou_path: Path, dtou_path: Path) -> int:
    """Compare period columns; return 0 on match, 1 on mismatch."""
    stou = _load(stou_path)
    dtou = _load(dtou_path)

    print(f"STOU: {stou.height} rows  ({stou_path})")
    print(f"DTOU: {dtou.height} rows  ({dtou_path})")

    if stou.height != dtou.height:
        print(
            f"ERROR: row count mismatch — STOU={stou.height}, DTOU={dtou.height}. "
            "Calendars must cover the same datetime range.",
            file=sys.stderr,
        )
        return 2

    # Join on datetime_chicago; unmatched → null period_dtou
    joined = stou.join(
        dtou.rename({PERIOD_COL: "period_dtou"}),
        on=KEY_COL,
        how="left",
        suffix="_dtou",
    ).rename({PERIOD_COL: "period_stou"})

    # Rows where keys don't align (nulls in period_dtou after left join)
    n_unjoined = joined.filter(pl.col("period_dtou").is_null()).height
    if n_unjoined:
        sample = joined.filter(pl.col("period_dtou").is_null()).head(5)[KEY_COL].to_list()
        print(
            f"ERROR: {n_unjoined} datetime(s) in STOU have no match in DTOU. Sample: {sample}",
            file=sys.stderr,
        )
        return 2

    mismatches = joined.filter(pl.col("period_stou") != pl.col("period_dtou"))
    n_mismatch = mismatches.height
    n_total = joined.height

    if n_mismatch == 0:
        print(f"OK — all {n_total} period assignments match between STOU and DTOU.")
        # Print period distribution as a quick sanity check
        dist = stou.group_by(PERIOD_COL).len().sort(PERIOD_COL).rename({"len": "hours"})
        print("\nPeriod distribution (STOU):")
        print(dist)
        return 0

    print(f"\nFAIL — {n_mismatch}/{n_total} period assignments differ between STOU and DTOU:\n")
    print(mismatches.head(20))

    # Summary by (stou_period, dtou_period) pair
    summary = (
        mismatches.group_by(["period_stou", "period_dtou"])
        .len()
        .sort(["period_stou", "period_dtou"])
        .rename({"len": "n_rows"})
    )
    print("\nMismatch summary (by period pair):")
    print(summary)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify STOU and DTOU TOU period assignments are identical.",
    )
    parser.add_argument(
        "--stou",
        type=Path,
        required=True,
        metavar="PATH",
        help="STOU calendar parquet file or directory of parquets.",
    )
    parser.add_argument(
        "--dtou",
        type=Path,
        required=True,
        metavar="PATH",
        help="DTOU calendar parquet file or directory of parquets.",
    )
    args = parser.parse_args(argv)
    return audit(args.stou, args.dtou)


if __name__ == "__main__":
    sys.exit(main())
