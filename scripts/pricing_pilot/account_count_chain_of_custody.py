#!/usr/bin/env python3
"""Account count chain-of-custody check across all delivery classes.

Verify no accounts are dropped between raw interval data and bill outputs.
For each of the four delivery classes (C23=sf_no_esh, C24=mf_no_esh, C26=sf_esh, C28=mf_esh),
compare distinct account counts at each pipeline stage for January and July 2023.

Stages:
  1) Raw interval data: ~/pricing_pilot/interval_data/year=2023/month=01|07/
  2) Bill files: ~/pricing_pilot/bills_unscaled/ (DTOU and STOU per class; flag if counts differ)
  3) Account-BG map: join bills to ~/pricing_pilot/account_bg_map_{yyyymm}.parquet

Output: Summary table + sample dropped account IDs when counts drop between stages.

Usage:
  uv run python scripts/pricing_pilot/account_count_chain_of_custody.py
  uv run python scripts/pricing_pilot/account_count_chain_of_custody.py --data-root ~/pricing_pilot/interval_data --bills-dir ~/pricing_pilot/bills_unscaled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# Delivery class: code (raw) <-> filename suffix (bills)
DELIVERY_CLASSES = ["C23", "C24", "C26", "C28"]  # sf_no_esh, mf_no_esh, sf_esh, mf_esh
CODE_TO_SUFFIX = {
    "C23": "sf_no_esh",
    "C24": "mf_no_esh",
    "C26": "sf_esh",
    "C28": "mf_esh",
}
SUFFIX_TO_CODE = {v: k for k, v in CODE_TO_SUFFIX.items()}

MONTHS = [("2023", "01", "202301"), ("2023", "07", "202307")]
MIN_PART_BYTES = 20 * 1024 * 1024  # match compute_household_bills_bounded stub filter


def _filter_valid_parquet(paths: list[Path], min_bytes: int) -> list[Path]:
    return [p for p in paths if p.stat().st_size >= min_bytes]


def stage1_raw_counts(data_root: Path) -> dict[tuple[str, str], int]:
    """Stage 1: distinct accounts per (month_yyyymm, delivery_class) from raw interval parquets."""
    class_col = "delivery_service_class"
    account_col = "account_identifier"
    out: dict[tuple[str, str], int] = {}

    for year, month, yyyymm in MONTHS:
        dir_path = data_root / f"year={year}" / f"month={month}"
        if not dir_path.exists():
            print(f"[Stage 1] Directory not found: {dir_path}", file=sys.stderr)
            for dc in DELIVERY_CLASSES:
                out[(yyyymm, dc)] = 0
            continue
        parts = sorted(dir_path.glob("*.parquet"))
        parts = _filter_valid_parquet(parts, MIN_PART_BYTES)
        if not parts:
            print(f"[Stage 1] No valid parquet parts in {dir_path} (min {MIN_PART_BYTES} bytes)", file=sys.stderr)
            for dc in DELIVERY_CLASSES:
                out[(yyyymm, dc)] = 0
            continue

        first = pl.scan_parquet(parts[0])
        schema_names = first.collect_schema().names()
        print(f"[Stage 1] Raw interval schema ({yyyymm}): {schema_names}")
        if class_col not in schema_names:
            print(f"[Stage 1] WARNING: column '{class_col}' not in schema for {yyyymm}", file=sys.stderr)

        # Single lazy scan across all parts — Polars pushes unique + group_by down
        print(f"[Stage 1] Counting distinct accounts for {yyyymm} ({len(parts)} parts)...", flush=True)
        counts = (
            pl.scan_parquet(parts)
            .select(pl.col(account_col), pl.col(class_col))
            .group_by(class_col)
            .agg(pl.col(account_col).n_unique().alias("n"))
            .collect()
        )
        for row in counts.iter_rows(named=True):
            dc = str(row[class_col]).strip()
            if dc in DELIVERY_CLASSES:
                out[(yyyymm, dc)] = row["n"]
        for dc in DELIVERY_CLASSES:
            if (yyyymm, dc) not in out:
                out[(yyyymm, dc)] = 0
    return out


def _parse_bill_path(path: Path) -> tuple[str, str, str] | None:
    """(yyyymm, 'dtou'|'stou', delivery_class_suffix) or None."""
    stem = path.stem
    if "_flat_vs_" not in stem:
        return None
    month = stem[:6]
    if not month.isdigit() or len(month) != 6:
        return None
    rate = "dtou" if "dtou" in stem else "stou"
    for suffix in CODE_TO_SUFFIX.values():
        if stem.endswith(suffix):
            return (month, rate, suffix)
    return None


def stage2_bill_counts(bills_dir: Path) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int], list[str]]:
    """Stage 2: distinct accounts per (month, delivery_class) for DTOU and STOU. Returns (dtou, stou, flags)."""
    account_col = "account_identifier"
    dtou_counts: dict[tuple[str, str], int] = {}
    stou_counts: dict[tuple[str, str], int] = {}
    flags: list[str] = []

    paths = sorted(bills_dir.glob("*_flat_vs_*_*.parquet"))
    for path in paths:
        parsed = _parse_bill_path(path)
        if not parsed:
            continue
        yyyymm, rate, suffix = parsed
        dc = SUFFIX_TO_CODE[suffix]
        n = pl.scan_parquet(path).select(pl.col(account_col).n_unique()).collect().item()
        if rate == "dtou":
            dtou_counts[(yyyymm, dc)] = n
        else:
            stou_counts[(yyyymm, dc)] = n

    for yyyymm, dc in set(dtou_counts) | set(stou_counts):
        key = (yyyymm, dc)
        d = dtou_counts.get(key, 0)
        s = stou_counts.get(key, 0)
        if d != s:
            flags.append(f"DTOU vs STOU count mismatch: {yyyymm} {dc} -> DTOU={d}, STOU={s}")

    return dtou_counts, stou_counts, flags


def stage3_bg_join_counts(
    bills_dir: Path,
    map_pattern: str,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Stage 3: for each bill file, join to account_bg_map; count with/without geoid_bg match."""
    account_col = "account_identifier"
    with_bg: dict[tuple[str, str], int] = {}
    without_bg: dict[tuple[str, str], int] = {}

    paths = sorted(bills_dir.glob("*_flat_vs_dtou_*.parquet"))  # one rate is enough for account set per class
    for path in paths:
        parsed = _parse_bill_path(path)
        if not parsed:
            continue
        yyyymm, _, suffix = parsed
        dc = SUFFIX_TO_CODE[suffix]
        map_path = Path(map_pattern.replace("{yyyymm}", yyyymm))
        if not map_path.exists():
            print(f"[Stage 3] Map not found: {map_path}", file=sys.stderr)
            with_bg[(yyyymm, dc)] = 0
            without_bg[(yyyymm, dc)] = 0
            continue
        bills = pl.scan_parquet(path).select(account_col).unique().collect()
        amap = pl.read_parquet(map_path, columns=[account_col, "geoid_bg"])
        joined = bills.join(amap, on=account_col, how="left")
        n_with = joined.filter(pl.col("geoid_bg").is_not_null()).height
        n_without = joined.filter(pl.col("geoid_bg").is_null()).height
        with_bg[(yyyymm, dc)] = n_with
        without_bg[(yyyymm, dc)] = n_without

    return with_bg, without_bg


def sample_missing(
    earlier_set: set[str],
    later_set: set[str],
    n: int = 5,
) -> list[str]:
    """Return up to n account IDs that are in earlier_set but not in later_set."""
    missing = list(earlier_set - later_set)[:n]
    return missing


def main() -> int:
    default_root = Path.home() / "pricing_pilot" / "interval_data"
    default_bills = Path.home() / "pricing_pilot" / "bills_unscaled"
    default_map = str(Path.home() / "pricing_pilot" / "account_bg_map_{yyyymm}.parquet")

    parser = argparse.ArgumentParser(description="Account count chain-of-custody across delivery classes.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help="Root of Hive-partitioned interval parquets (default: ~/pricing_pilot/interval_data). Layout: <root>/year=YYYY/month=MM/*.parquet",
    )
    parser.add_argument("--bills-dir", type=Path, default=default_bills, help="Directory of bill parquets")
    parser.add_argument(
        "--account-bg-map-pattern", type=str, default=default_map, help="Path with {yyyymm} for account_bg_map"
    )
    args = parser.parse_args()

    data_root = args.data_root
    bills_dir = args.bills_dir
    map_pattern = args.account_bg_map_pattern

    if not bills_dir.exists():
        print(f"Bills directory not found: {bills_dir}", file=sys.stderr)
        return 1

    print("=== Stage 1: Raw interval data (distinct accounts per delivery class) ===")
    raw = stage1_raw_counts(data_root)

    print("\n=== Stage 2: Bill files (DTOU and STOU distinct accounts per class) ===")
    dtou, stou, flags = stage2_bill_counts(bills_dir)
    for f in flags:
        print(f"  FLAG: {f}")

    print("\n=== Stage 3: Join to account_bg_map (with vs without geoid_bg) ===")
    with_bg, without_bg = stage3_bg_join_counts(bills_dir, map_pattern)

    # Build summary table: month, delivery_class, raw_interval_accounts, bill_dtou_accounts, bill_stou_accounts, accounts_with_bg_match, accounts_without_bg_match
    rows = []
    for yyyymm in ["202301", "202307"]:
        for dc in DELIVERY_CLASSES:
            rows.append({
                "month": yyyymm,
                "delivery_class": dc,
                "raw_interval_accounts": raw.get((yyyymm, dc), 0),
                "bill_dtou_accounts": dtou.get((yyyymm, dc), 0),
                "bill_stou_accounts": stou.get((yyyymm, dc), 0),
                "accounts_with_bg_match": with_bg.get((yyyymm, dc), 0),
                "accounts_without_bg_match": without_bg.get((yyyymm, dc), 0),
            })
    table = pl.DataFrame(rows)

    print("\n--- Summary table ---")
    print(table)

    # Drops: raw -> bills; bills -> bg_match
    print("\n--- Drops (sample account IDs when count decreases) ---")
    any_drops = False

    # Load account sets for sampling (only where we need them)
    for yyyymm in ["202301", "202307"]:
        map_path = Path(map_pattern.replace("{yyyymm}", yyyymm))
        amap = (
            pl.read_parquet(map_path, columns=["account_identifier", "geoid_bg"])
            if map_path.exists()
            else pl.DataFrame({"account_identifier": [], "geoid_bg": []})
        )

        for dc in DELIVERY_CLASSES:
            r = table.filter((pl.col("month") == yyyymm) & (pl.col("delivery_class") == dc)).to_dicts()[0]
            raw_n = r["raw_interval_accounts"]
            bill_d = r["bill_dtou_accounts"]
            bill_s = r["bill_stou_accounts"]
            without_bg_n = r["accounts_without_bg_match"]

            # Drop from raw to bill (use DTOU as bill reference)
            if raw_n > 0 and bill_d < raw_n:
                any_drops = True
                year, month = yyyymm[:4], yyyymm[4:6]
                dir_path = data_root / f"year={year}" / f"month={month}"
                parts = (
                    _filter_valid_parquet(sorted(dir_path.glob("*.parquet")), MIN_PART_BYTES)
                    if dir_path.exists()
                    else []
                )
                raw_accounts = set()
                if parts:
                    raw_df = (
                        pl.scan_parquet(parts)
                        .filter(pl.col("delivery_service_class") == dc)
                        .select("account_identifier")
                        .unique()
                        .collect()
                    )
                    raw_accounts = set(raw_df["account_identifier"].cast(pl.Utf8).to_list())
                bill_path = bills_dir / f"{yyyymm}_flat_vs_dtou_{CODE_TO_SUFFIX[dc]}.parquet"
                bill_accounts = set()
                if bill_path.exists():
                    bill_accounts = set(
                        pl.scan_parquet(bill_path)
                        .select(pl.col("account_identifier").cast(pl.Utf8))
                        .unique()
                        .collect()
                        .to_series()
                        .to_list()
                    )
                samples = sample_missing(raw_accounts, bill_accounts, 5)
                print(
                    f"  {yyyymm} {dc}: raw ({raw_n}) -> bill ({bill_d}); sample accounts in raw but not in bills: {samples}"
                )

            # Drop from bill to bg_match (accounts in bills without geoid_bg)
            if (bill_d > 0 or bill_s > 0) and without_bg_n > 0:
                any_drops = True
                bill_path = bills_dir / f"{yyyymm}_flat_vs_dtou_{CODE_TO_SUFFIX[dc]}.parquet"
                if bill_path.exists():
                    bills_df = pl.scan_parquet(bill_path).select("account_identifier").collect()
                    joined = bills_df.join(amap, on="account_identifier", how="left")
                    no_bg = (
                        joined.filter(pl.col("geoid_bg").is_null())
                        .select("account_identifier")
                        .to_series()
                        .cast(pl.Utf8)
                        .to_list()
                    )
                    samples = no_bg[:5]
                    print(
                        f"  {yyyymm} {dc}: bill accounts without BG match: {without_bg_n}; sample account IDs: {samples}"
                    )

    if not any_drops:
        print("  (No drops between stages.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
