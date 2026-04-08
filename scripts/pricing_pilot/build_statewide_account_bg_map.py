#!/usr/bin/env python3
"""Build statewide account→block-group map from combined billing Parquets.

Unlike build_account_bg_map.py (which reads raw interval Parquets for a single
month), this script reads the already-combined STOU/DTOU billing Parquets that
already contain account_identifier and zip_code columns.  This avoids scanning
hundreds of GB of interval data.

Pipeline
--------
1. Read account_identifier and zip_code from four combined Parquet files:
     {billing-output-dir}/stou_combined/stou_combined_202301.parquet
     {billing-output-dir}/stou_combined/stou_combined_202307.parquet
     {billing-output-dir}/dtou_combined/dtou_combined_202301.parquet
     {billing-output-dir}/dtou_combined/dtou_combined_202307.parquet
2. Deduplicate to unique (account_identifier, zip_code) pairs.
3. Normalise zip_code → zip4 (#####-####).
4. Load ZIP+4→BG crosswalk and resolve fan-out with min(geoid_bg) per zip4,
   reusing the exact crosswalk logic from build_account_bg_map.py.
5. Left-join accounts to crosswalk on zip4.
6. Resolve accounts with multiple BGs via min(geoid_bg) per account.
7. Assert strict 1:1 mapping; abort if unmatched rate exceeds 5%.
8. Write {billing-output-dir}/account_bg_map_statewide.parquet.

Usage::

    python scripts/pricing_pilot/build_statewide_account_bg_map.py \\
        --billing-output-dir /ebs/home/griffin_switch_box/runs/billing_output \\
        --crosswalk data/reference/comed_bg_zip4_crosswalk.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# Expected combined Parquet files (relative to --billing-output-dir)
INPUT_FILES = [
    "stou_combined/stou_combined_202301.parquet",
    "stou_combined/stou_combined_202307.parquet",
    "dtou_combined/dtou_combined_202301.parquet",
    "dtou_combined/dtou_combined_202307.parquet",
]


# ---------------------------------------------------------------------------
# ZIP normalisation (mirrors build_account_bg_map.py/_normalize_zip4_expr)
# ---------------------------------------------------------------------------


def _normalize_zip4_expr() -> pl.Expr:
    """Polars expression: derive zip4 (#####-####) from zip_code.

    Handles both ``#####-####`` (already correct) and ``#########`` (9-digit).
    Mirrors build_account_bg_map.py/_normalize_zip4_expr() exactly so the
    join key is identical across all pipeline scripts.
    """
    raw = pl.col("zip_code").cast(pl.Utf8).str.strip_chars()
    return (
        pl.when(raw.str.contains(r"^\d{5}-\d{4}$"))
        .then(raw)
        .when(raw.str.contains(r"^\d{9}$"))
        .then(raw.str.slice(0, 5) + pl.lit("-") + raw.str.slice(5, 4))
        .otherwise(pl.lit(None))
        .alias("zip4")
    )


# ---------------------------------------------------------------------------
# Crosswalk loading (mirrors build_account_bg_map.py/_load_crosswalk)
# ---------------------------------------------------------------------------


def _load_crosswalk(crosswalk_path: Path) -> pl.DataFrame:
    """Load ZIP+4→BG crosswalk with deterministic 1:1 linkage.

    Fan-out (one zip4 → multiple BGs) is resolved by taking the smallest
    ``geoid_bg`` per ``zip4``.  This is arbitrary but reproducible and
    matches build_account_bg_map.py and the R stage-2 scripts exactly.

    Returns a collected DataFrame with columns ``zip4`` and ``geoid_bg``.
    """
    lf = (
        pl.scan_csv(crosswalk_path, separator="\t", infer_schema_length=10_000)
        .with_columns(
            (pl.col("Zip").cast(pl.Utf8).str.zfill(5) + pl.lit("-") + pl.col("Zip4").cast(pl.Utf8).str.zfill(4)).alias(
                "zip4"
            ),
            pl.col("CensusKey2023").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("geoid_bg"),
        )
        .select("zip4", "geoid_bg")
        .drop_nulls()
    )

    n_rows, n_zip4, n_bg = (
        lf.select(pl.len(), pl.col("zip4").n_unique(), pl.col("geoid_bg").n_unique()).collect().row(0)
    )
    log.info("Crosswalk: %d rows, %d unique zip4s, %d unique block groups", n_rows, n_zip4, n_bg)

    fanout = (
        lf.group_by("zip4")
        .agg(pl.col("geoid_bg").n_unique().alias("n_bg"))
        .filter(pl.col("n_bg") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    if fanout:
        log.warning(
            "Crosswalk fan-out: %d zip4(s) map to multiple block groups; resolving with min(geoid_bg) per zip4.",
            fanout,
        )

    mapping = lf.group_by("zip4").agg(pl.col("geoid_bg").min()).collect()
    log.info("Crosswalk 1:1 mapping: %d zip4 → geoid_bg entries", mapping.height)
    return mapping


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def main() -> int:
    default_crosswalk = REPO_ROOT / "data/reference/comed_bg_zip4_crosswalk.txt"

    parser = argparse.ArgumentParser(
        description="Build statewide account→block-group map from combined billing Parquets.",
    )
    parser.add_argument(
        "--billing-output-dir",
        type=Path,
        required=True,
        help="Root of billing output directory containing stou_combined/ and dtou_combined/ subdirectories.",
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=default_crosswalk,
        help=f"ZIP+4→BG crosswalk TSV (default: {default_crosswalk}).",
    )
    parser.add_argument(
        "--max-drop-pct",
        type=float,
        default=5.0,
        help="Abort if more than this %% of accounts have no BG match (default: 5.0).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate input paths
    # ------------------------------------------------------------------
    if not args.crosswalk.exists():
        log.error("--crosswalk not found: %s", args.crosswalk)
        return 1
    if not args.billing_output_dir.exists():
        log.error("--billing-output-dir not found: %s", args.billing_output_dir)
        return 1

    input_paths: list[Path] = []
    for rel in INPUT_FILES:
        p = args.billing_output_dir / rel
        if not p.exists():
            log.error("Input file not found: %s", p)
            return 1
        input_paths.append(p)
    log.info("Reading %d combined Parquet files from %s", len(input_paths), args.billing_output_dir)

    # ------------------------------------------------------------------
    # 1. Read account_identifier + zip_code from all four files
    # ------------------------------------------------------------------
    frames: list[pl.LazyFrame] = []
    for p in input_paths:
        lf = pl.scan_parquet(p).select("account_identifier", "zip_code")
        frames.append(lf)
    pairs = pl.concat(frames).unique(subset=["account_identifier", "zip_code"]).collect()

    n_pairs = pairs.height
    n_accounts = pairs.select(pl.col("account_identifier").n_unique()).item()
    log.info("Total unique accounts across all input files: %d", n_accounts)
    log.info("Unique (account, zip_code) pairs: %d", n_pairs)

    if n_accounts == 0:
        log.error("No accounts found in input files.")
        return 1

    # ------------------------------------------------------------------
    # 2. Normalise zip_code → zip4 (#####-####)
    # ------------------------------------------------------------------
    pairs = pairs.with_columns(_normalize_zip4_expr())
    n_null_zip4 = pairs.filter(pl.col("zip4").is_null()).height
    if n_null_zip4:
        log.warning(
            "%d (account, zip_code) pair(s) produced a null zip4 (unparseable format); dropping.",
            n_null_zip4,
        )
        pairs = pairs.drop_nulls("zip4")

    # ------------------------------------------------------------------
    # 3. Load crosswalk (1:1 zip4→geoid_bg via min tie-break)
    # ------------------------------------------------------------------
    crosswalk = _load_crosswalk(args.crosswalk)

    # ------------------------------------------------------------------
    # 4. Left-join accounts to crosswalk on zip4
    # ------------------------------------------------------------------
    joined = pairs.join(crosswalk, on="zip4", how="left")

    n_matched_pairs = joined.filter(pl.col("geoid_bg").is_not_null()).height
    n_unmatched_pairs = joined.height - n_matched_pairs
    log.info("Crosswalk join: %d matched, %d unmatched pairs.", n_matched_pairs, n_unmatched_pairs)

    # ------------------------------------------------------------------
    # 5. Check unmatched accounts
    # ------------------------------------------------------------------
    matched_accounts: set[str] = set(
        joined.filter(pl.col("geoid_bg").is_not_null()).get_column("account_identifier").to_list()
    )
    all_accounts: set[str] = set(joined.get_column("account_identifier").to_list())
    no_bg_accounts = sorted(all_accounts - matched_accounts)
    n_no_bg = len(no_bg_accounts)

    if n_no_bg:
        pct_no_bg = n_no_bg / len(all_accounts) * 100
        log.warning(
            "%d account(s) (%.2f%%) have no BG match. Sample: %s",
            n_no_bg,
            pct_no_bg,
            no_bg_accounts[:10],
        )
        if pct_no_bg > args.max_drop_pct:
            log.error(
                "Unmatched account rate %.2f%% exceeds --max-drop-pct %.2f%%. Aborting.",
                pct_no_bg,
                args.max_drop_pct,
            )
            return 1
    else:
        log.info("All accounts matched to at least one BG.")

    # ------------------------------------------------------------------
    # 6. Resolve accounts with multiple BGs → min(geoid_bg)
    # ------------------------------------------------------------------
    mapping = joined.drop_nulls("geoid_bg").group_by("account_identifier").agg(pl.col("geoid_bg").min())

    # ------------------------------------------------------------------
    # 7. Assert strict 1:1 mapping
    # ------------------------------------------------------------------
    n_unique_acct = mapping.select(pl.col("account_identifier").n_unique()).item()
    if mapping.height != n_unique_acct:
        raise RuntimeError(f"BUG: mapping is not 1:1 — {mapping.height} rows but {n_unique_acct} unique accounts.")
    n_null_geoid = mapping.filter(pl.col("geoid_bg").is_null()).height
    if n_null_geoid:
        raise RuntimeError(f"BUG: {n_null_geoid} null geoid_bg value(s) in final mapping.")

    # ------------------------------------------------------------------
    # 8. Validation summary
    # ------------------------------------------------------------------
    n_mapped = mapping.height
    n_bg_out = mapping.select(pl.col("geoid_bg").n_unique()).item()
    match_rate = n_mapped / n_accounts * 100

    print(f"\n{'=' * 50}")
    print("Statewide account→BG map — validation summary")
    print(f"{'=' * 50}")
    print(f"Total unique accounts across input files: {n_accounts:,}")
    print(f"Crosswalk rows loaded:                    {crosswalk.height:,}")
    print(f"Crosswalk unique zip4 values:             {crosswalk.select(pl.col('zip4').n_unique()).item():,}")
    print(f"Match count:                              {n_mapped:,}")
    print(f"Match rate:                               {match_rate:.2f}%")
    print(f"Unmatched count:                          {n_no_bg:,}")
    print(f"Unique block groups in output:            {n_bg_out:,}")
    print(f"Output row count:                         {n_mapped:,}")

    # ------------------------------------------------------------------
    # 9. Write output
    # ------------------------------------------------------------------
    out_path = args.billing_output_dir / "account_bg_map_statewide.parquet"
    mapping.select("account_identifier", "geoid_bg").write_parquet(out_path)
    log.info("Wrote %s (%d rows).", out_path, n_mapped)
    print(f"Output: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
