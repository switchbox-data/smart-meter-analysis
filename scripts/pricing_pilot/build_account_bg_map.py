#!/usr/bin/env python3
"""Build account→block-group map for the pricing pilot analysis.

Replaces ad-hoc shell one-liners used to generate
~/pricing_pilot/account_bg_map_{yyyymm}.parquet on EC2.

Pipeline
--------
1. Locate canonical long Parquet shards for the given month under a
   Hive-partitioned tree: <parquet-dir>/year=YYYY/month=MM/*.parquet
   (default: ~/pricing_pilot/chicago_only).
2. Apply a defensive Chicago-area ZIP filter using
   data/reference/geography/chicago_zip_codes.csv — the same reference
   file used by debug_check_chicago_presence.py and others.  If the
   input is already the chicago_only subset this is a no-op; it acts
   as a guardrail when pointed at the full service-territory parquets.
3. Extract unique (account_identifier, zip_code) pairs (column-projected
   scan: only these two columns are read from disk).
4. Normalise zip_code → zip4 (#####-####).  Handles both the
   "#####-####" and "#########" variants present in meter data.
   Mirrors analysis/rtp/build_regression_dataset.py/_normalize_zip4_expr()
   exactly so mappings are byte-for-byte identical across the codebase.
5. Load the ZIP+4→BG crosswalk (data/reference/comed_bg_zip4_crosswalk.txt,
   tab-separated, columns Zip/Zip4/CensusKey2023).  Derive geoid_bg as
   CensusKey2023.zfill(15)[:12].
6. Resolve crosswalk fan-out (one zip4 → multiple BGs) with
   min(geoid_bg) per zip4 — same deterministic tie-break used by
   build_regression_dataset.py and the stage-2 R scripts.
7. Left-join (account, zip4) pairs to the crosswalk.  For accounts
   that appear with multiple zip4s, apply a second min(geoid_bg)
   tie-break per account_identifier.
8. Log counts and any accounts that fail to join to any BG.
   Abort if the unmatched rate exceeds --max-drop-pct.
9. Assert strict 1:1 account→BG output; fail loudly on violation.
10. Write account_bg_map_{yyyymm}.parquet with columns:
      account_identifier (Utf8), geoid_bg (Utf8, 12-digit FIPS).

Output consumers
----------------
- bill_stats_and_bg_correlation.py   (reads account_identifier, geoid_bg)
- export_delta_geojson_by_class.py   (reads account_identifier, geoid_bg)
- account_count_chain_of_custody.py  (reads account_identifier, geoid_bg)
- validate_mf_esh_bg_count.py        (reads account_identifier, geoid_bg)

Usage::

    uv run python scripts/pricing_pilot/build_account_bg_map.py --month 202307

    uv run python scripts/pricing_pilot/build_account_bg_map.py \\
        --month 202301 \\
        --parquet-dir ~/pricing_pilot/chicago_only \\
        --crosswalk data/reference/comed_bg_zip4_crosswalk.txt \\
        --chicago-zips data/reference/geography/chicago_zip_codes.csv \\
        --output-dir ~/pricing_pilot
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


# ---------------------------------------------------------------------------
# ZIP normalisation (mirrors build_regression_dataset.py/_normalize_zip4_expr)
# ---------------------------------------------------------------------------


def _normalize_zip4_expr() -> pl.Expr:
    """Polars expression: derive zip4 (#####-####) from zip_code.

    Handles both ``#####-####`` (already correct) and ``#########`` (9-digit).
    Mirrors analysis/rtp/build_regression_dataset.py/_normalize_zip4_expr()
    exactly so the join key is identical across all pipeline scripts.
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
# Crosswalk loading (mirrors build_regression_dataset.py/load_crosswalk_one_to_one)
# ---------------------------------------------------------------------------


def _load_crosswalk(crosswalk_path: Path) -> pl.DataFrame:
    """Load ZIP+4→BG crosswalk with deterministic 1:1 linkage.

    Fan-out (one zip4 → multiple BGs) is resolved by taking the smallest
    ``geoid_bg`` per ``zip4``.  This is arbitrary but reproducible and
    matches the R stage-2 scripts and build_regression_dataset.py exactly.

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
            "Crosswalk fan-out: %d zip4(s) map to multiple block groups; "
            "resolving with min(geoid_bg) per zip4 (mirrors build_regression_dataset.py).",
            fanout,
        )

    mapping = lf.group_by("zip4").agg(pl.col("geoid_bg").min()).collect()
    log.info("Crosswalk 1:1 mapping: %d zip4 → geoid_bg entries", mapping.height)
    return mapping


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def main() -> int:
    repo_root = REPO_ROOT
    default_parquet_dir = Path.home() / "pricing_pilot" / "chicago_only"
    default_crosswalk = repo_root / "data/reference/comed_bg_zip4_crosswalk.txt"
    default_chicago_zips = repo_root / "data/reference/geography/chicago_zip_codes.csv"
    default_output_dir = Path.home() / "pricing_pilot"

    parser = argparse.ArgumentParser(
        description="Build account→block-group map (account_bg_map_{yyyymm}.parquet).",
    )
    parser.add_argument(
        "--month",
        required=True,
        help="Month to process in YYYYMM format (e.g. 202307).",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=default_parquet_dir,
        help=(
            "Root of Hive-partitioned canonical long parquets "
            f"(default: {default_parquet_dir}). "
            "Expected layout: <dir>/year=YYYY/month=MM/*.parquet"
        ),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=default_crosswalk,
        help=f"ZIP+4→BG crosswalk TSV (default: {default_crosswalk}).",
    )
    parser.add_argument(
        "--chicago-zips",
        type=Path,
        default=default_chicago_zips,
        help=f"CSV with zip5 column of Chicago-area ZIP codes (default: {default_chicago_zips}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for account_bg_map_{{yyyymm}}.parquet (default: {default_output_dir}).",
    )
    parser.add_argument(
        "--max-drop-pct",
        type=float,
        default=5.0,
        help="Abort if more than this %% of accounts have no BG match (default: 5.0).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate --month
    # ------------------------------------------------------------------
    month: str = args.month
    if len(month) != 6 or not month.isdigit():
        log.error("--month must be 6 digits (YYYYMM), got: %r", month)
        return 1

    # ------------------------------------------------------------------
    # Validate input paths
    # ------------------------------------------------------------------
    for path, label in [
        (args.crosswalk, "--crosswalk"),
        (args.chicago_zips, "--chicago-zips"),
        (args.parquet_dir, "--parquet-dir"),
    ]:
        if not path.exists():
            log.error("%s not found: %s", label, path)
            return 1

    # ------------------------------------------------------------------
    # 1. Locate monthly shard directory
    # ------------------------------------------------------------------
    year, mm = month[:4], month[4:6]
    month_dir = args.parquet_dir / f"year={year}" / f"month={mm}"
    if not month_dir.exists():
        log.error("Month directory not found: %s", month_dir)
        return 1

    parts = sorted(month_dir.glob("*.parquet"))
    if not parts:
        log.error("No parquet shards found in %s", month_dir)
        return 1
    log.info("Found %d parquet shard(s) in %s", len(parts), month_dir)

    # ------------------------------------------------------------------
    # 2. Schema check: verify required columns exist before full scan
    # ------------------------------------------------------------------
    peek_schema = pl.scan_parquet(parts[0]).collect_schema()
    required_cols = {"account_identifier", "zip_code"}
    missing = required_cols - set(peek_schema.names())
    if missing:
        log.error(
            "Parquet shards missing required columns: %s. Found: %s",
            sorted(missing),
            sorted(peek_schema.names()),
        )
        return 1

    # ------------------------------------------------------------------
    # 3. Load Chicago-area ZIP codes
    # ------------------------------------------------------------------
    chicago_zips = pl.read_csv(args.chicago_zips, schema_overrides={"zip5": pl.Utf8}).get_column("zip5").to_list()
    log.info("Loaded %d Chicago-area ZIP codes from %s", len(chicago_zips), args.chicago_zips)

    # ------------------------------------------------------------------
    # 4. Lazy scan: project to needed columns, filter to Chicago ZIPs,
    #    deduplicate (account_identifier, zip_code) pairs.
    #    Column projection avoids loading all 60 canonical columns.
    # ------------------------------------------------------------------
    zip5_expr = pl.col("zip_code").cast(pl.Utf8).str.replace_all(r"[^0-9]", "").str.slice(0, 5).str.zfill(5)

    pairs = (
        pl.scan_parquet(parts)
        .select(["account_identifier", "zip_code"])
        .with_columns(zip5_expr.alias("_zip5"))
        .filter(pl.col("_zip5").is_in(chicago_zips))
        .drop("_zip5")
        .unique(subset=["account_identifier", "zip_code"])
        .collect()
    )

    n_pairs = pairs.height
    n_accounts_raw = pairs.select(pl.col("account_identifier").n_unique()).item()
    log.info(
        "After Chicago ZIP filter: %d unique (account, zip_code) pairs from %d distinct accounts.",
        n_pairs,
        n_accounts_raw,
    )
    if n_accounts_raw == 0:
        log.error("No accounts remain after Chicago ZIP filter. Check --parquet-dir and --chicago-zips.")
        return 1

    # ------------------------------------------------------------------
    # 5. Normalise zip_code → zip4 (#####-####)
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
    # 6. Load crosswalk (already 1:1 zip4→geoid_bg via min tie-break)
    # ------------------------------------------------------------------
    crosswalk = _load_crosswalk(args.crosswalk)

    # ------------------------------------------------------------------
    # 7. Join pairs to crosswalk on zip4 (left join to detect misses)
    # ------------------------------------------------------------------
    joined = pairs.join(crosswalk, on="zip4", how="left")

    n_matched_pairs = joined.filter(pl.col("geoid_bg").is_not_null()).height
    n_unmatched_pairs = joined.height - n_matched_pairs
    log.info(
        "Crosswalk join: %d matched pairs, %d unmatched pairs (zip4 not in crosswalk).",
        n_matched_pairs,
        n_unmatched_pairs,
    )

    # Accounts whose zip4s are ALL unmatched — they get no BG at all.
    matched_accounts = set(joined.filter(pl.col("geoid_bg").is_not_null()).get_column("account_identifier").to_list())
    all_accounts = set(joined.get_column("account_identifier").to_list())
    no_bg_accounts = sorted(all_accounts - matched_accounts)
    n_no_bg = len(no_bg_accounts)
    if n_no_bg:
        pct_no_bg = n_no_bg / len(all_accounts) * 100
        log.warning(
            "%d account(s) (%.2f%%) have no BG match across any of their zip4s. Sample: %s",
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
    # 8. Resolve accounts with multiple zip4s → min(geoid_bg) per account
    #    (second tie-break, same logic as the crosswalk-level tie-break)
    # ------------------------------------------------------------------
    mapping = joined.drop_nulls("geoid_bg").group_by("account_identifier").agg(pl.col("geoid_bg").min())
    n_mapped = mapping.height
    log.info(
        "Final mapping: %d of %d accounts have a 1:1 BG assignment (%d dropped — no BG match).",
        n_mapped,
        n_accounts_raw,
        n_accounts_raw - n_mapped,
    )

    # ------------------------------------------------------------------
    # 9. Enforce strict 1:1 output (fail loudly on data integrity violations)
    # ------------------------------------------------------------------
    n_unique_acct = mapping.select(pl.col("account_identifier").n_unique()).item()
    if mapping.height != n_unique_acct:
        raise RuntimeError(f"BUG: mapping is not 1:1 — {mapping.height} rows but {n_unique_acct} unique accounts.")
    n_null_geoid_bg = mapping.filter(pl.col("geoid_bg").is_null()).height
    if n_null_geoid_bg:
        raise RuntimeError(f"BUG: {n_null_geoid_bg} null geoid_bg value(s) in final mapping.")

    # ------------------------------------------------------------------
    # 10. Write output
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"account_bg_map_{month}.parquet"
    mapping.select("account_identifier", "geoid_bg").write_parquet(out_path)
    log.info("Wrote %s (%d rows).", out_path, mapping.height)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
