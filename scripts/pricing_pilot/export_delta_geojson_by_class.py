#!/usr/bin/env python3
"""Export GeoJSON layers by delivery class for Felt maps (save/lose by customer class).

Reads household bill parquets from ~/pricing_pilot/bills_unscaled/ with naming:
  {yyyymm}_flat_vs_{dtou|stou}_{delivery_class}.parquet
e.g. 202301_flat_vs_dtou_sf_no_esh.parquet

For each file: joins to account_bg_map_{yyyymm}.parquet (202301 for January, 202307 for July),
aggregates mean bill delta to block group (geoid_bg), joins to BG geometry, writes one GeoJSON.
Delta column: bill_diff_dollars with fallback to net_bill_diff_dollars (same as
bill_stats_and_bg_correlation.py).

Produces 16 GeoJSON files (2 months x 2 rate comparisons x 4 delivery classes).

Each GeoJSON contains:
  - geoid_bg: string, 12-digit Census block group ID (FIPS)
  - mean_delta: float, mean monthly bill change ($)
  - n_households: int, count of simulated households
  - geometry: polygon, Census BG boundary

Output files: {output_dir}/{yyyymm}_{dtou|stou}_{delivery_class}.geojson
e.g. 202301_dtou_sf_no_esh.geojson, 202307_stou_mf_esh.geojson

Geometry: Block group shapefile (default repo data/shapefiles/...). If missing, script
exits with error; provide a path via --shapefile.

Usage::

  uv run python scripts/pricing_pilot/export_delta_geojson_by_class.py

  # Override paths:
  uv run python scripts/pricing_pilot/export_delta_geojson_by_class.py \\
    --bills-dir ~/pricing_pilot/bills_unscaled \\
    --account-bg-map-pattern ~/pricing_pilot/account_bg_map_{yyyymm}.parquet \\
    --shapefile /path/to/tl_2023_17_bg.shp \\
    --output-dir ~/pricing_pilot/geojson_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]


def _choose_delta(cols: list[str]) -> pl.Expr:
    """Prefer bill_diff_dollars, fallback to net_bill_diff_dollars (same as bill_stats_and_bg_correlation)."""
    if "bill_diff_dollars" in cols:
        return pl.col("bill_diff_dollars")
    if "net_bill_diff_dollars" in cols:
        return pl.col("net_bill_diff_dollars")
    if "bill_b_dollars" in cols and "bill_a_dollars" in cols:
        return pl.col("bill_b_dollars") - pl.col("bill_a_dollars")
    raise ValueError(f"No delta column found. Have: {cols}")


def _parse_bill_filename(path: Path) -> tuple[str, str, str] | None:
    """Parse {yyyymm}_flat_vs_{dtou|stou}_{delivery_class}.parquet -> (yyyymm, rate, delivery_class)."""
    stem = path.name.replace(".parquet", "")
    if "_flat_vs_" not in stem or stem.count("_") < 4:
        return None
    month = stem[:6]
    if not month.isdigit() or len(month) != 6:
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
    return (month, rate, dc)


def main() -> int:
    default_bills_dir = Path.home() / "pricing_pilot" / "bills_unscaled"
    default_map_pattern = str(Path.home() / "pricing_pilot" / "account_bg_map_{yyyymm}.parquet")
    default_out = Path.home() / "pricing_pilot" / "geojson_out"

    parser = argparse.ArgumentParser(
        description="Export GeoJSON by delivery class from household bills (one file per month/rate/class).",
    )
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
        help="Path with {yyyymm} placeholder for account->BG crosswalk (default: ~/pricing_pilot/account_bg_map_{{yyyymm}}.parquet).",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=REPO_ROOT / "data/shapefiles/tiger2023_il_bg/tl_2023_17_bg.shp",
        help="Block group shapefile (default: repo data/shapefiles/...). If missing, script exits; provide path or add shapefile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Output directory for GeoJSON files (default: {default_out}).",
    )
    args = parser.parse_args()

    # Glob: {yyyymm}_flat_vs_{dtou|stou}_{delivery_class}.parquet
    bills_dir = args.bills_dir
    if not bills_dir.exists():
        print(f"Bills directory not found: {bills_dir}", file=sys.stderr)
        return 1
    bill_paths = sorted(bills_dir.glob("*_flat_vs_*_*.parquet"))
    # Keep only files that parse correctly
    bill_paths = [p for p in bill_paths if _parse_bill_filename(p) is not None]
    if not bill_paths:
        print(
            f"No bill files matching *_flat_vs_*_*.parquet (with parseable month/rate/class) in {bills_dir}",
            file=sys.stderr,
        )
        return 1

    if not args.shapefile.exists():
        print(f"Shapefile not found: {args.shapefile}", file=sys.stderr)
        print("Provide a block group shapefile via --shapefile (e.g. Census TIGER tl_2023_17_bg.shp).", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    g = gpd.read_file(args.shapefile)
    if "GEOID" not in g.columns:
        print("Shapefile missing GEOID column.", file=sys.stderr)
        return 1
    g["GEOID"] = g["GEOID"].astype(str).str.strip()

    written_paths: list[tuple[Path, int, float, float, float]] = []  # (path, n_features, mean, min, max)

    for bill_path in bill_paths:
        bill_path = Path(bill_path)
        parsed = _parse_bill_filename(bill_path)
        if not parsed:
            continue
        month, rate, delivery_class = parsed

        map_path = Path(args.account_bg_map_pattern.format(yyyymm=month))
        if not map_path.exists():
            print(f"Account-BG map not found for {month}: {map_path}", file=sys.stderr)
            continue

        df = pl.read_parquet(bill_path)
        cols = df.columns
        if "account_identifier" not in cols:
            print(f"Bills missing account_identifier: {bill_path}", file=sys.stderr)
            continue

        delta_expr = _choose_delta(cols)
        df = df.with_columns(delta_expr.alias("delta_dollars"))

        amap = pl.read_parquet(map_path).select(
            pl.col("account_identifier").cast(pl.Utf8),
            pl.col("geoid_bg").cast(pl.Utf8),
        )
        joined = (
            df.select(
                pl.col("account_identifier").cast(pl.Utf8),
                pl.col("delta_dollars").cast(pl.Float64),
            )
            .join(amap, on="account_identifier", how="inner")
            .filter(pl.col("geoid_bg").is_not_null())
        )

        bg = joined.group_by("geoid_bg").agg(
            pl.col("delta_dollars").mean().alias("mean_delta"),
            pl.len().alias("n_households"),
        )
        if bg.height == 0:
            continue

        # geoid_bg: 12-digit string for join (same type/format as shapefile GEOID)
        bg_pd = bg.to_pandas()
        bg_pd["geoid_bg"] = bg_pd["geoid_bg"].astype(str).str.strip()
        bg_pd["mean_delta"] = bg_pd["mean_delta"].astype("float64")
        bg_pd["n_households"] = bg_pd["n_households"].astype("int64")
        merged = g.merge(bg_pd, left_on="GEOID", right_on="geoid_bg", how="inner")
        n_unmatched = len(bg_pd) - len(merged)
        if n_unmatched > 0:
            print(
                f"  WARNING: {n_unmatched}/{len(bg_pd)} BG(s) in {bill_path.name} had no"
                " matching GEOID in shapefile and were dropped.",
                file=sys.stderr,
            )
        out_gdf = merged[["geoid_bg", "mean_delta", "n_households", "geometry"]].copy()
        out_gdf.set_geometry("geometry", inplace=True)

        out_path = args.output_dir / f"{month}_{rate}_{delivery_class}.geojson"
        out_gdf.to_file(out_path, driver="GeoJSON")
        n_f = len(out_gdf)
        mean_d = out_gdf["mean_delta"].mean()
        min_d = out_gdf["mean_delta"].min()
        max_d = out_gdf["mean_delta"].max()
        written_paths.append((out_path, n_f, mean_d, min_d, max_d))
        print(f"Wrote {out_path} | BGs: {n_f} | class: {delivery_class}")

    print(f"\nDone. {len(written_paths)} GeoJSON file(s) in {args.output_dir}")
    if written_paths:
        print("\nSanity check (features and mean_delta stats per file):")
        for path, n_f, mean_d, min_d, max_d in written_paths:
            print(f"  {path.name}: features={n_f}, mean_delta={mean_d:.2f}, min={min_d:.2f}, max={max_d:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
