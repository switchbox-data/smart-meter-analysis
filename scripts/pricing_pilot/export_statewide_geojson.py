#!/usr/bin/env python3
"""Export GeoJSON choropleth layers for statewide pricing simulation maps.

Reads BG-level CSVs produced by run_statewide_analysis.py and joins them to
Census block group geometry to produce GeoJSON files suitable for web map
embedding (e.g. Felt, Mapbox).

Inputs:
  - BG-level CSVs: {analysis-dir}/bg_level_{stou|dtou}_{jan|jul}.csv
    Expected columns: geoid_bg, mean_delta, mean_pct_savings, n_households,
    median_household_income (plus others that are ignored).
  - TIGER shapefile: block group boundaries with a GEOID column.

Outputs (one per scenario):
  {out-dir}/{scenario}.geojson with fields:
    geoid_bg, mean_delta, mean_pct_savings, n_households,
    median_household_income, mean_delta_clamped, geometry

  {out-dir}/color_range.json — symmetric color bounds for consistent legends.

Sign convention (matches run_statewide_analysis.py / compute_household_bills.py):
  delta = flat_rate - alternative_rate
  Positive = customer SAVES under the alternative rate (TOU is cheaper)
  Negative = customer PAYS MORE under the alternative rate (TOU is worse)

All IL block groups appear in the output (left join from shapefile). BGs with
no modeled data have null metric values and n_households = 0.

Usage::

  python scripts/pricing_pilot/export_statewide_geojson.py \\
      --analysis-dir /ebs/home/griffin_switch_box/runs/billing_output/statewide_analysis \\
      --shapefile data/shapefiles/tiger2023_il_bg/tl_2023_17_bg.shp \\
      --out-dir /ebs/home/griffin_switch_box/runs/billing_output/statewide_geojson
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

SIGN_CONVENTION = (
    "delta = flat_rate - alternative_rate; "
    "positive = customer saves under alt rate; "
    "negative = customer pays more under alt rate"
)

# The 20 scenarios we expect: 4 pooled + 16 per-class, mapped to CSV filenames.
SCENARIOS = {
    "stou_jan": "bg_level_stou_jan.csv",
    "stou_jul": "bg_level_stou_jul.csv",
    "dtou_jan": "bg_level_dtou_jan.csv",
    "dtou_jul": "bg_level_dtou_jul.csv",
    "stou_jan_sf_no_esh": "bg_level_stou_jan_sf_no_esh.csv",
    "stou_jan_mf_no_esh": "bg_level_stou_jan_mf_no_esh.csv",
    "stou_jan_sf_esh": "bg_level_stou_jan_sf_esh.csv",
    "stou_jan_mf_esh": "bg_level_stou_jan_mf_esh.csv",
    "stou_jul_sf_no_esh": "bg_level_stou_jul_sf_no_esh.csv",
    "stou_jul_mf_no_esh": "bg_level_stou_jul_mf_no_esh.csv",
    "stou_jul_sf_esh": "bg_level_stou_jul_sf_esh.csv",
    "stou_jul_mf_esh": "bg_level_stou_jul_mf_esh.csv",
    "dtou_jan_sf_no_esh": "bg_level_dtou_jan_sf_no_esh.csv",
    "dtou_jan_mf_no_esh": "bg_level_dtou_jan_mf_no_esh.csv",
    "dtou_jan_sf_esh": "bg_level_dtou_jan_sf_esh.csv",
    "dtou_jan_mf_esh": "bg_level_dtou_jan_mf_esh.csv",
    "dtou_jul_sf_no_esh": "bg_level_dtou_jul_sf_no_esh.csv",
    "dtou_jul_mf_no_esh": "bg_level_dtou_jul_mf_no_esh.csv",
    "dtou_jul_sf_esh": "bg_level_dtou_jul_sf_esh.csv",
    "dtou_jul_mf_esh": "bg_level_dtou_jul_mf_esh.csv",
}

# Columns carried from the BG-level CSV into the GeoJSON output.
METRIC_COLS = [
    "mean_delta",
    "mean_pct_savings",
    "n_households",
    "median_household_income",
]


def main() -> int:
    """Export statewide GeoJSON choropleth layers from BG-level analysis CSVs."""
    parser = argparse.ArgumentParser(
        description="Export statewide GeoJSON choropleth layers from BG-level analysis CSVs.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing bg_level_*.csv files from run_statewide_analysis.py.",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=REPO_ROOT / "data/shapefiles/tiger2023_il_bg/tl_2023_17_bg.shp",
        help="Block group shapefile with GEOID column (default: repo data/shapefiles/...).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for GeoJSON files and color_range.json.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.80,
        metavar="Q",
        help="Quantile of |mean_delta| for symmetric color bound (default: 0.80).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not args.analysis_dir.is_dir():
        print(f"Analysis directory not found: {args.analysis_dir}", file=sys.stderr)
        return 1
    if not args.shapefile.exists():
        print(f"Shapefile not found: {args.shapefile}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Load shapefile
    # ------------------------------------------------------------------
    print(f"Loading shapefile: {args.shapefile}")
    geo = gpd.read_file(args.shapefile)
    if "GEOID" not in geo.columns:
        print("Shapefile missing GEOID column.", file=sys.stderr)
        return 1
    geo["GEOID"] = geo["GEOID"].astype(str).str.strip()
    print(f"  {len(geo)} block group features loaded.")

    # ------------------------------------------------------------------
    # Pass 1: load all scenario CSVs, collect mean_delta values
    # ------------------------------------------------------------------
    print(f"\nPass 1: loading BG-level CSVs from {args.analysis_dir} ...")
    loaded: dict[str, pd.DataFrame] = {}
    all_mean_deltas: list[float] = []

    for scenario_name, csv_name in SCENARIOS.items():
        csv_path = args.analysis_dir / csv_name
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found; skipping scenario '{scenario_name}'.", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path, dtype={"geoid_bg": str})
        df["geoid_bg"] = df["geoid_bg"].astype(str).str.strip().str.zfill(12)

        # Validate expected columns are present
        missing = [c for c in METRIC_COLS if c not in df.columns]
        if missing:
            print(f"  ERROR: {csv_name} missing columns: {missing}", file=sys.stderr)
            return 1

        loaded[scenario_name] = df
        non_null_deltas = df["mean_delta"].dropna().tolist()
        all_mean_deltas.extend(non_null_deltas)
        print(f"  {csv_name}: {len(df)} BGs, {len(non_null_deltas)} with mean_delta values")

    if not loaded:
        print("No scenario CSVs were loaded.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Compute global symmetric color bound
    # ------------------------------------------------------------------
    if not all_mean_deltas:
        print("No mean_delta values across any scenario.", file=sys.stderr)
        return 1

    q = args.quantile
    bound = float(np.quantile(np.abs(all_mean_deltas), q))
    print(f"\nGlobal symmetric bound: {bound:.4f}")
    print(f"  (p{int(q * 100)} of |mean_delta| across {len(loaded)} scenarios, {len(all_mean_deltas)} BG values)")
    print(f"  Legend range: [-{bound:.2f}, +{bound:.2f}]")

    # ------------------------------------------------------------------
    # Pass 2: join geometry, clamp, validate, write GeoJSON
    # ------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPass 2: writing GeoJSON to {args.out_dir} ...")
    written: list[Path] = []
    expected_count: int | None = None

    for scenario_name, df in loaded.items():
        # Left join: all BGs in shapefile appear; unmatched get null metrics.
        merged = geo.merge(df[["geoid_bg", *METRIC_COLS]], left_on="GEOID", right_on="geoid_bg", how="left")

        # Fill join key and n_households for unmatched BGs.
        merged["geoid_bg"] = merged["geoid_bg"].fillna(merged["GEOID"])
        merged["n_households"] = merged["n_households"].fillna(0).astype("int64")

        # Validate geoid_bg uniqueness.
        if merged["geoid_bg"].duplicated().any():
            dupes = merged.loc[merged["geoid_bg"].duplicated(keep=False), "geoid_bg"].unique()[:10]
            print(f"ERROR: duplicate geoid_bg in {scenario_name}: {dupes.tolist()}", file=sys.stderr)
            return 1

        # Validate consistent feature count.
        n_features = len(merged)
        if expected_count is None:
            expected_count = n_features
        elif n_features != expected_count:
            print(
                f"ERROR: feature count mismatch in {scenario_name}: {n_features} vs expected {expected_count}",
                file=sys.stderr,
            )
            return 1

        # Warn about data BGs that had no geometry match.
        n_unmatched = int((~df["geoid_bg"].isin(geo["GEOID"])).sum())
        if n_unmatched > 0:
            print(
                f"  WARNING: {n_unmatched}/{len(df)} BG(s) in {scenario_name} had no matching GEOID in shapefile.",
                file=sys.stderr,
            )

        # Clamp mean_delta to symmetric bound; NaN stays NaN.
        merged["mean_delta_clamped"] = merged["mean_delta"].clip(-bound, bound)

        # Select output columns.
        out_cols = [
            "geoid_bg",
            "mean_delta",
            "mean_pct_savings",
            "n_households",
            "median_household_income",
            "mean_delta_clamped",
            "geometry",
        ]
        out_gdf = gpd.GeoDataFrame(merged[out_cols], geometry="geometry")

        out_path = args.out_dir / f"{scenario_name}.geojson"
        out_gdf.to_file(out_path, driver="GeoJSON")

        # Per-file summary.
        n_nonnull = int(out_gdf["mean_delta"].notna().sum())
        mean_d = out_gdf["mean_delta"].mean()
        min_d = out_gdf["mean_delta"].min()
        max_d = out_gdf["mean_delta"].max()
        n_hh = int(out_gdf["n_households"].sum())
        print(
            f"  {out_path.name}: {n_features} features ({n_nonnull} with data), "
            f"mean_delta [{min_d:.2f}, {max_d:.2f}] mean={mean_d:.2f}, "
            f"n_households={n_hh:,}"
        )
        written.append(out_path)

    # ------------------------------------------------------------------
    # Write color_range.json sidecar
    # ------------------------------------------------------------------
    range_meta = {
        "bound_sym": bound,
        "method": f"p{int(q * 100)}_abs_over_all_scenarios",
        "quantile": q,
        "legend_min": -bound,
        "legend_max": bound,
        "n_scenarios": len(loaded),
        "n_bg_values_used": len(all_mean_deltas),
        "sign_convention": SIGN_CONVENTION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    range_path = args.out_dir / "color_range.json"
    range_path.write_text(json.dumps(range_meta, indent=2) + "\n")
    print(f"\nWrote {range_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\nDone. {len(written)} GeoJSON file(s) in {args.out_dir}")
    print(f"  Symmetric bound: +/-{bound:.4f}")
    if expected_count is not None:
        print(f"  All layers have {expected_count} BG features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
