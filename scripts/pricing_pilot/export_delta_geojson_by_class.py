#!/usr/bin/env python3
"""Export GeoJSON layers by delivery class for Felt maps (save/lose by customer class).

Reads household bill parquets from ~/pricing_pilot/bills_unscaled/ with naming:
  {yyyymm}_flat_vs_{dtou|stou}_{delivery_class}.parquet
e.g. 202301_flat_vs_dtou_sf_no_esh.parquet

For each file: joins to account_bg_map_{yyyymm}.parquet (202301 for January, 202307 for July),
aggregates mean bill delta to block group (geoid_bg), joins to BG geometry, writes one GeoJSON.
Delta computed as bill_a_dollars - bill_b_dollars when available; fallback to
net_bill_diff_dollars, then bill_diff_dollars.

Sign convention (canonical, matches compute_household_bills.py):
  delta = flat_rate - alternative_rate (bill_a - bill_b)
  - positive (+) means customer SAVES under the alternative rate (TOU is cheaper)
  - negative (-) means customer PAYS MORE under the alternative rate (TOU is worse)

BG universe: by default, all block groups in the shapefile (Illinois statewide).
When --zip-file is provided, the universe is narrowed to block groups reachable
from those ZIPs via the crosswalk (e.g. Chicago-only analysis).

Produces 16 GeoJSON files (2 months x 2 rate comparisons x 4 delivery classes).

Each GeoJSON contains:
  - geoid_bg: string, 12-digit Census block group ID (FIPS)
  - mean_delta: float, mean monthly bill delta ($, positive=saves); null for BGs with no data
  - n_households: int, count of simulated households; null for BGs with no data
  - mean_delta_cap_global_sym: float, mean_delta clamped to [-X, +X] where X is
      the p80 of |mean_delta| across ALL scenarios; null where mean_delta is null.
      Use this field for Felt choropleth with a consistent legend range.
  - geometry: polygon, Census BG boundary

Domain anchors: two synthetic features (__DOMAIN_ANCHOR_MIN__, __DOMAIN_ANCHOR_MAX__)
are injected by default to force Felt to use the full symmetric color domain across
maps. Disable with --no-domain-anchors.

Output files: {output_dir}/{yyyymm}_{dtou|stou}_{delivery_class}.geojson
e.g. 202301_dtou_sf_no_esh.geojson, 202307_stou_mf_esh.geojson

Sidecar: {output_dir}/range_global.json — contains bound_sym (X), method, sign convention.

Geometry: Block group shapefile (default repo data/shapefiles/...). If missing, script
exits with error; provide a path via --shapefile.

Usage::

  # Statewide (default): all BGs in the shapefile
  uv run python scripts/pricing_pilot/export_delta_geojson_by_class.py

  # Chicago-filtered: restrict to BGs reachable from Chicago ZIPs
  uv run python scripts/pricing_pilot/export_delta_geojson_by_class.py \\
    --zip-file data/reference/geography/chicago_zip_codes.csv

  # Override paths:
  uv run python scripts/pricing_pilot/export_delta_geojson_by_class.py \\
    --bills-dir ~/pricing_pilot/bills_unscaled \\
    --account-bg-map-pattern ~/pricing_pilot/account_bg_map_{yyyymm}.parquet \\
    --shapefile /path/to/tl_2023_17_bg.shp \\
    --output-dir ~/pricing_pilot/geojson_out

  # Disable global range (omits mean_delta_cap_global_sym and range_global.json):
  uv run python scripts/pricing_pilot/export_delta_geojson_by_class.py --no-global-range

Felt usage:
  - Set numericAttribute to mean_delta_cap_global_sym (NOT mean_delta)
  - Set legend min = -bound_sym, max = +bound_sym for ALL layers
  - bound_sym is printed at run end and stored in range_global.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sign convention (canonical, matches compute_household_bills.py):
#   delta = flat_rate - alternative_rate (bill_a - bill_b)
#   Positive = customer SAVES under the alternative (TOU is cheaper)
#   Negative = customer PAYS MORE under the alternative (TOU is worse)
SIGN_CONVENTION = "delta = flat_rate - alternative_rate (bill_a - bill_b); positive = customer saves under alt rate; negative = customer pays more under alt rate"


@dataclass
class _ScenarioData:
    month: str
    rate: str
    delivery_class: str
    bill_path: Path
    # pandas DataFrame with columns: geoid_bg (str), mean_delta (float), n_households (int)
    bg_pd: pd.DataFrame
    county_fips_set: frozenset


def _choose_delta(cols: list[str]) -> pl.Expr:
    """Return a Polars expression for bill delta with the canonical sign convention.

    Preference order:
      1. bill_a_dollars - bill_b_dollars  (explicit; matches sign convention by construction)
      2. net_bill_diff_dollars            (fallback; semantics assumed to be flat - alt)
      3. bill_diff_dollars                (last resort; semantics may vary — Pass 1 consistency
                                           check will warn if sign is flipped)

    Sign convention: delta = flat_rate - alternative_rate (bill_a - bill_b).
    Positive = customer SAVES under alt rate; negative = customer pays more.
    """
    if "bill_b_dollars" in cols and "bill_a_dollars" in cols:
        return pl.col("bill_a_dollars") - pl.col("bill_b_dollars")
    if "net_bill_diff_dollars" in cols:
        # Assumed semantics: flat - alt. Caller must verify.
        return pl.col("net_bill_diff_dollars")
    if "bill_diff_dollars" in cols:
        # Last resort. Semantics may vary; consistency check below will warn if sign-flipped.
        return pl.col("bill_diff_dollars")
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
    """Export GeoJSON layers by delivery class from household bill parquets."""
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
        "--zip-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with a zip5 column to restrict the BG universe to block groups "
            "reachable from those ZIPs via the crosswalk. When omitted, all BGs in the "
            "shapefile are included. Example: data/reference/geography/chicago_zip_codes.csv"
        ),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=REPO_ROOT / "data/reference/comed_bg_zip4_crosswalk.txt",
        help="ZIP+4→BG crosswalk TSV. Required when --zip-file is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Output directory for GeoJSON files (default: {default_out}).",
    )
    parser.add_argument(
        "--no-global-range",
        dest="global_range",
        action="store_false",
        default=True,
        help="Disable global symmetric range: omits mean_delta_cap_global_sym column and range_global.json.",
    )
    parser.add_argument(
        "--global-range-quantile",
        type=float,
        default=0.80,
        metavar="Q",
        help="Quantile of |mean_delta| used for global symmetric bound X (default: 0.80 = p80).",
    )
    parser.add_argument(
        "--no-domain-anchors",
        dest="domain_anchors",
        action="store_false",
        default=True,
        help="Disable domain anchor injection (two synthetic features that force Felt to use the full symmetric color domain).",
    )
    parser.add_argument(
        "--expected-bg-count",
        type=int,
        default=None,
        metavar="N",
        help="Expected BG feature count per output file. If provided, fails if any file differs. "
        "If omitted, the first file's count becomes the expectation and subsequent files are checked against it.",
    )
    args = parser.parse_args()

    bills_dir = args.bills_dir
    if not bills_dir.exists():
        print(f"Bills directory not found: {bills_dir}", file=sys.stderr)
        return 1
    bill_paths = sorted(bills_dir.glob("*_flat_vs_*_*.parquet"))
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

    # =========================================================================
    # Build BG universe.
    # When --zip-file is provided, universe = BG GEOIDs reachable from those
    # ZIP5s via the crosswalk. Otherwise, all BGs in the shapefile are included.
    # BGs outside the universe are excluded entirely; BGs inside with no billing
    # data are kept as gray "no-data" polygons (n_households=0, metric null).
    # =========================================================================
    if args.zip_file is not None:
        # Sub-geography mode: build BG universe from ZIP filter via crosswalk
        if not args.zip_file.exists():
            print(f"--zip-file not found: {args.zip_file}", file=sys.stderr)
            return 1
        if not args.crosswalk.exists():
            print(f"--crosswalk not found: {args.crosswalk}", file=sys.stderr)
            return 1
        _zip5_raw = pl.read_csv(args.zip_file, infer_schema_length=0)
        _zip5s = set(_zip5_raw["zip5"].str.strip_chars().str.zfill(5).to_list())
        _xw = pl.read_csv(args.crosswalk, separator="\t", infer_schema_length=0)
        _xw_filtered = _xw.filter(pl.col("Zip").str.strip_chars().str.zfill(5).is_in(_zip5s))
        _bg_universe_geoids = frozenset(
            _xw_filtered["CensusKey2023"].str.strip_chars().str.zfill(15).str.slice(0, 12).unique().to_list()
        )
        g_universe = g[g["GEOID"].isin(_bg_universe_geoids)].copy()
        if len(g_universe) == 0:
            print(
                "ERROR: ZIP-based BG universe produced 0 geometry features."
                " Check --zip-file, --crosswalk, and shapefile GEOID format.",
                file=sys.stderr,
            )
            return 1
        geo_scope = f"zip_filter:{args.zip_file.name}"
        print(f"[geo] BG universe (ZIP-filtered): kept {len(g_universe)}/{len(g)} features")
    else:
        # Statewide mode: all BGs in shapefile
        g_universe = g.copy()
        geo_scope = "statewide"
        print(f"[geo] BG universe (statewide): all {len(g_universe)} features from shapefile")

    # =========================================================================
    # PASS 1: load all scenarios and aggregate to BG level.
    # We do this before writing any files so we can compute the global range X
    # from ALL scenarios before committing any output.
    # =========================================================================
    print(f"Pass 1: aggregating {len(bill_paths)} bill file(s) to block-group level...")
    scenarios: list[_ScenarioData] = []
    all_mean_deltas: list[float] = []  # collected across every scenario for global X

    for bill_path in bill_paths:
        bill_path = Path(bill_path)
        parsed = _parse_bill_filename(bill_path)
        if not parsed:
            continue
        month, rate, delivery_class = parsed

        map_path = Path(args.account_bg_map_pattern.format(yyyymm=month))
        if not map_path.exists():
            print(f"  Account-BG map not found for {month}: {map_path}", file=sys.stderr)
            continue

        lf = pl.scan_parquet(bill_path)
        cols = lf.collect_schema().names()
        # Only read columns needed for delta computation and join.
        _needed = {"account_identifier"} & set(cols)
        for c in ("bill_a_dollars", "bill_b_dollars", "bill_diff_dollars", "net_bill_diff_dollars"):
            if c in cols:
                _needed.add(c)
        _needed.add("account_identifier")
        df = lf.select(sorted(_needed)).collect()
        if "account_identifier" not in cols:
            print(f"  Bills missing account_identifier: {bill_path}", file=sys.stderr)
            continue

        # Internal consistency check: if both bill_a/b_dollars AND bill_diff_dollars
        # are present, verify that bill_diff_dollars matches (a - b) and is not sign-flipped.
        # This catches upstream sign errors early and loudly.
        if {"bill_a_dollars", "bill_b_dollars", "bill_diff_dollars"}.issubset(cols):
            _chk = df.select(
                (pl.col("bill_a_dollars") - pl.col("bill_b_dollars")).alias("d_ab"),
                pl.col("bill_diff_dollars").alias("d_col"),
            ).with_columns(
                (pl.col("d_ab") - pl.col("d_col")).abs().alias("diff_same"),
                (pl.col("d_ab") + pl.col("d_col")).abs().alias("diff_flip"),
            )
            mean_same = _chk["diff_same"].mean()
            mean_flip = _chk["diff_flip"].mean()
            if mean_flip < mean_same:
                print(
                    f"  WARNING [{bill_path.name}]: bill_diff_dollars is OPPOSITE-SIGN to"
                    f" (bill_a - bill_b) [mean|a-b - diff|={mean_same:.4g},"
                    f" mean|a-b + diff|={mean_flip:.4g}]."
                    " Using (bill_a_dollars - bill_b_dollars); bill_diff_dollars IGNORED.",
                    file=sys.stderr,
                )
            elif mean_same > 1e-6:
                print(
                    f"  WARNING [{bill_path.name}]: bill_diff_dollars differs from (bill_a - bill_b)"
                    f" but is not a clean sign-flip [mean|same|={mean_same:.4g},"
                    f" mean|flip|={mean_flip:.4g}]. Using (bill_a - bill_b).",
                    file=sys.stderr,
                )
            # else: agree within tolerance — no warning

        # Sign convention: delta = flat_rate - alternative_rate (bill_a - bill_b)
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
            print(f"  No BG data after join for {bill_path.name}; skipping.", file=sys.stderr)
            continue

        bg_pd = bg.to_pandas()
        bg_pd["geoid_bg"] = bg_pd["geoid_bg"].astype(str).str.strip()
        bg_pd["mean_delta"] = bg_pd["mean_delta"].astype("float64")
        bg_pd["n_households"] = bg_pd["n_households"].astype("int64")

        county_fips_set = frozenset(bg_pd["geoid_bg"].str[:5])
        all_mean_deltas.extend(bg_pd["mean_delta"].dropna().tolist())
        scenarios.append(
            _ScenarioData(
                month=month,
                rate=rate,
                delivery_class=delivery_class,
                bill_path=bill_path,
                bg_pd=bg_pd,
                county_fips_set=county_fips_set,
            )
        )
        print(f"  Loaded {bill_path.name}: {bg.height} BGs with data, counties={sorted(county_fips_set)}")

    if not scenarios:
        print("No scenarios were successfully processed.", file=sys.stderr)
        return 1

    # =========================================================================
    # Compute global symmetric bound X.
    # X = quantile(|mean_delta|, q) across all non-null BG values in all scenarios.
    # Using a symmetric bound ensures +$5 means the same thing on every layer.
    # =========================================================================
    bound_sym: float | None = None
    if args.global_range:
        if not all_mean_deltas:
            print("WARNING: global range requested but no mean_delta values collected.", file=sys.stderr)
        else:
            q = args.global_range_quantile
            bound_sym = float(np.quantile(np.abs(all_mean_deltas), q))
            print(
                f"\nGlobal symmetric bound: X = {bound_sym:.4f}"
                f"  (p{int(q * 100)} of |mean_delta| across {len(scenarios)} scenarios,"
                f" {len(all_mean_deltas)} BG values)"
            )
            print(f"  => Felt legend: min = -{bound_sym:.2f}  max = +{bound_sym:.2f}")
            print(f"  => Sign convention: {SIGN_CONVENTION}\n")

    # =========================================================================
    # PASS 2: join geometry, add mean_delta_cap_global_sym, validate, write.
    # =========================================================================
    print(f"Pass 2: joining geometry and writing {len(scenarios)} GeoJSON file(s)...")
    written_paths: list[Path] = []
    expected_bg_count: int | None = args.expected_bg_count  # None until first file sets it

    for s in scenarios:
        # Scope geometry to BG universe (statewide or ZIP-filtered).
        # g_universe is fixed across all scenarios; computed once before Pass 1.
        g_bg_universe = g_universe

        # Left join: every BG in the universe appears; null for BGs with no household data.
        merged = g_bg_universe.merge(s.bg_pd, left_on="GEOID", right_on="geoid_bg", how="left")
        # Fill geoid_bg from GEOID for rows with no matching household data.
        merged["geoid_bg"] = merged["geoid_bg"].fillna(merged["GEOID"])
        # Fill n_households to 0 for BGs with no modeled households.
        merged["n_households"] = merged["n_households"].fillna(0).astype("int64")

        # ---- Fail-loud validation ----------------------------------------

        # 1. geoid_bg uniqueness
        if merged["geoid_bg"].duplicated().any():
            dupes = merged.loc[merged["geoid_bg"].duplicated(keep=False), "geoid_bg"].tolist()
            print(
                f"ERROR: duplicate geoid_bg in {s.bill_path.name}: {dupes[:10]}",
                file=sys.stderr,
            )
            return 1

        # 2. Feature count consistency across all files
        n_f = len(merged)
        if expected_bg_count is None:
            expected_bg_count = n_f  # first file sets the expectation
        elif n_f != expected_bg_count:
            print(
                f"ERROR: BG feature count mismatch in {s.bill_path.name}: got {n_f}, expected {expected_bg_count}.",
                file=sys.stderr,
            )
            return 1

        # 3. Warn if any BGs with data had no matching geometry (data is lost).
        n_unmatched = int((~s.bg_pd["geoid_bg"].isin(g_bg_universe["GEOID"])).sum())
        if n_unmatched > 0:
            print(
                f"  WARNING: {n_unmatched}/{len(s.bg_pd)} BG(s) with data in {s.bill_path.name}"
                " had no matching GEOID in shapefile and were excluded.",
                file=sys.stderr,
            )

        # ---- Build output GeoDataFrame -----------------------------------

        out_cols = ["geoid_bg", "mean_delta", "n_households"]

        if bound_sym is not None:
            # Clamp mean_delta to [-X, +X]; NaN stays NaN (no data BGs remain null).
            merged["mean_delta_cap_global_sym"] = merged["mean_delta"].clip(-bound_sym, bound_sym)
            out_cols.append("mean_delta_cap_global_sym")

        out_cols.append("geometry")
        out_gdf = merged[out_cols].copy()
        out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry")

        # ---- Domain anchor injection -------------------------------------
        # Two synthetic features that pin the color domain to [-bound_sym, +bound_sym]
        # so Felt uses the full symmetric range across all maps.
        if bound_sym is not None and args.domain_anchors:
            # Fail-loud: no real BG (n_households > 0) should have null capped delta
            real_with_null_cap = out_gdf[(out_gdf["n_households"] > 0) & (out_gdf["mean_delta_cap_global_sym"].isna())]
            if len(real_with_null_cap) > 0:
                print(
                    f"ERROR: {len(real_with_null_cap)} BG(s) with households have null "
                    f"mean_delta_cap_global_sym in {s.bill_path.name}",
                    file=sys.stderr,
                )
                return 1

            anchor_min = {
                "geoid_bg": "__DOMAIN_ANCHOR_MIN__",
                "mean_delta": None,
                "n_households": 0,
                "mean_delta_cap_global_sym": -bound_sym,
                "geometry": None,
            }
            anchor_max = {
                "geoid_bg": "__DOMAIN_ANCHOR_MAX__",
                "mean_delta": None,
                "n_households": 0,
                "mean_delta_cap_global_sym": bound_sym,
                "geometry": None,
            }
            anchors_gdf = gpd.GeoDataFrame([anchor_min, anchor_max], geometry="geometry", crs=out_gdf.crs)
            out_gdf = gpd.GeoDataFrame(pd.concat([out_gdf, anchors_gdf], ignore_index=True), geometry="geometry")

        out_path = args.output_dir / f"{s.month}_{s.rate}_{s.delivery_class}.geojson"
        out_gdf.to_file(out_path, driver="GeoJSON")

        # ---- Per-file summary --------------------------------------------
        n_nonnull = int(out_gdf["mean_delta"].notna().sum())
        mean_d = float(out_gdf["mean_delta"].mean())
        min_d = float(out_gdf["mean_delta"].min())
        max_d = float(out_gdf["mean_delta"].max())
        n_hh = int(out_gdf["n_households"].dropna().sum())

        summary_parts = [
            f"n_features_total={n_f}",
            f"n_features_nonnull={n_nonnull}",
            f"mean_delta min={min_d:.2f} mean={mean_d:.2f} max={max_d:.2f}",
        ]
        if bound_sym is not None:
            cap_min = float(out_gdf["mean_delta_cap_global_sym"].min())
            cap_max = float(out_gdf["mean_delta_cap_global_sym"].max())
            summary_parts.append(f"cap=[{cap_min:.2f}, {cap_max:.2f}]")
        summary_parts.append(f"n_households_nonnull={n_hh}")

        print(f"  Wrote {out_path.name} | " + " | ".join(summary_parts))
        written_paths.append(out_path)

    # =========================================================================
    # Write range_global.json sidecar.
    # =========================================================================
    if bound_sym is not None:
        q = args.global_range_quantile
        range_meta = {
            "bound_sym": bound_sym,
            "method": f"p{int(q * 100)}_abs_over_all_scenarios",
            "global_range_quantile": q,
            "geo_scope": geo_scope,
            "domain_anchors": args.domain_anchors,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sign_convention": SIGN_CONVENTION,
            "n_scenarios": len(scenarios),
            "n_bg_values_used": len(all_mean_deltas),
        }
        range_path = args.output_dir / "range_global.json"
        range_path.write_text(json.dumps(range_meta, indent=2) + "\n")
        print(f"\nWrote {range_path}")

    # =========================================================================
    # Final summary.
    # =========================================================================
    print(f"\nDone. {len(written_paths)} GeoJSON file(s) in {args.output_dir}")
    if bound_sym is not None:
        print(f"  All layers share bound_sym = ±{bound_sym:.4f}")
        print(
            f"  Felt: set numericAttribute=mean_delta_cap_global_sym, legend min=-{bound_sym:.2f}, max=+{bound_sym:.2f}"
        )
    if expected_bg_count is not None:
        print(f"  All layers have {expected_bg_count} BG features (consistent coverage).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
