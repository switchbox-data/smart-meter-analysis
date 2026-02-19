#!/usr/bin/env python3
"""Validate mf_esh block-group count: A) distinct accounts in bills, B) distinct BGs after join, C) features in GeoJSON."""

from pathlib import Path

import polars as pl

BASE = Path.home() / "pricing_pilot"
BILLS = BASE / "bills_unscaled"
MAP_DIR = BASE
GEOJSON = BASE / "geojson_out"

MONTHS = [
    ("202301", "January"),
    ("202307", "July"),
]


def main() -> None:
    rows = []
    for yyyymm, label in MONTHS:
        bill_path = BILLS / f"{yyyymm}_flat_vs_dtou_mf_esh.parquet"
        map_path = MAP_DIR / f"account_bg_map_{yyyymm}.parquet"
        geojson_path = GEOJSON / f"{yyyymm}_dtou_mf_esh.geojson"

        # A) Distinct account_identifier in bills
        bills = pl.read_parquet(bill_path)
        if "account_identifier" not in bills.columns:
            # try common alternatives
            id_col = next((c for c in bills.columns if "account" in c.lower() or c == "account_id"), None)
            if id_col is None:
                raise RuntimeError(f"No account column in {bill_path}. Columns: {bills.columns}")
            account_col = id_col
        else:
            account_col = "account_identifier"
        a_count = bills.select(pl.col(account_col).n_unique()).item()

        # B) Join to account_bg_map, count distinct geoid_bg
        amap = (
            pl.read_parquet(map_path)
            .select(
                pl.col("account_identifier").cast(pl.Utf8),
                pl.col("geoid_bg").cast(pl.Utf8),
            )
            .unique(subset=["account_identifier"], keep="first")
        )
        accounts = bills.select(pl.col(account_col).cast(pl.Utf8).alias("account_identifier"))
        joined = accounts.join(amap, on="account_identifier", how="inner").filter(
            pl.col("geoid_bg").is_not_null() & (pl.col("geoid_bg").str.strip_chars() != "")
        )
        b_count = joined.select(pl.col("geoid_bg").n_unique()).item()
        b_geoids = set(joined["geoid_bg"].unique().to_list())

        # C) Count features in GeoJSON
        import json

        with open(geojson_path) as f:
            gj = json.load(f)
        features = gj.get("features", [])
        c_count = len(features)
        c_geoids = set()
        for feat in features:
            props = feat.get("properties", {})
            g = props.get("geoid_bg") or props.get("GEOID") or props.get("geoid")
            if g is not None:
                c_geoids.add(str(g).strip())

        rows.append((label, yyyymm, a_count, b_count, c_count))
        print(f"--- {label} ({yyyymm}) ---")
        print(f"  A) Distinct accounts (bills):     {a_count}")
        print(f"  B) Distinct geoid_bg (join):     {b_count}")
        print(f"  C) GeoJSON feature count:         {c_count}")
        if b_count != c_count:
            print("  >>> B ≠ C: checking for join/type mismatches")
            only_in_join = b_geoids - c_geoids
            only_in_geojson = c_geoids - b_geoids
            if only_in_join:
                sample_join = sorted(only_in_join)[:10]
                print(f"  Sample geoid_bg only in join (not in GeoJSON): {sample_join}")
                if sample_join:
                    print(f"  Types/repr from join: {[repr(x) for x in sample_join]}")
            if only_in_geojson:
                sample_gj = sorted(only_in_geojson)[:10]
                print(f"  Sample geoid_bg only in GeoJSON (not in join): {sample_gj}")
                if sample_gj:
                    print(f"  Types/repr from GeoJSON: {[repr(x) for x in sample_gj]}")
        print()

    # Side-by-side summary
    print("=" * 60)
    print("Summary (side by side)")
    print("=" * 60)
    print(f"{'Month':<12} {'A (accounts)':>14} {'B (BGs join)':>14} {'C (GeoJSON)':>14}")
    print("-" * 60)
    for label, _yyyymm, a, b, c in rows:
        print(f"{label:<12} {a:>14} {b:>14} {c:>14}")
    print("=" * 60)


if __name__ == "__main__":
    main()
