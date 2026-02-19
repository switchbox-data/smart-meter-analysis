import glob
import os
import polars as pl
import geopandas as gpd

ROOT = "/ebs/home/griffin_switch_box"
REPO = f"{ROOT}/smart-meter-analysis"

BILLS_DIR = f"{ROOT}/pricing_pilot/bills"
MAP_DIR = f"{ROOT}/pricing_pilot"
SHP_PATH = f"{REPO}/data/shapefiles/tiger2023_il_bg/tl_2023_17_bg.shp"
OUT_DIR = f"{ROOT}/pricing_pilot/regression/maps"

os.makedirs(OUT_DIR, exist_ok=True)

jobs = [
    ("jan_2023_dtou", f"{BILLS_DIR}/chicago_202301_*vs_dtou_scaled_allin_*.parquet", f"{MAP_DIR}/account_bg_map_202301.parquet"),
    ("jan_2023_stou", f"{BILLS_DIR}/chicago_202301_*vs_rate_best*_scaled*.parquet", f"{MAP_DIR}/account_bg_map_202301.parquet"),
    ("jul_2023_dtou", f"{BILLS_DIR}/chicago_202307_*vs_dtou*_scaled*.parquet", f"{MAP_DIR}/account_bg_map_202307.parquet"),
    ("jul_2023_stou", f"{BILLS_DIR}/chicago_202307_*vs_rate_best*_scaled*.parquet", f"{MAP_DIR}/account_bg_map_202307.parquet"),
]

def pick_many(pattern: str) -> list[str]:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise RuntimeError(f"No files matched glob: {pattern}")
    return hits

def choose_delta(cols: list[str]) -> pl.Expr:
    # IMPORTANT: these columns are already Alternative - Flat in your billing outputs.
    if "net_bill_diff_dollars" in cols:
        return pl.col("net_bill_diff_dollars")
    if "bill_diff_dollars" in cols:
        return pl.col("bill_diff_dollars")
    if "bill_b_dollars" in cols and "bill_a_dollars" in cols:
        return pl.col("bill_b_dollars") - pl.col("bill_a_dollars")
    raise RuntimeError(f"No valid delta columns found. cols={cols}")

if not os.path.exists(SHP_PATH):
    raise FileNotFoundError(SHP_PATH)

g = gpd.read_file(SHP_PATH)
if "GEOID" not in g.columns:
    raise RuntimeError("Shapefile missing GEOID")
g["GEOID"] = g["GEOID"].astype(str)

for label, bills_glob, map_path in jobs:
    bill_files = pick_many(bills_glob)
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)

    lf = pl.scan_parquet(bill_files)
    cols = lf.collect_schema().names()
    if "account_identifier" not in cols:
        raise RuntimeError(f"Missing account_identifier in one of: {bills_glob}")

    delta_expr = choose_delta(cols).cast(pl.Float64).alias("delta_dollars")

    bills = lf.select([
        pl.col("account_identifier").cast(pl.Utf8),
        delta_expr,
    ]).collect()

    amap = pl.read_parquet(map_path).select([
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("geoid_bg").cast(pl.Utf8),
    ])

    joined = bills.join(amap, on="account_identifier", how="inner").filter(pl.col("geoid_bg").is_not_null())

    bg = joined.group_by("geoid_bg").agg([
        pl.col("delta_dollars").mean().alias("mean_delta"),
        pl.len().alias("n_households"),
    ])

    stats = bg.select([
        pl.len().alias("n_bgs"),
        pl.col("mean_delta").mean().alias("mean_of_bg_means"),
        pl.col("mean_delta").min().alias("min"),
        pl.col("mean_delta").max().alias("max"),
        (pl.col("mean_delta") > 0).mean().alias("share_pos"),
        (pl.col("mean_delta") < 0).mean().alias("share_neg"),
    ]).to_dict(as_series=False)

    print(f"\n[{label}] bills_files={len(bill_files)}")
    for k,v in stats.items():
        print(f"  {k}: {v[0]}")

    # Fail-loud guardrail: if everything is one sign AND magnitude is big, something is wrong.
    if (stats["share_pos"][0] == 0.0 or stats["share_neg"][0] == 0.0) and abs(stats["mean_of_bg_means"][0]) > 5.0:
        raise RuntimeError(f"Suspicious BG deltas for {label}: all one sign with large magnitude. Aborting export.")

    bg_pd = bg.to_pandas()
    bg_pd["geoid_bg"] = bg_pd["geoid_bg"].astype(str)

    merged = g.merge(bg_pd, left_on="GEOID", right_on="geoid_bg", how="inner")

    out_path = f"{OUT_DIR}/{label}_delta.geojson"
    merged.to_file(out_path, driver="GeoJSON")

    used = ",".join([os.path.basename(x) for x in bill_files])
    print(f"WROTE {out_path} | BG count: {len(merged)}")
    print(f"  bills: {used}")

print("\nDONE: 4 Felt-ready GeoJSON layers created.")
