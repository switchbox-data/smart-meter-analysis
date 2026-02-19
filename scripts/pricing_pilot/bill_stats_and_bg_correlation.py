#!/usr/bin/env python3
"""
Report for ~/pricing_pilot/bills_unscaled/:
1. Per file: row count, mean bill difference, median bill difference.
2. Join to account_bg_map, aggregate mean bill_diff_dollars by block group (geoid_bg).
3. For each (month, delivery_class): Pearson correlation between DTOU and Rate BEST (STOU)
   mean deltas at block group level — report all 8 in a table.
4. For each of the 16 files: share of block groups where mean delta is positive (savings)
   vs negative (losses).
"""
from pathlib import Path
import pandas as pd
import numpy as np

BILLS_DIR = Path.home() / "pricing_pilot" / "bills_unscaled"
MAP_202301 = Path.home() / "pricing_pilot" / "account_bg_map_202301.parquet"
MAP_202307 = Path.home() / "pricing_pilot" / "account_bg_map_202307.parquet"
# Prefer bill_diff_dollars per user; fallback to net_bill_diff_dollars
DELTA_COL_CANDIDATES = ("bill_diff_dollars", "net_bill_diff_dollars")

CLASSES_ORDER = ("sf_no_esh", "mf_no_esh", "sf_esh", "mf_esh")


def _pick_delta_col(df: pd.DataFrame) -> str:
    for c in DELTA_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No delta column in {list(df.columns)}; expected one of {DELTA_COL_CANDIDATES}")


def parse_name(name: str):
    # e.g. 202301_flat_vs_dtou_mf_esh.parquet -> (202301, dtou, mf_esh)
    stem = name.replace(".parquet", "")
    month = stem[:6]
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
        dc = "unknown"
    return month, rate, dc


def main():
    bills_dir = BILLS_DIR
    if not bills_dir.exists():
        raise FileNotFoundError(f"Bills dir not found: {bills_dir}")

    files = sorted(bills_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {bills_dir}")

    # Resolve delta column from first file
    sample = pd.read_parquet(files[0])
    DELTA_COL = _pick_delta_col(sample)
    del sample

    # --- 1. Per-file stats ---
    print("=" * 70)
    print("1. PER-FILE STATS (row count, mean bill diff, median bill diff)")
    print("   Bill difference column:", DELTA_COL)
    print("=" * 70)

    for f in files:
        df = pd.read_parquet(f)
        n = len(df)
        if n == 0:
            mean_d, median_d = np.nan, np.nan
        else:
            mean_d = df[DELTA_COL].mean()
            median_d = df[DELTA_COL].median()
        print(f"  {f.name}")
        print(f"    row_count={n}, mean_bill_diff={mean_d:.4f}, median_bill_diff={median_d:.4f}")

    # --- 2. Aggregate to block group (mean delta per BG) ---
    print()
    print("=" * 70)
    print("2. BLOCK GROUP MEAN DELTAS (per file, joined with account_bg_map)")
    print("=" * 70)

    maps = {"202301": pd.read_parquet(MAP_202301), "202307": pd.read_parquet(MAP_202307)}

    bg_means = {}  # (month, rate, dc) -> DataFrame with geoid_bg, mean_delta

    for f in files:
        month, rate, dc = parse_name(f.name)
        df = pd.read_parquet(f)
        if len(df) == 0:
            bg_means[(month, rate, dc)] = pd.DataFrame(columns=["geoid_bg", "mean_delta"])
            continue
        m = maps[month]
        merged = df[["account_identifier", DELTA_COL]].merge(
            m[["account_identifier", "geoid_bg"]], on="account_identifier", how="inner"
        )
        bg = merged.groupby("geoid_bg", as_index=False)[DELTA_COL].mean().rename(
            columns={DELTA_COL: "mean_delta"}
        )
        bg_means[(month, rate, dc)] = bg
        print(f"  {f.name} -> {len(bg)} block groups")

    # --- 3. Pearson correlation table: DTOU vs Rate BEST (STOU) per (month, delivery_class) ---
    print()
    print("=" * 70)
    print("3. PEARSON CORRELATION: DTOU vs Rate BEST (STOU) mean deltas by block group")
    print("   (2 months × 4 classes = 8 correlations)")
    print("=" * 70)

    corr_rows = []
    for month in ("202301", "202307"):
        for dc in CLASSES_ORDER:
            key_dtou = (month, "dtou", dc)
            key_stou = (month, "stou", dc)
            bg_dtou = bg_means.get(key_dtou)
            bg_stou = bg_means.get(key_stou)
            if bg_dtou is None or bg_stou is None or len(bg_dtou) == 0 or len(bg_stou) == 0:
                r, n_bg = np.nan, 0
                corr_rows.append({"month": month, "delivery_class": dc, "pearson_r": r, "n_bg": n_bg})
                continue
            combined = bg_dtou.rename(columns={"mean_delta": "mean_delta_dtou"}).merge(
                bg_stou.rename(columns={"mean_delta": "mean_delta_stou"}),
                on="geoid_bg",
                how="inner",
            )
            n_bg = len(combined)
            if n_bg < 2:
                r = np.nan
            else:
                r = combined["mean_delta_dtou"].corr(combined["mean_delta_stou"])
            corr_rows.append({"month": month, "delivery_class": dc, "pearson_r": r, "n_bg": n_bg})

    corr_df = pd.DataFrame(corr_rows)
    # Pivot table: rows = month, columns = delivery_class, values = pearson_r
    pivot = corr_df.pivot(index="month", columns="delivery_class", values="pearson_r")
    pivot = pivot.reindex(columns=CLASSES_ORDER)
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"))
    print()

    # --- 4. Share of block groups with savings (mean_delta > 0) vs losses (mean_delta < 0) per file ---
    print("=" * 70)
    print("4. SHARE OF BLOCK GROUPS: positive mean delta (savings) vs negative (losses)")
    print("   Per file (16 files). July all-losses check: DTOU and/or Rate BEST.")
    print("=" * 70)

    share_rows = []
    for f in files:
        month, rate, dc = parse_name(f.name)
        bg = bg_means.get((month, rate, dc))
        if bg is None or len(bg) == 0:
            share_pos = share_neg = np.nan
            n_bg = 0
        else:
            n_bg = len(bg)
            pos = (bg["mean_delta"] > 0).sum()
            neg = (bg["mean_delta"] < 0).sum()
            share_pos = pos / n_bg
            share_neg = neg / n_bg
        share_rows.append({
            "file": f.name,
            "month": month,
            "rate": rate,
            "delivery_class": dc,
            "n_bg": n_bg,
            "pct_bg_savings": share_pos if np.isfinite(share_pos) else np.nan,
            "pct_bg_losses": share_neg if np.isfinite(share_neg) else np.nan,
        })
        pct_s = f"{100 * share_pos:.2f}%" if np.isfinite(share_pos) else "N/A"
        pct_l = f"{100 * share_neg:.2f}%" if np.isfinite(share_neg) else "N/A"
        print(f"  {f.name}")
        print(f"    n_bg={n_bg}, share BGs with savings (Δ>0)={pct_s}, share BGs with losses (Δ<0)={pct_l}")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
