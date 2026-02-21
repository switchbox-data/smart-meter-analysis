#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, MaxNLocator

from smart_meter_analysis.census import fetch_acs_data

# Use Times New Roman throughout all figures.
# NOTE: re-applied inside _plot_scatter after sns.set_theme(), which resets rcParams.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
})

logger = logging.getLogger(__name__)


# ----------------------------
# Defaults (EC2)
# ----------------------------

DEFAULT_BILLS_DIR = Path("/ebs/home/griffin_switch_box/pricing_pilot/bills_unscaled")
DEFAULT_OUT_DIR = Path("/ebs/home/griffin_switch_box/pricing_pilot/regression")

DEFAULT_ACCOUNT_BG_MAP_202301 = Path("/ebs/home/griffin_switch_box/pricing_pilot/account_bg_map_202301.parquet")
DEFAULT_ACCOUNT_BG_MAP_202307 = Path("/ebs/home/griffin_switch_box/pricing_pilot/account_bg_map_202307.parquet")

DEFAULT_CENSUS_CACHE = Path("/ebs/home/griffin_switch_box/pricing_pilot/census_min_income_acs2023.parquet")


# ----------------------------
# Label maps
# ----------------------------

CLASS_LABELS: dict[str, str] = {
    "sf_no_esh": "Single-Family, No Electric Space Heat",
    "mf_no_esh": "Multifamily, No Electric Space Heat",
    "sf_esh": "Single-Family, Electric Space Heat",
    "mf_esh": "Multifamily, Electric Space Heat",
}

COMPARISON_LABELS: dict[str, str] = {
    "stou_vs_flat": "Rate BEST vs Flat Rate",
    "dtou_vs_flat": "DTOU vs Flat Rate",
}

MONTH_PREFIXES: dict[str, str] = {
    "202301": "jan_2023",
    "202307": "jul_2023",
}


# ----------------------------
# Small utilities
# ----------------------------


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _dollar_formatter(x: float, _pos: int) -> str:
    return f"${x:,.0f}"


def _pct_formatter(x: float, _pos: int) -> str:
    return f"{x:.1f}%"


def _month_title(month: str) -> str:
    if month == "202301":
        return "January 2023"
    if month == "202307":
        return "July 2023"
    return month


def _y_label(*, pct: bool = False) -> str:
    if pct:
        return "Average Monthly Bill Change (%)"
    # mean_delta = block-group average of (Flat - Alt); positive = savings under alternative.
    return "Average Monthly Savings (Flat \u2212 Alt, $)"


def _class_label(code: str) -> str:
    """Human-readable delivery class name, e.g. 'sf_no_esh' → 'Single-Family, No Electric Space Heat'."""
    return CLASS_LABELS.get(code, code)


def _comparison_label(lbl: str) -> str:
    """Human-readable comparison label, e.g. 'stou_vs_flat' → 'Rate BEST vs Flat Rate'."""
    return COMPARISON_LABELS.get(lbl, lbl)


def _month_prefix(ym: str) -> str:
    """Short filename prefix for a month, e.g. '202301' → 'jan_2023'."""
    return MONTH_PREFIXES.get(ym, ym)


def _extract_class_from_path(path: Path) -> str | None:
    """Extract delivery class code from a bill filename.

    Expected pattern: {yyyymm}_flat_vs_{dtou|stou}_{class_code}.parquet
    Returns the class code segment, e.g. 'sf_no_esh', or None if unrecognised.
    """
    stem = path.stem
    for marker in ("vs_dtou_", "vs_stou_"):
        idx = stem.find(marker)
        if idx >= 0:
            return stem[idx + len(marker) :]
    return None


def _comparison_label_from_filename(name: str) -> str | None:
    """
    Pilot policy:
      - DTOU: filenames containing 'vs_dtou' (e.g. {yyyymm}_flat_vs_dtou_{sf_no_esh,...}.parquet)
      - STOU: filenames containing 'vs_stou' or 'vs_rate_best' (rate_best only when _scaled_)
    """
    n = name.lower()
    if "vs_dtou" in n:
        return "dtou_vs_flat"
    if "vs_stou" in n:
        return "stou_vs_flat"
    if "vs_rate_best" in n:
        return "stou_vs_flat" if "_scaled_" in n else None
    return None


def _discover_scenarios(bills_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """
    Returns mapping (yyyymm, comparison_label) -> list[parquet paths].
    File pattern: {yyyymm}_flat_vs_{dtou|stou}_{sf_no_esh,mf_no_esh,sf_esh,mf_esh}.parquet
    Months: 202301, 202307. Each scenario gets 4 delivery-class files.
    """
    out: dict[tuple[str, str], list[Path]] = {}
    for ym in ("202301", "202307"):
        for comp in ("dtou", "stou"):
            lbl = f"{comp}_vs_flat"
            files = sorted(bills_dir.glob(f"{ym}_flat_vs_{comp}_*.parquet"))
            if files:
                out[(ym, lbl)] = files
    return out


# ----------------------------
# Data prep
# ----------------------------


def _compute_household_delta(hh: pl.DataFrame) -> pl.DataFrame:
    """
    Standardizes household-level delta for pilot bills.

    ΔBill = Flat - Alternative (positive = savings under alternative), prefer:
      1) net_bill_diff_dollars
      2) bill_diff_dollars
      3) bill_b_dollars - bill_a_dollars
    """
    cols = set(hh.columns)

    if "net_bill_diff_dollars" in cols:
        delta = pl.col("net_bill_diff_dollars").cast(pl.Float64)
    elif "bill_diff_dollars" in cols:
        delta = pl.col("bill_diff_dollars").cast(pl.Float64)
    elif "bill_b_dollars" in cols and "bill_a_dollars" in cols:
        delta = pl.col("bill_b_dollars").cast(pl.Float64) - pl.col("bill_a_dollars").cast(pl.Float64)
    else:
        raise RuntimeError(
            "Could not compute household delta. Expected one of: net_bill_diff_dollars, bill_diff_dollars, "
            "or (bill_b_dollars & bill_a_dollars). "
            f"Got columns: {sorted(cols)}"
        )

    if "total_kwh" not in cols:
        raise RuntimeError("Expected total_kwh column in pilot bills (for mean_kwh hue/diagnostics).")

    return hh.with_columns([
        delta.alias("delta_dollars"),
        pl.col("total_kwh").cast(pl.Float64).alias("kwh"),
        pl.col("account_identifier").cast(pl.Utf8).alias("account_identifier"),
    ])


def _compute_household_pct_change(hh: pl.DataFrame) -> pl.DataFrame:
    """Standardizes household-level percentage change for pilot bills.

    pct_change = (flat_bill - alt_bill) / flat_bill * 100
    (positive = savings under alternative)

    Households where flat_bill <= 0 yield null and are dropped before return.

    Column priority mirrors _compute_household_delta:
      1) net_pct_savings   (pre-computed; matches net_bill_diff_dollars priority)
      2) pct_savings       (pre-computed; matches bill_diff_dollars priority)
      3) net_bill_diff_dollars / bill_a_dollars * 100
      4) bill_diff_dollars / bill_a_dollars * 100
      5) (bill_a_dollars - bill_b_dollars) / bill_a_dollars * 100
    """
    cols = set(hh.columns)

    if "net_pct_savings" in cols:
        pct_expr = pl.col("net_pct_savings").cast(pl.Float64)
    elif "pct_savings" in cols:
        pct_expr = pl.col("pct_savings").cast(pl.Float64)
    elif "net_bill_diff_dollars" in cols and "bill_a_dollars" in cols:
        pct_expr = (
            pl.when(pl.col("bill_a_dollars").cast(pl.Float64) > 0)
            .then(pl.col("net_bill_diff_dollars").cast(pl.Float64) / pl.col("bill_a_dollars").cast(pl.Float64) * 100)
            .otherwise(None)
        )
    elif "bill_diff_dollars" in cols and "bill_a_dollars" in cols:
        pct_expr = (
            pl.when(pl.col("bill_a_dollars").cast(pl.Float64) > 0)
            .then(pl.col("bill_diff_dollars").cast(pl.Float64) / pl.col("bill_a_dollars").cast(pl.Float64) * 100)
            .otherwise(None)
        )
    elif "bill_a_dollars" in cols and "bill_b_dollars" in cols:
        pct_expr = (
            pl.when(pl.col("bill_a_dollars").cast(pl.Float64) > 0)
            .then(
                (pl.col("bill_a_dollars").cast(pl.Float64) - pl.col("bill_b_dollars").cast(pl.Float64))
                / pl.col("bill_a_dollars").cast(pl.Float64)
                * 100
            )
            .otherwise(None)
        )
    else:
        raise RuntimeError(
            "Could not compute household pct change. Expected one of: net_pct_savings, pct_savings, "
            "(net_bill_diff_dollars & bill_a_dollars), (bill_diff_dollars & bill_a_dollars), "
            "or (bill_a_dollars & bill_b_dollars). "
            f"Got columns: {sorted(cols)}"
        )

    if "total_kwh" not in cols:
        raise RuntimeError("Expected total_kwh column in pilot bills (for mean_kwh hue/diagnostics).")

    # Reuse delta_dollars column name so all downstream aggregation/regression code
    # works without modification; units are percent, not dollars.
    out = hh.with_columns([
        pct_expr.alias("delta_dollars"),
        pl.col("total_kwh").cast(pl.Float64).alias("kwh"),
        pl.col("account_identifier").cast(pl.Utf8).alias("account_identifier"),
    ])
    n_before = out.height
    out = out.drop_nulls(["delta_dollars"])
    n_dropped = n_before - out.height
    if n_dropped:
        logger.info("_compute_household_pct_change: dropped %d/%d rows with flat_bill <= 0.", n_dropped, n_before)
    return out


def _load_account_bg_map(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Account->BG map not found: {path}")
    df = pl.read_parquet(path).select([
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("geoid_bg").cast(pl.Utf8),
    ])
    # De-dupe deterministically
    return df.unique(subset=["account_identifier"], keep="first")


def _attach_geoid_bg(hh: pl.DataFrame, acct_bg: pl.DataFrame) -> pl.DataFrame:
    out = hh.join(acct_bg, on="account_identifier", how="left")
    miss = out.select(pl.col("geoid_bg").is_null().mean()).item()
    if miss > 0:
        logger.warning("Missing geoid_bg for %.6f%% of household rows after map join.", miss * 100.0)
    return out.drop_nulls(["geoid_bg"])


def _aggregate_bg(hh: pl.DataFrame) -> pl.DataFrame:
    return hh.group_by("geoid_bg").agg([
        pl.mean("delta_dollars").alias("mean_delta"),
        pl.median("delta_dollars").alias("median_delta"),
        pl.mean("kwh").alias("mean_kwh"),
        pl.len().alias("n_households"),
    ])


def _assert_no_bg_duplicates(bg: pl.DataFrame, *, tag: str) -> None:
    n = bg.height
    n_unique = bg.select(pl.col("geoid_bg").n_unique()).item()
    if n != n_unique:
        raise RuntimeError(f"{tag}: BG duplicates detected: rows={n} unique_geoid_bg={n_unique}")


# ----------------------------
# Census income (minimal cache)
# ----------------------------


def _load_or_build_income_cache(
    *,
    cache_path: Path,
    needed_geoids: set[str],
    acs_year: int,
    state_fips: str,
    county_fips: str | None,
) -> pl.DataFrame:
    """
    Builds (or loads) a minimal census table:
      geoid_bg (12-digit) + median_income (float)
    using smart_meter_analysis.census.fetch_acs_data() and the engineered
    'median_household_income' feature (from B19013_001E).

    If county_fips is provided, only that county is fetched (faster).
    After fetching, we filter to needed_geoids and cache to parquet.
    """
    if cache_path.exists():
        logger.info("Loading cached income parquet: %s", cache_path)
        df = pl.read_parquet(cache_path)
        return df.select([pl.col("geoid_bg").cast(pl.Utf8), pl.col("median_income").cast(pl.Float64)])

    logger.info(
        "Building income cache via ACS API (year=%d state=%s county=%s).",
        acs_year,
        state_fips,
        county_fips or "*",
    )

    acs = fetch_acs_data(state_fips=state_fips, year=acs_year, county_fips=county_fips)

    if "GEOID" not in acs.columns or "median_household_income" not in acs.columns:
        raise RuntimeError(f"ACS pull did not contain required columns. Found columns: {acs.columns}")

    df = acs.select([
        pl.col("GEOID").cast(pl.Utf8).alias("geoid_bg"),
        pl.col("median_household_income").cast(pl.Float64, strict=False).alias("median_income"),
    ])

    if needed_geoids:
        df = df.filter(pl.col("geoid_bg").is_in(list(needed_geoids)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    logger.info("Wrote income cache: %s (rows=%d)", cache_path, df.height)
    return df


def _attach_income(bg: pl.DataFrame, income: pl.DataFrame, *, tag: str) -> pl.DataFrame:
    """
    Attach income via left join, then drop null income (and null mean_delta defensively),
    with explicit logging. No silent data loss.
    """
    n0 = bg.height
    out = bg.join(income, on="geoid_bg", how="left")
    n_missing_join = out.select(pl.col("median_income").is_null().sum()).item()

    # Drop nulls per requirements (income + mean_delta)
    out2 = out.drop_nulls(["median_income", "mean_delta"])
    n1 = out2.height

    logger.info(
        "%s: BG income join+sanitize: n0=%d missing_income=%d dropped=%d n=%d",
        tag,
        n0,
        int(n_missing_join),
        int(n0 - n1),
        n1,
    )

    if out2.height == 0:
        raise RuntimeError(f"{tag}: zero BG rows after dropping null income/mean_delta")

    # Fail-loud
    if out2.select(pl.col("median_income").is_null().any()).item():
        raise RuntimeError(f"{tag}: median_income nulls remain after drop_nulls")
    if out2.select(pl.col("mean_delta").is_null().any()).item():
        raise RuntimeError(f"{tag}: mean_delta nulls remain after drop_nulls")

    return out2


# ----------------------------
# Regression + plotting
# ----------------------------


def _sanitize_for_statsmodels(pdf: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    """
    Drop NaNs and assert finite X/Y before OLS. Log counts.
    """
    n0 = len(pdf)

    # Drop NaNs (defensive; should already be gone)
    null_x = int(pd.isna(pdf["median_income"]).sum())
    null_y = int(pd.isna(pdf["mean_delta"]).sum())
    pdf2 = pdf.dropna(subset=["median_income", "mean_delta"]).copy()

    # Finite check
    x = pdf2["median_income"].to_numpy(dtype=float, copy=False)
    y = pdf2["mean_delta"].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(x) & np.isfinite(y)
    bad = int((~finite).sum())
    if bad:
        pdf2 = pdf2.loc[finite].copy()

    n2 = len(pdf2)

    logger.info(
        "%s: statsmodels sanitize: n0=%d null_x=%d null_y=%d nonfinite_rows=%d n=%d",
        tag,
        n0,
        null_x,
        null_y,
        bad,
        n2,
    )

    if n2 == 0:
        raise RuntimeError(f"{tag}: empty regression frame after sanitization")

    # Fail-loud invariants
    if pdf2["median_income"].isna().any():
        raise RuntimeError(f"{tag}: median_income NaNs remain")
    if pdf2["mean_delta"].isna().any():
        raise RuntimeError(f"{tag}: mean_delta NaNs remain")
    if not np.isfinite(pdf2["median_income"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{tag}: median_income contains non-finite values")
    if not np.isfinite(pdf2["mean_delta"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{tag}: mean_delta contains non-finite values")

    return pdf2


def _run_ols_hc1(df: pl.DataFrame, *, tag: str) -> tuple[Any, pd.DataFrame]:
    """
    OLS(mean_delta ~ const + median_income) with HC1 robust SE.
    Logs N, no silent drops.
    """
    pdf = df.select(["median_income", "mean_delta", "mean_kwh", "n_households", "geoid_bg"]).to_pandas()
    pdf = _sanitize_for_statsmodels(pdf, tag=tag)

    y = pdf["mean_delta"].astype(float)
    X = sm.add_constant(pdf["median_income"].astype(float), has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HC1")

    logger.info(
        "%s: OLS(HC1) beta1=%.6g se=%.6g p=%.4g r2=%.4f n=%d",
        tag,
        float(model.params["median_income"]),
        float(model.bse["median_income"]),
        float(model.pvalues["median_income"]),
        float(model.rsquared),
        int(model.nobs),
    )

    return model, pdf


def _save_regression_json(model: Any, out_path: Path) -> None:
    res = {
        "beta_0": float(model.params["const"]),
        "beta_1": float(model.params["median_income"]),
        "se_beta_1": float(model.bse["median_income"]),
        "p_value_beta_1": float(model.pvalues["median_income"]),
        "r_squared": float(model.rsquared),
        "n": int(model.nobs),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))


def _plot_scatter(
    pdf: pd.DataFrame,
    model: Any,
    title: str,
    out_path: Path,
    *,
    pct: bool = False,
) -> None:
    """Produce one publication-ready scatterplot with OLS overlay.

    Parameters
    ----------
    title:
        Full two-line title already formatted as ``"Line1\\nBlock Group Level"``.
    pct:
        When True the y-axis is percentage change; adjusts label and tick formatter.
    """
    sns.set_theme(style="whitegrid")
    # sns.set_theme() resets rcParams; re-apply font settings immediately after.
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
    })

    fig, (ax, ax_stats) = plt.subplots(
        1,
        2,
        figsize=(10.5, 6.5),
        gridspec_kw={"width_ratios": [4.5, 1.5]},
    )

    sns.scatterplot(
        data=pdf,
        x="median_income",
        y="mean_delta",
        hue="mean_kwh",
        palette="viridis",
        edgecolor=None,
        alpha=0.8,
        legend=False,
        ax=ax,
    )

    # Colorbar: dot colour encodes block-group mean electricity use
    cbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cbar.set_label("Block-Group Mean Usage (kWh)", fontsize=10, fontfamily="serif")

    # Regression line, no CI band
    sns.regplot(
        data=pdf,
        x="median_income",
        y="mean_delta",
        scatter=False,
        ci=None,
        ax=ax,
    )

    # Remove any legend seaborn may have attached
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.set_xlabel("Median Household Income ($)")
    ax.set_ylabel(_y_label(pct=pct))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_formatter if pct else _dollar_formatter))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)

    # ── OLS stats panel (right side, out of the way) ──────────────────────
    beta1 = float(model.params["median_income"])
    se = float(model.bse["median_income"])
    pval = float(model.pvalues["median_income"])
    r2 = float(model.rsquared)

    stats_parts = [
        "Ordinary Least Squares Regression",
        "",
        f"\u03b2\u2081 = {beta1:.4g}",
        f"SE   = {se:.4g}",
        f"p     = {pval:.4g}",
        f"R\u00b2  = {r2:.4f}",
        f"N    = {len(pdf)}",
    ]

    ax_stats.axis("off")
    ax_stats.text(
        0.0,
        1.0,
        "\n".join(stats_parts),
        ha="left",
        va="top",
        fontsize=10,
        fontfamily="serif",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
    )

    # Safety net: force Times New Roman on every text element of the main axes.
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        item.set_fontfamily("serif")
        item.set_fontname("Times New Roman")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------
# BG-table construction helpers
# ----------------------------


def _build_aggregate_bg(
    paths: list[Path],
    acct_bg: pl.DataFrame,
    tag: str,
    *,
    pct: bool = False,
) -> tuple[pl.DataFrame, float]:
    """Pool all delivery-class files for one scenario; aggregate to BG level.

    Returns (bg_table, household_mean_delta_or_pct).
    """
    hh = pl.concat([pl.read_parquet(p) for p in paths], how="vertical")
    hh = _compute_household_pct_change(hh) if pct else _compute_household_delta(hh)
    hh_mean = float(hh.select(pl.col("delta_dollars").mean()).item())
    logger.info(
        "%s: household mean(delta_%s)=%.6g (should be ~0; upstream authoritative)",
        tag,
        "pct" if pct else "dollars",
        hh_mean,
    )
    hh = _attach_geoid_bg(hh, acct_bg)
    bg = _aggregate_bg(hh)
    _assert_no_bg_duplicates(bg, tag=tag)
    return bg, hh_mean


def _build_per_class_bg(
    path: Path,
    acct_bg: pl.DataFrame,
    tag: str,
    *,
    pct: bool = False,
) -> pl.DataFrame:
    """Aggregate a single delivery-class bill file to BG level."""
    hh = pl.read_parquet(path)
    hh = _compute_household_pct_change(hh) if pct else _compute_household_delta(hh)
    hh = _attach_geoid_bg(hh, acct_bg)
    bg = _aggregate_bg(hh)
    _assert_no_bg_duplicates(bg, tag=tag)
    return bg


# ----------------------------
# Main
# ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bills-dir", type=Path, default=DEFAULT_BILLS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    p.add_argument("--account-bg-map-202301", type=Path, default=DEFAULT_ACCOUNT_BG_MAP_202301)
    p.add_argument("--account-bg-map-202307", type=Path, default=DEFAULT_ACCOUNT_BG_MAP_202307)

    # Census fetch controls (we only need income)
    p.add_argument("--acs-year", type=int, default=2023)
    p.add_argument("--state-fips", type=str, default="17")
    # Default Cook only for speed; set to None to pull all IL counties if needed.
    p.add_argument(
        "--county-fips",
        type=str,
        default="031",
        help="County FIPS for ACS pull (default Cook=031). Use '' to pull all counties.",
    )
    p.add_argument("--income-cache", type=Path, default=DEFAULT_CENSUS_CACHE)

    p.add_argument(
        "--pct",
        action="store_true",
        default=False,
        help=(
            "Compute percentage change instead of absolute dollars. "
            "pct_change = (flat_bill - alt_bill) / flat_bill * 100. "
            "Households with flat_bill <= 0 are excluded. "
            "Appends _pct to all output filenames (PNGs, JSONs) and writes "
            "regression_summary_pct.txt instead of regression_summary.txt."
        ),
    )

    return p.parse_args()


def main() -> int:  # noqa: C901
    _configure_logging()
    args = parse_args()

    pct: bool = args.pct
    file_suffix = "_pct" if pct else ""

    bills_dir: Path = args.bills_dir
    out_dir: Path = args.out_dir
    out_reg_dir = out_dir
    out_fig_dir = out_dir / "figures"

    if not bills_dir.exists():
        raise FileNotFoundError(f"--bills-dir not found: {bills_dir}")

    scenarios = _discover_scenarios(bills_dir)
    if not scenarios:
        raise RuntimeError(f"No scenarios discovered in {bills_dir}. Expected files like 202301_flat_vs_dtou_*.parquet")

    logger.info("Discovered %d scenarios.", len(scenarios))

    # Load maps (fail-loud)
    acct_bg_202301 = _load_account_bg_map(args.account_bg_map_202301)
    acct_bg_202307 = _load_account_bg_map(args.account_bg_map_202307)

    # Collect needed BGs from all scenarios, then load income once.
    needed_geoids: set[str] = set()
    bg_agg: dict[tuple[str, str], pl.DataFrame] = {}
    bg_class: dict[tuple[str, str, str], pl.DataFrame] = {}
    hh_mean_deltas: dict[tuple[str, str], float] = {}

    for (ym, lbl), paths in sorted(scenarios.items()):
        tag = f"{ym}_{lbl}"
        logger.info("Scenario month=%s label=%s inputs=%d", ym, lbl, len(paths))
        acct_bg = acct_bg_202301 if ym == "202301" else acct_bg_202307

        # Aggregate (pool all delivery classes)
        bg, hh_mean = _build_aggregate_bg(paths, acct_bg, tag, pct=pct)
        bg_agg[(ym, lbl)] = bg
        hh_mean_deltas[(ym, lbl)] = hh_mean
        needed_geoids.update(bg["geoid_bg"].to_list())

        # Per delivery class
        for path in paths:
            class_code = _extract_class_from_path(path)
            if class_code is None:
                logger.warning("Cannot extract class code from %s; skipping per-class plot.", path.name)
                continue
            class_tag = f"{tag}_{class_code}"
            logger.info("Building per-class BG table: %s", class_tag)
            bg_class[(ym, lbl, class_code)] = _build_per_class_bg(path, acct_bg, class_tag, pct=pct)

    county_fips = args.county_fips.strip() or None

    income = _load_or_build_income_cache(
        cache_path=args.income_cache,
        needed_geoids=needed_geoids,
        acs_year=args.acs_year,
        state_fips=args.state_fips,
        county_fips=county_fips,
    )

    out_reg_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    ran_any = False

    # ── Aggregate plots (4: 2 months x 2 comparisons) ────────────────────
    for (ym, lbl), bg in sorted(bg_agg.items()):
        tag = f"{ym}_{lbl}_aggregate"
        bg2 = _attach_income(bg, income, tag=tag)

        # N reasonable (~1,600-1,900 BGs) -- log, don't hard fail unless extreme
        if not (1000 <= bg2.height <= 2500):
            logger.warning("%s: aggregate BG N outside expected 1000-2500: n=%d", tag, bg2.height)

        model, pdf = _run_ols_hc1(bg2, tag=tag)

        month_title = _month_title(ym)
        comp_label = _comparison_label(lbl)
        mp = _month_prefix(ym)

        title = f"{month_title} — {comp_label} — All Delivery Classes\nBlock Group Level"
        fig_stem = f"{mp}_{lbl}_aggregate{file_suffix}"

        _plot_scatter(pdf, model, title, out_fig_dir / f"{fig_stem}_scatter.png", pct=pct)
        _save_regression_json(model, out_reg_dir / f"{fig_stem}_regression.json")

        summary_lines.append(f"=== AGGREGATE: {month_title} — {comp_label} — All Delivery Classes ===")
        summary_lines.append(model.summary().as_text())
        summary_lines.append("")
        ran_any = True

    # ── Per-class plots (up to 16: 4 classes x 2 months x 2 comparisons) ─
    for (ym, lbl, class_code), bg in sorted(bg_class.items()):
        tag = f"{ym}_{lbl}_{class_code}"
        bg2 = _attach_income(bg, income, tag=tag)

        if bg2.height < 10:
            logger.warning("%s: only %d BG rows — skipping (too few for OLS).", tag, bg2.height)
            continue

        model, pdf = _run_ols_hc1(bg2, tag=tag)

        month_title = _month_title(ym)
        comp_label = _comparison_label(lbl)
        class_lbl = _class_label(class_code)
        mp = _month_prefix(ym)

        title = f"{month_title} — {comp_label} — {class_lbl}\nBlock Group Level"
        fig_stem = f"{mp}_{lbl}_{class_code}{file_suffix}"

        _plot_scatter(
            pdf,
            model,
            title,
            out_fig_dir / f"{fig_stem}_scatter.png",
            pct=pct,
        )
        _save_regression_json(model, out_reg_dir / f"{fig_stem}_regression.json")

        summary_lines.append(f"=== {class_lbl}: {month_title} — {comp_label} ===")
        summary_lines.append(model.summary().as_text())
        summary_lines.append("")
        ran_any = True

    if not ran_any:
        raise RuntimeError("No regressions were run (likely due to too few BGs after income join). Check logs above.")

    if pct:
        summary_lines = [
            "=== Percentage-change regressions (--pct) ===",
            "Y variable: mean_pct_change = mean((flat_bill - alt_bill) / flat_bill * 100) per block group",
            "Households with flat_bill <= 0 excluded.",
            "",
            *summary_lines,
        ]

    summary_fname = f"regression_summary{file_suffix}.txt"
    (out_reg_dir / summary_fname).write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Wrote %s -> %s", summary_fname, out_reg_dir / summary_fname)
    logger.info("Figures dir -> %s", out_fig_dir)
    logger.info("Income cache -> %s", args.income_cache)

    # Final validation summary (explicit, no silent assumptions)
    for (ym, lbl), v in sorted(hh_mean_deltas.items()):
        logger.info("VALIDATION: %s_%s household mean(delta_%s)=%.6g", ym, lbl, "pct" if pct else "dollars", v)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
