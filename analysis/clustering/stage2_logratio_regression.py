#!/usr/bin/env python3
"""Stage 2: Block-Group-Level Log-Ratio Regression of Cluster Composition (HOUSEHOLD-DAY UNITS)

Aggregates household-day cluster assignments to Census block groups and fits
additive log-ratio (ALR) regressions of Laplace-smoothed cluster proportions
relative to a baseline cluster, using WLS with weights = total household-day
observations per block group.

Regulatory deliverables (written to output_dir):
- regression_results.parquet            # long coefficient table (primary)
- regression_diagnostics.json           # R² + residual stats + data-loss metrics
- regression_diagnostics.png            # diagnostic plots (per outcome)
- interpretation_guide.md               # coefficient interpretation guide
- stage2_metadata.json                  # full provenance (parameters, inputs, sizes)
- regression_data_blockgroups_wide.parquet
- statsmodels_summaries_wls.txt
- statsmodels_summaries_ols.txt (optional)
- regression_report_logratio_blockgroups.txt
- stage2_manifest.json (+ predictor lists) via write_stage2_manifest

Design:
- Household-day inputs processed lazily; only block-group aggregates collected.
- Laplace smoothing alpha=0.5 default.
- Smoothing denominator uses global K (from full cluster label set),
  not K among post-filter clusters.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence

from smart_meter_analysis.census import fetch_census_data
from smart_meter_analysis.census_specs import STAGE2_PREDICTORS_47
from smart_meter_analysis.run_manifest import write_stage2_manifest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------
# IO + aggregation helpers
# ----------------------------


def _validate_clusters_schema(lf: pl.LazyFrame, path: Path) -> list[str]:
    schema = lf.collect_schema()
    required = ["account_identifier", "zip_code", "cluster"]
    missing = [c for c in required if c not in schema.names()]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    cluster_type = schema.get("cluster")
    if cluster_type not in (pl.Int32, pl.Int64, pl.Utf8, pl.String):
        raise ValueError(f"cluster column must be integer or string, got {cluster_type}")

    return required


def _compute_dominance_stats_streaming(path: Path) -> dict[str, Any]:
    """Household dominance = share of a household's sampled days assigned to its most common cluster.
    Computed streaming (does not materialize the full household-day table).
    """
    lf = pl.scan_parquet(path).select(["account_identifier", "cluster"])
    dom = (
        lf.group_by(["account_identifier", "cluster"])
        .agg(pl.len().alias("n"))
        .group_by("account_identifier")
        .agg(
            pl.col("n").sum().alias("total_days"),
            pl.col("n").max().alias("max_in_cluster"),
        )
        .with_columns((pl.col("max_in_cluster") / pl.col("total_days")).alias("dominance"))
        .select(
            pl.col("dominance").mean().alias("dominance_mean"),
            pl.col("dominance").median().alias("dominance_median"),
            (pl.col("dominance") > 0.5).mean().alias("pct_above_50"),
        )
        .collect(streaming=True)  # type: ignore[call-overload]
        .to_dicts()[0]
    )
    return dom  # type: ignore[no-any-return]


def load_household_day_clusters(path: Path) -> tuple[pl.LazyFrame, dict[str, Any], dict[str, Any]]:
    """Load household-day cluster assignments as a LazyFrame with required columns only.

    Returns:
        household_days_lf: lazy frame (account_identifier, zip_code, cluster[, date])
        basic_stats: small dict of counts for logging/manifesting
        dominance_stats: household-level dominance metrics (streaming)

    """
    lf = pl.scan_parquet(path)
    required = _validate_clusters_schema(lf, path)

    cols = required.copy()
    schema_names = lf.collect_schema().names()
    if "date" in schema_names:
        cols.append("date")

    household_days_lf = lf.select(cols)

    stats = (
        household_days_lf.select(
            pl.len().alias("n_household_days"),
            pl.col("account_identifier").n_unique().alias("n_households"),
            pl.col("cluster").n_unique().alias("n_clusters"),
        )
        .collect(streaming=True)  # type: ignore[call-overload]
        .to_dicts()[0]
    )

    dominance_stats = _compute_dominance_stats_streaming(path)

    logger.info(
        "Loaded clusters: %s household-days, %s households, %s clusters",
        f"{stats['n_household_days']:,}",
        f"{stats['n_households']:,}",
        stats["n_clusters"],
    )

    return household_days_lf, stats, dominance_stats


def choose_baseline_cluster(household_days_lf: pl.LazyFrame) -> str:
    """Baseline is the most frequent cluster by household-day observations."""
    dist = (
        household_days_lf.group_by("cluster")
        .agg(pl.len().alias("n_obs"))
        .sort("n_obs", descending=True)
        .collect(streaming=True)  # type: ignore[call-overload]
    )
    if dist.is_empty():
        raise ValueError("No cluster assignments found; cannot choose baseline.")
    return str(dist["cluster"][0])


def list_cluster_labels(household_days_lf: pl.LazyFrame) -> list[str]:
    """Return stable, sorted cluster labels present in the full household-day data.
    Used to lock global K for smoothing.
    """
    labels = (
        household_days_lf.select(pl.col("cluster").cast(pl.Utf8).alias("cluster"))
        .unique()
        .collect(streaming=True)  # type: ignore[call-overload]
        .get_column("cluster")
        .to_list()
    )
    return sorted([str(x) for x in labels if x is not None])


def load_crosswalk_one_to_one(
    crosswalk_path: Path,
    *,
    zip_codes_lf: pl.LazyFrame | None = None,
) -> tuple[pl.LazyFrame, dict[str, Any]]:
    """Load ZIP+4 -> block group crosswalk and enforce deterministic 1-to-1 linkage.

    Crosswalk expected to contain: Zip, Zip4, CensusKey2023 (tab-separated).
    Output columns: zip_code, block_group_geoid

    Returns:
        mapping_lf: 1 row per zip_code with deterministic BG assignment
        metrics: fanout + size metrics for provenance

    """
    logger.info("Loading crosswalk: %s", crosswalk_path)

    lf = (
        pl.scan_csv(crosswalk_path, separator="\t", infer_schema_length=10000)
        .with_columns([
            (pl.col("Zip").cast(pl.Utf8).str.zfill(5) + "-" + pl.col("Zip4").cast(pl.Utf8).str.zfill(4)).alias(
                "zip_code",
            ),
            pl.col("CensusKey2023").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("block_group_geoid"),
        ])
        .select(["zip_code", "block_group_geoid"])
        .drop_nulls()
    )

    if zip_codes_lf is not None:
        lf = lf.join(zip_codes_lf, on="zip_code", how="semi")

    metrics = (
        lf.select(
            pl.len().alias("n_rows"),
            pl.col("zip_code").n_unique().alias("n_zip4"),
            pl.col("block_group_geoid").n_unique().alias("n_bg"),
        )
        .collect(streaming=True)  # type: ignore[call-overload]
        .to_dicts()[0]
    )

    fanout = (
        lf.group_by("zip_code")
        .agg(pl.col("block_group_geoid").n_unique().alias("n_bg"))
        .filter(pl.col("n_bg") > 1)
        .select(pl.len().alias("n_zip4_multi_bg"))
        .collect(streaming=True)  # type: ignore[call-overload]
        .item()
    )
    metrics["n_zip4_multi_bg"] = int(fanout)

    if fanout:
        logger.warning(
            "Crosswalk fan-out: %s ZIP+4s map to multiple block groups; using smallest GEOID per ZIP+4.",
            f"{fanout:,}",
        )

    # Deterministic 1-to-1 mapping: smallest GEOID per ZIP+4
    mapping = lf.group_by("zip_code").agg(pl.col("block_group_geoid").min().alias("block_group_geoid"))
    return mapping, metrics


def attach_block_groups(
    household_days_lf: pl.LazyFrame,
    crosswalk_lf: pl.LazyFrame,
) -> tuple[pl.LazyFrame, dict[str, Any]]:
    """Join ZIP+4 -> block group onto household-day cluster assignments.

    Returns:
        joined_lf: household-day LF with block_group_geoid (nulls dropped)
        metrics: join/drop metrics for provenance

    """
    before = household_days_lf.select(pl.len().alias("n")).collect(streaming=True).item()  # type: ignore[call-overload]
    joined = household_days_lf.join(crosswalk_lf, on="zip_code", how="left")
    after_nonnull = (
        joined.select(pl.col("block_group_geoid").is_not_null().sum().alias("n")).collect(streaming=True).item()  # type: ignore[call-overload]
    )

    dropped = int(before - after_nonnull)
    metrics = {
        "household_days_before_crosswalk": int(before),
        "household_days_after_crosswalk_nonnull": int(after_nonnull),
        "household_days_dropped_missing_crosswalk": dropped,
        "pct_dropped_missing_crosswalk": float(dropped / before) if before else 0.0,
    }

    if dropped:
        logger.warning(
            "Dropped %s household-days (%.2f%%) due to missing ZIP+4 crosswalk mapping.",
            f"{dropped:,}",
            metrics["pct_dropped_missing_crosswalk"] * 100.0,
        )

    return joined.drop_nulls("block_group_geoid"), metrics


def aggregate_blockgroup_composition(
    household_days_bg_lf: pl.LazyFrame,
    *,
    cluster_labels: Iterable[str],
) -> pl.DataFrame:
    """Aggregate to block-group-level composition.

    Returns one row per block group with:
      - total_obs (household-day count; WLS weight)
      - total_households (unique households; descriptive)
      - cluster_<label> counts (wide) for ALL labels in cluster_labels
    """
    counts_long = (
        household_days_bg_lf.group_by(["block_group_geoid", "cluster"])
        .agg(pl.len().alias("n_obs"))
        .collect(streaming=True)  # type: ignore[call-overload]
        .with_columns(pl.col("cluster").cast(pl.Utf8))
    )

    totals_and_hh = (
        household_days_bg_lf.group_by("block_group_geoid")
        .agg([
            pl.len().alias("total_obs"),
            pl.col("account_identifier").n_unique().alias("total_households"),
        ])
        .collect(streaming=True)  # type: ignore[call-overload]
    )

    wide = counts_long.pivot(
        index="block_group_geoid",
        columns="cluster",
        values="n_obs",
        aggregate_function="first",
    ).fill_null(0)

    # Ensure all cluster columns exist (global K locked)
    for lab in cluster_labels:
        if lab not in wide.columns:
            wide = wide.with_columns(pl.lit(0).alias(lab))

    cluster_cols_present = sorted([c for c in wide.columns if c != "block_group_geoid"])
    expected = sorted([str(x) for x in cluster_labels])
    if cluster_cols_present != expected:
        raise ValueError(f"Cluster column mismatch: expected {expected}, got {cluster_cols_present}")

    # Rename to cluster_<label>
    cluster_cols = [c for c in wide.columns if c != "block_group_geoid"]
    wide = wide.rename({c: f"cluster_{c}" for c in cluster_cols})

    out = totals_and_hh.join(wide, on="block_group_geoid", how="left").fill_null(0)

    logger.info("Aggregated: %s block groups, %s cluster columns", f"{out.height:,}", len(cluster_cols))
    return out  # type: ignore[no-any-return]


# ----------------------------
# Census + regression helpers
# ----------------------------


def fetch_or_load_census(
    cache_path: Path,
    *,
    state_fips: str,
    acs_year: int,
    force_fetch: bool,
) -> pl.DataFrame:
    if cache_path.exists() and not force_fetch:
        logger.info("Loading cached census data: %s", cache_path)
        return pl.read_parquet(cache_path)

    logger.info("Fetching census data (state=%s, ACS=%s)", state_fips, acs_year)
    df = fetch_census_data(state_fips=state_fips, acs_year=acs_year)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    logger.info("Cached census data: %s", cache_path)
    return df


def detect_predictors(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    predictors = list(STAGE2_PREDICTORS_47)
    available = set(df.columns)

    used: list[str] = []
    excluded_all_null: list[str] = []

    for p in predictors:
        if p not in available:
            continue
        col = df.get_column(p)
        if col.null_count() == df.height:
            excluded_all_null.append(p)
            continue
        used.append(p)

    return used, excluded_all_null


def prepare_regression_df(
    df: pl.DataFrame,
    predictors: list[str],
    *,
    min_obs_per_bg: int,
) -> tuple[pl.DataFrame, list[str], dict[str, Any]]:
    """Filter to statistically stable block groups and complete-case predictors.

    Returns:
        filtered_df
        predictors_used
        metrics (row counts dropped, etc.)

    """
    metrics: dict[str, Any] = {}

    n0 = int(df.height)
    df = df.filter(pl.col("total_obs") >= min_obs_per_bg)
    n1 = int(df.height)

    usable = []
    for p in predictors:
        if p in df.columns and df.get_column(p).null_count() < df.height:
            usable.append(p)

    if not usable:
        raise ValueError("No usable predictors available after filtering.")

    # Drop rows with any null in usable predictors (complete-case)
    df = df.filter(~pl.any_horizontal(pl.col(usable).is_null()))
    n2 = int(df.height)

    metrics["rows_before_filters"] = n0
    metrics["rows_after_min_obs_filter"] = n1
    metrics["rows_after_complete_case_filter"] = n2
    metrics["rows_dropped_min_obs"] = int(n0 - n1)
    metrics["rows_dropped_complete_case"] = int(n1 - n2)
    return df, usable, metrics


def add_smoothed_logratios(
    df: pl.DataFrame,
    *,
    baseline_cluster: str,
    alpha: float,
    k_total: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Add smoothed proportions and log-ratios using a GLOBAL K (k_total)."""
    cluster_cols = sorted([c for c in df.columns if c.startswith("cluster_")])
    if len(cluster_cols) < 2:
        raise ValueError("Need at least 2 clusters to form log-ratios.")

    baseline_col = f"cluster_{baseline_cluster}"
    if baseline_col not in cluster_cols:
        available = [c.replace("cluster_", "") for c in cluster_cols]
        raise ValueError(f"Baseline cluster '{baseline_cluster}' not found in data. Available clusters: {available}")

    denom = pl.col("total_obs") + (k_total * alpha)
    df = df.with_columns([((pl.col(c) + alpha) / denom).alias(f"p_{c}") for c in cluster_cols])

    base_p = f"p_{baseline_col}"

    outcomes: list[str] = []
    for c in cluster_cols:
        label = c.replace("cluster_", "")
        if label == baseline_cluster:
            continue
        out = f"log_ratio_{label}"
        df = df.with_columns((pl.col(f"p_{c}") / pl.col(base_p)).log().alias(out))
        outcomes.append(out)

    return df, outcomes


def fit_models(
    df: pl.DataFrame,
    *,
    predictors: list[str],
    outcomes: list[str],
    weight_col: str = "total_obs",
    standardize: bool = False,
    run_ols: bool = True,
) -> dict[str, Any]:
    # Stable ordering for reproducibility of row-index-based diagnostics
    df = df.sort("block_group_geoid")

    X_raw = df.select(predictors).to_numpy().astype(np.float64)
    w_raw = df.get_column(weight_col).to_numpy().astype(np.float64)

    valid = np.isfinite(X_raw).all(axis=1) & np.isfinite(w_raw) & (w_raw > 0)
    X = X_raw[valid]
    w = w_raw[valid]

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X = sm.add_constant(X)
    param_names = ["const", *predictors]

    results: dict[str, Any] = {
        "n_rows_input": int(df.height),
        "n_valid_rows": int(valid.sum()),
        "weight_col": weight_col,
        "param_names": param_names,
        "standardized": bool(standardize),
    }

    if scaler is not None:
        results["scaler"] = {
            "means": {p: float(m) for p, m in zip(predictors, scaler.mean_)},
            "scales": {p: float(s) for p, s in zip(predictors, scaler.scale_)},
        }

    wls_models: dict[str, Any] = {}
    ols_models: dict[str, Any] = {}

    for out in outcomes:
        y = df.get_column(out).to_numpy().astype(np.float64)[valid]

        m_wls = sm.WLS(y, X, weights=w).fit()
        wls_models[out] = m_wls

        if run_ols:
            m_ols = sm.OLS(y, X).fit()
            ols_models[out] = m_ols

    results["wls"] = wls_models
    results["ols"] = ols_models
    results["valid_mask"] = valid  # used for diagnostics alignment
    results["block_group_geoid_sorted"] = df.get_column("block_group_geoid").to_list()
    return results


def _summarize_models_for_json(models: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, m in models.items():
        out[name] = {
            "params": {k: float(v) for k, v in zip(m.model.exog_names, m.params)},
            "bse": {k: float(v) for k, v in zip(m.model.exog_names, m.bse)},
            "tvalues": {k: float(v) for k, v in zip(m.model.exog_names, m.tvalues)},
            "pvalues": {k: float(v) for k, v in zip(m.model.exog_names, m.pvalues)},
            "rsquared": float(getattr(m, "rsquared", np.nan)),
            "nobs": int(m.nobs),
        }
    return out


def _residual_diagnostics(m: Any) -> dict[str, Any]:
    resid = np.asarray(m.resid, dtype=np.float64)
    fitted = np.asarray(m.fittedvalues, dtype=np.float64)

    def _safe(x: float) -> float:
        return float(x) if np.isfinite(x) else float("nan")

    return {
        "resid_mean": _safe(float(np.mean(resid))),
        "resid_std": _safe(float(np.std(resid, ddof=1))) if resid.size > 1 else float("nan"),
        "resid_min": _safe(float(np.min(resid))),
        "resid_max": _safe(float(np.max(resid))),
        "fitted_min": _safe(float(np.min(fitted))),
        "fitted_max": _safe(float(np.max(fitted))),
    }


def write_coefficients_table_parquet(
    out_path: Path,
    *,
    wls_models: dict[str, Any],
    baseline_cluster: str,
    alpha: float,
    weight_col: str,
    standardized: bool,
    param_names: list[str],
) -> None:
    """Required output: coefficient table for each non-baseline cluster outcome.

    Schema:
      cluster, predictor, coefficient, std_error, t_stat, p_value, r_squared, nobs,
      baseline_cluster, alpha, weight_col, standardized
    """
    rows: list[dict[str, Any]] = []
    for out_name, m in wls_models.items():
        # out_name format: log_ratio_<clusterlabel>
        cluster_label = out_name.replace("log_ratio_", "")
        rsq = float(getattr(m, "rsquared", np.nan))
        nobs = int(m.nobs)

        for pred, coef, se, tv, pv in zip(param_names, m.params, m.bse, m.tvalues, m.pvalues):
            rows.append({
                "cluster": cluster_label,
                "predictor": str(pred),
                "coefficient": float(coef),
                "std_error": float(se),
                "t_stat": float(tv),
                "p_value": float(pv),
                "r_squared": rsq,
                "nobs": nobs,
                "baseline_cluster": str(baseline_cluster),
                "alpha": float(alpha),
                "weight_col": str(weight_col),
                "standardized": bool(standardized),
            })

    pl.DataFrame(rows).write_parquet(out_path)


def write_diagnostic_plots(
    out_path: Path,
    *,
    models: dict[str, Any],
) -> None:
    """Write a single multi-page PNG with per-outcome diagnostics:
    - Residuals vs fitted
    - QQ plot
    - Leverage vs standardized residuals (OLS influence proxy)
    """
    # Create one figure per outcome and save sequentially into one PNG is not supported directly.
    # Instead: write a single tall figure with rows per outcome (3 panels each).
    n_out = max(len(models), 1)
    fig, axes = plt.subplots(n_out, 3, figsize=(15, 4 * n_out))

    if n_out == 1:
        axes = np.array([axes])

    for i, (name, m) in enumerate(models.items()):
        resid = np.asarray(m.resid, dtype=np.float64)
        fitted = np.asarray(m.fittedvalues, dtype=np.float64)

        ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]
        ax0.scatter(fitted, resid, s=8)
        ax0.axhline(0.0, linewidth=1)
        ax0.set_title(f"{name}: Residuals vs Fitted")
        ax0.set_xlabel("Fitted")
        ax0.set_ylabel("Residuals")

        qqplot(resid, line="45", ax=ax1)
        ax1.set_title(f"{name}: Q-Q Plot")

        # Influence diagnostics: statsmodels influence works reliably for OLS.
        # For WLS, we approximate leverage via OLSInfluence on the same fitted model structure.
        try:
            infl = OLSInfluence(m)
            lev = np.asarray(infl.hat_matrix_diag, dtype=np.float64)
            sresid = np.asarray(infl.resid_studentized_internal, dtype=np.float64)
            ax2.scatter(lev, sresid, s=8)
            ax2.set_title(f"{name}: Leverage vs Std Residuals")
            ax2.set_xlabel("Leverage (hat diag)")
            ax2.set_ylabel("Studentized residual")
        except Exception:
            ax2.text(0.5, 0.5, "Influence plot unavailable", ha="center", va="center")
            ax2.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_interpretation_guide(
    out_path: Path,
    *,
    baseline_cluster: str,
) -> None:
    md = f"""# How to Read Stage 2 Coefficients (ALR / Log-Ratio Regression)

## Model
For each non-baseline cluster q:

log(π_q / π_{{baseline}}) = β0_q + β_q · X + ε

This is **not** logistic regression. Coefficients are **not odds ratios**.

## Interpretation
For a one-unit increase in predictor X_k:

- The log-ratio changes by β_k
- The **ratio multiplier** is exp(β_k)

Meaning:
- exp(β_k) = 1.10 implies the ratio (π_q / π_{{baseline}}) is multiplied by 1.10
- exp(β_k) = 0.90 implies the ratio is multiplied by 0.90 (a 10% decrease)

Baseline cluster: **{baseline_cluster}**

## Notes
- Proportions π are Laplace-smoothed prior to log transform to avoid log(0).
- WLS weights use total household-day observations per block group.
"""
    out_path.write_text(md, encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------
# CLI
# ----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: Block-group-level log-ratio regression using household-day units.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--clusters", type=Path, required=True, help="cluster_assignments.parquet")
    parser.add_argument("--crosswalk", type=Path, required=True, help="ZIP+4 → block-group crosswalk")

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Stage 2 output directory (recommend: data/runs/{run_name}/stage2)",
    )

    parser.add_argument(
        "--census-cache",
        type=Path,
        default=None,
        help="Per-run Census cache path. If omitted, defaults to {output_dir}/census_cache_{state}_{acs_year}.parquet",
    )
    parser.add_argument("--fetch-census", action="store_true", help="Force re-fetch Census data")
    parser.add_argument("--state-fips", default="17")
    parser.add_argument("--acs-year", type=int, default=2023)

    # Statistical stability: 50 obs ≈ minimum for reliable proportion estimates with k=4.
    parser.add_argument(
        "--min-obs-per-bg",
        type=int,
        default=50,
        help="Minimum household-day obs per block group (default: 50 for statistical stability)",
    )

    parser.add_argument("--standardize", action="store_true", help="Standardize predictors before regression")
    parser.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing pseudocount")

    parser.add_argument(
        "--baseline-cluster",
        type=str,
        default=None,
        help="Baseline cluster label (default: most frequent by household-day observations)",
    )

    parser.add_argument("--no-ols", action="store_true", help="Skip OLS robustness check (only run WLS)")
    parser.add_argument(
        "--predictors-from",
        type=str,
        default=None,
        help="Optional: path to a file listing predictors to force an exact list (one per line).",
    )

    args = parser.parse_args()

    if not args.clusters.exists():
        raise FileNotFoundError(f"Cluster assignments not found: {args.clusters}")
    if not args.crosswalk.exists():
        raise FileNotFoundError(f"Crosswalk not found: {args.crosswalk}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    census_cache = args.census_cache
    if census_cache is None:
        census_cache = args.output_dir / f"census_cache_{args.state_fips}_{args.acs_year}.parquet"

    # ---- Load clusters ----
    household_days_lf, _stats, dominance_stats = load_household_day_clusters(args.clusters)

    cluster_labels = list_cluster_labels(household_days_lf)
    k_total = len(cluster_labels)
    if k_total < 2:
        raise ValueError("Need at least 2 clusters for Stage 2.")
    logger.info("Detected global K=%s clusters: %s", k_total, cluster_labels)

    baseline_cluster = args.baseline_cluster or choose_baseline_cluster(household_days_lf)
    logger.info("Baseline cluster: %s", baseline_cluster)

    # ---- Crosswalk ----
    zip_codes_lf = household_days_lf.select(pl.col("zip_code")).unique()
    crosswalk_lf, crosswalk_metrics = load_crosswalk_one_to_one(args.crosswalk, zip_codes_lf=zip_codes_lf)
    household_days_bg_lf, join_metrics = attach_block_groups(household_days_lf, crosswalk_lf)

    # ---- Aggregate ----
    bg_comp = aggregate_blockgroup_composition(household_days_bg_lf, cluster_labels=cluster_labels)

    # ---- Census ----
    census_df = fetch_or_load_census(
        census_cache,
        state_fips=args.state_fips,
        acs_year=args.acs_year,
        force_fetch=args.fetch_census,
    )

    if "block_group_geoid" not in census_df.columns:
        if "GEOID" in census_df.columns:
            census_df = census_df.rename({"GEOID": "block_group_geoid"})
            logger.info("Renamed 'GEOID' to 'block_group_geoid' in Census data")
        else:
            raise ValueError(
                "Census data missing required column 'block_group_geoid' or 'GEOID'. "
                f"Available columns: {census_df.columns}",
            )

    demo_df = bg_comp.join(census_df, on="block_group_geoid", how="left")
    bg_total = int(demo_df["block_group_geoid"].n_unique())

    # ---- Predictors ----
    if args.predictors_from is not None:
        p_path = Path(args.predictors_from)
        predictors_detected_list = [ln.strip() for ln in p_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        excluded_all_null: list[str] = []
        predictors = predictors_detected_list
    else:
        predictors_detected_list, excluded_all_null = detect_predictors(demo_df)
        predictors = predictors_detected_list

    reg_df, predictors_used, filter_metrics = prepare_regression_df(
        demo_df,
        predictors,
        min_obs_per_bg=args.min_obs_per_bg,
    )
    bg_after_filters = int(reg_df["block_group_geoid"].n_unique())

    # ---- ALR outcomes ----
    reg_df, outcomes = add_smoothed_logratios(
        reg_df,
        baseline_cluster=baseline_cluster,
        alpha=args.alpha,
        k_total=k_total,
    )

    # ---- Write regression data ----
    reg_df_path = args.output_dir / "regression_data_blockgroups_wide.parquet"
    reg_df.write_parquet(reg_df_path)

    # ---- Fit models ----
    results = fit_models(
        reg_df,
        predictors=predictors_used,
        outcomes=outcomes,
        weight_col="total_obs",
        standardize=args.standardize,
        run_ols=not args.no_ols,
    )

    # ---- Required coefficient table ----
    coef_path = args.output_dir / "regression_results.parquet"
    write_coefficients_table_parquet(
        coef_path,
        wls_models=results["wls"],
        baseline_cluster=baseline_cluster,
        alpha=args.alpha,
        weight_col="total_obs",
        standardized=bool(args.standardize),
        param_names=results["param_names"],
    )

    # ---- statsmodels summaries ----
    (args.output_dir / "statsmodels_summaries_wls.txt").write_text(
        "\n\n".join([m.summary().as_text() for m in results["wls"].values()]),
        encoding="utf-8",
    )
    if not args.no_ols:
        (args.output_dir / "statsmodels_summaries_ols.txt").write_text(
            "\n\n".join([m.summary().as_text() for m in results["ols"].values()]),
            encoding="utf-8",
        )

    # ---- Diagnostics JSON (required) ----
    diagnostics: dict[str, Any] = {
        "k_total": int(k_total),
        "cluster_labels": cluster_labels,
        "baseline_cluster": str(baseline_cluster),
        "alpha": float(args.alpha),
        "min_obs_per_bg": int(args.min_obs_per_bg),
        "weights": {"weight_col": "total_obs", "definition": "total household-day observations per block group"},
        "n_block_groups_total": int(bg_total),
        "n_block_groups_after_filters": int(bg_after_filters),
        "dominance_stats": dominance_stats,
        "crosswalk_metrics": crosswalk_metrics,
        "crosswalk_resolution_rule": "min(block_group_geoid) per ZIP+4 when multiple block groups exist",
        "join_metrics": join_metrics,
        "filter_metrics": filter_metrics,
        "predictors_used": predictors_used,
        "predictors_excluded_all_null": excluded_all_null,
        "wls": {},
    }

    for out_name, m in results["wls"].items():
        diagnostics["wls"][out_name] = {
            "rsquared": float(getattr(m, "rsquared", np.nan)),
            "nobs": int(m.nobs),
            "residuals": _residual_diagnostics(m),
        }

    (args.output_dir / "regression_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # ---- Diagnostic plots (required) ----
    diag_plot_path = args.output_dir / "regression_diagnostics.png"
    write_diagnostic_plots(diag_plot_path, models=results["wls"])

    # ---- Interpretation guide (required) ----
    write_interpretation_guide(args.output_dir / "interpretation_guide.md", baseline_cluster=baseline_cluster)

    # ---- Stage2 metadata (required) ----
    stage2_meta = {
        "created_utc": _utc_now_iso(),
        "command": " ".join(sys.argv),
        "inputs": {
            "clusters_path": str(args.clusters),
            "crosswalk_path": str(args.crosswalk),
            "census_cache_path": str(census_cache),
        },
        "parameters": {
            "alpha": float(args.alpha),
            "baseline_cluster": str(baseline_cluster),
            "min_obs_per_bg": int(args.min_obs_per_bg),
            "standardize": bool(args.standardize),
            "run_ols": bool(not args.no_ols),
        },
        "sizes": {
            "k_total": int(k_total),
            "n_household_days_before_crosswalk": int(join_metrics["household_days_before_crosswalk"]),
            "n_household_days_after_crosswalk_nonnull": int(join_metrics["household_days_after_crosswalk_nonnull"]),
            "n_household_days_modeled": int(results["n_valid_rows"]),
            "n_block_groups_total": int(bg_total),
            "n_block_groups_after_filters": int(bg_after_filters),
        },
        "outputs": {
            "regression_results_parquet": str(coef_path),
            "regression_diagnostics_json": str(args.output_dir / "regression_diagnostics.json"),
            "regression_diagnostics_png": str(diag_plot_path),
            "interpretation_guide_md": str(args.output_dir / "interpretation_guide.md"),
            "regression_data_wide_parquet": str(reg_df_path),
        },
    }
    (args.output_dir / "stage2_metadata.json").write_text(
        json.dumps(stage2_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # ---- Human-readable report (keep) ----
    cluster_dist = (
        household_days_lf.group_by("cluster")
        .agg(pl.len().alias("n_obs"))
        .collect(streaming=True)  # type: ignore[call-overload]
        .sort("cluster")
        .with_columns((pl.col("n_obs") / pl.col("n_obs").sum() * 100).alias("pct"))
    )

    report_lines = ["STAGE 2 LOG-RATIO REGRESSION REPORT", ""]
    report_lines.append("Dominance stats (household-level):")
    for k, v in dominance_stats.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("")
    report_lines.append("Cluster distribution (household-day):")
    for row in cluster_dist.to_dicts():
        report_lines.append(f"  cluster={row['cluster']} n_obs={row['n_obs']} pct={row['pct']:.2f}")
    report_lines.append("")
    report_lines.append("Crosswalk join metrics:")
    for k, v in join_metrics.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("")

    for out_name, m in results["wls"].items():
        report_lines.append("=" * 80)
        report_lines.append(f"Outcome: {out_name} (WLS)")
        report_lines.append(m.summary().as_text())
        report_lines.append("")

    report_path = args.output_dir / "regression_report_logratio_blockgroups.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # ---- Manifest ----
    write_stage2_manifest(
        output_dir=args.output_dir,
        command=" ".join(sys.argv),
        repo_root=".",
        clusters_path=args.clusters,
        crosswalk_path=args.crosswalk,
        census_cache_path=census_cache,
        baseline_cluster=baseline_cluster,
        min_obs_per_bg=args.min_obs_per_bg,
        alpha=args.alpha,
        weight_column="total_obs",
        predictors_detected=len(predictors_detected_list),
        predictors_used=predictors_used,
        predictors_excluded_all_null=excluded_all_null,
        block_groups_total=int(bg_total),
        block_groups_after_min_obs=int(filter_metrics["rows_after_min_obs_filter"]),
        block_groups_after_drop_null_predictors=int(bg_after_filters),
        regression_data_path=reg_df_path,
        regression_report_path=report_path,
        run_log_path=args.output_dir / "run.log",
    )

    logger.info("Stage 2 complete: outputs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
