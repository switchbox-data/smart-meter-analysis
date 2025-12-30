#!/usr/bin/env python3
"""
Stage 2: Block-Group-Level Multinomial Logit Regression (HOUSEHOLD-DAY UNITS)

Design goals (boss-aligned)
---------------------------
1) Filter as far upstream as possible, for explicit stated reasons.
2) Fail cleanly but loudly: downstream errors raise with clear, actionable signals.

Pipeline summary
----------------
- Load household-day cluster assignments (lazy).
- Join ZIP+4 -> block group crosswalk; DROP rows with missing mapping (upstream, explicit).
- Aggregate to block-group counts and totals.
- Filter block groups by min_obs_per_bg BEFORE census join (upstream, explicit).
- Join census; enforce minimum join coverage threshold (explicit).
- Select predictors; compute missingness; enforce complete-case drop threshold (explicit).
- Fit joint multinomial logit on aggregated counts (no expansion).
  - Stable log-likelihood (no epsilon hacks)
  - Analytic observed information at MLE for SEs
  - Intercept-only multinomial null (overall outcome shares)
- Write regulatory deliverables + manifest.

Outputs (output_dir)
--------------------
- regression_results.parquet
- regression_diagnostics.json
- regression_diagnostics.png
- interpretation_guide.md
- stage2_metadata.json
- regression_data_blockgroups_wide.parquet
- regression_summaries.txt
- regression_report_multinomial_blockgroups.txt
- stage2_manifest.json (via write_stage2_manifest)
- run.log
- stage2_failure.json (only on failure; partial diagnostics)

Notes
-----
- Inference uses asymptotic normal z-statistics.
- BIC uses n_total (total household-days) to match multinomial likelihood scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import optimize
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler

from smart_meter_analysis.census import fetch_census_data
from smart_meter_analysis.census_specs import STAGE2_PREDICTORS_47
from smart_meter_analysis.run_manifest import write_stage2_manifest

logger = logging.getLogger(__name__)


# ----------------------------
# Utilities: logging + errors
# ----------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _configure_logging(output_dir: Path) -> Path:
    """Log to stdout and output_dir/run.log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers across repeated invocations
    existing = {(type(h), getattr(h, "baseFilename", None)) for h in root.handlers}
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if (logging.StreamHandler, None) not in existing:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if (logging.FileHandler, str(log_path)) not in existing:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return log_path


def _fail(msg: str, *, hint: str | None = None) -> None:
    """Raise a loud, user-facing error with consistent prefix."""
    full = f"STAGE2_ERROR: {msg}"
    if hint:
        full = f"{full}\nHINT: {hint}"
    raise RuntimeError(full)


def _write_failure_json(output_dir: Path, diagnostics_partial: dict[str, Any]) -> None:
    try:
        out = output_dir / "stage2_failure.json"
        out.write_text(json.dumps(diagnostics_partial, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as e:
        # If even failure reporting fails, there's not much we can do—avoid masking the root error.
        logger.debug("Failed to write failure JSON: %s", e, exc_info=True)


# ----------------------------
# IO + aggregation helpers
# ----------------------------


def _validate_clusters_schema(lf: pl.LazyFrame, path: Path) -> list[str]:
    schema = lf.collect_schema()
    required = ["account_identifier", "zip_code", "cluster"]
    missing = [c for c in required if c not in schema.names()]
    if missing:
        _fail(
            f"Missing required columns in {path}: {missing}",
            hint="Ensure Stage 1 produced cluster_assignments.parquet with account_identifier, zip_code, cluster.",
        )

    cluster_type = schema.get("cluster")
    if cluster_type not in (pl.Int32, pl.Int64, pl.Utf8, pl.String):
        _fail(f"cluster column must be integer or string, got {cluster_type}")

    return required


def _compute_dominance_stats_streaming(path: Path) -> dict[str, Any]:
    """Household dominance = share of a household's days assigned to its most common cluster."""
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
    dist = (
        household_days_lf.group_by("cluster")
        .agg(pl.len().alias("n_obs"))
        .sort("n_obs", descending=True)
        .collect(streaming=True)  # type: ignore[call-overload]
    )
    if dist.is_empty():
        _fail("No cluster assignments found; cannot choose baseline.")
    return str(dist["cluster"][0])


def list_cluster_labels(household_days_lf: pl.LazyFrame) -> list[str]:
    labels = (
        household_days_lf.select(pl.col("cluster").cast(pl.Utf8).alias("cluster"))
        .unique()
        .collect(streaming=True)  # type: ignore[call-overload]
        .get_column("cluster")
        .to_list()
    )
    out = sorted([str(x) for x in labels if x is not None])
    if len(out) < 2:
        _fail("Need at least 2 clusters for Stage 2.", hint="Verify Stage 1 clustering output.")
    return out


def load_crosswalk_one_to_one(
    crosswalk_path: Path,
    *,
    zip_codes_lf: pl.LazyFrame | None = None,
) -> tuple[pl.LazyFrame, dict[str, Any]]:
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

    # Upstream filter: only ZIP+4s that actually appear in the cluster parquet
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

    mapping = lf.group_by("zip_code").agg(pl.col("block_group_geoid").min().alias("block_group_geoid"))
    return mapping, metrics


def attach_block_groups(
    household_days_lf: pl.LazyFrame,
    crosswalk_lf: pl.LazyFrame,
) -> tuple[pl.LazyFrame, dict[str, Any]]:
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

    # Upstream filter: only keep rows with a valid BG
    return joined.drop_nulls("block_group_geoid"), metrics


def aggregate_blockgroup_composition(
    household_days_bg_lf: pl.LazyFrame,
    *,
    cluster_labels: Iterable[str],
) -> pl.DataFrame:
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

    # Ensure all cluster columns exist
    for lab in cluster_labels:
        if lab not in wide.columns:
            wide = wide.with_columns(pl.lit(0).alias(lab))

    cluster_cols_present = sorted([c for c in wide.columns if c != "block_group_geoid"])
    expected = sorted([str(x) for x in cluster_labels])
    if cluster_cols_present != expected:
        _fail(f"Cluster column mismatch: expected {expected}, got {cluster_cols_present}")

    # Rename to cluster_<label>
    cluster_cols = [c for c in wide.columns if c != "block_group_geoid"]
    wide = wide.rename({c: f"cluster_{c}" for c in cluster_cols})

    out = totals_and_hh.join(wide, on="block_group_geoid", how="left").fill_null(0)

    logger.info("Aggregated: %s block groups, %s cluster columns", f"{out.height:,}", len(cluster_cols))
    return out  # type: ignore[no-any-return]


# ----------------------------
# Census + predictor selection
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


def compute_missingness(df: pl.DataFrame, predictors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = max(int(df.height), 1)
    for p in predictors:
        if p not in df.columns:
            continue
        nulls = int(df.get_column(p).null_count())
        rows.append({"predictor": p, "null_count": nulls, "null_pct": float(nulls / n)})
    rows.sort(key=lambda r: r["null_pct"], reverse=True)
    return rows


def prepare_regression_df(
    df: pl.DataFrame,
    predictors: list[str],
    *,
    min_obs_per_bg: int,
) -> tuple[pl.DataFrame, list[str], dict[str, Any], list[dict[str, Any]]]:
    """
    Filtering logic (explicit, ordered):
      1) Drop BGs with total_obs < min_obs_per_bg (statistical stability)
      2) Select usable predictors (present and not all-null)
      3) Drop rows with any null across usable predictors (complete-case)
    """
    metrics: dict[str, Any] = {}

    n0 = int(df.height)

    # 1) Upstream filter: stable BGs
    df = df.filter(pl.col("total_obs") >= min_obs_per_bg)
    n1 = int(df.height)

    # 2) Usable predictors
    usable: list[str] = []
    for p in predictors:
        if p in df.columns and df.get_column(p).null_count() < df.height:
            usable.append(p)
    if not usable:
        _fail(
            "No usable predictors available after min_obs_per_bg filtering.",
            hint="Check census join coverage and STAGE2_PREDICTORS_47 definitions.",
        )

    missingness_pre = compute_missingness(df, usable)

    # 3) Complete-case
    df = df.filter(~pl.any_horizontal(pl.col(usable).is_null()))
    n2 = int(df.height)

    metrics["rows_before_filters"] = n0
    metrics["rows_after_min_obs_filter"] = n1
    metrics["rows_after_complete_case_filter"] = n2
    metrics["rows_dropped_min_obs"] = int(n0 - n1)
    metrics["rows_dropped_complete_case"] = int(n1 - n2)

    return df, usable, metrics, missingness_pre


# ----------------------------
# Multinomial logit core (stable LL, analytic Hessian)
# ----------------------------


def _loglike_pnon_residuals(
    beta_all: np.ndarray,
    X: np.ndarray,
    counts: np.ndarray,
    n_per_row: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Stable LL for baseline-parameterized multinomial.

    Baseline outcome has eta=0; non-baseline outcomes have eta = X @ beta^T.

    Returns:
      ll        : scalar log-likelihood
      p_non     : (n, J) non-baseline probabilities
      residuals : (n, J) counts_non - n_i * p_non
    """
    n, p = X.shape
    k_outcomes = counts.shape[1]
    J = k_outcomes - 1

    beta = beta_all.reshape(J, p)  # (J, p)
    eta = X @ beta.T  # (n, J)

    # denom_i = exp(0) + sum_j exp(eta_ij)
    # Use m_i = max(0, max_j eta_ij) for stability (includes baseline=0)
    row_max = np.max(eta, axis=1)  # (n,)
    m = np.maximum(0.0, row_max)  # (n,)

    exp0 = np.exp(-m)  # exp(0 - m)
    exp_eta = np.exp(eta - m[:, None])
    denom = exp0 + np.sum(exp_eta, axis=1)  # (n,)
    log_denom = m + np.log(denom)

    # log probs
    logp0 = -log_denom
    logp_non = eta - log_denom[:, None]

    ll = float(np.sum(counts[:, 0] * logp0) + np.sum(counts[:, 1:] * logp_non))

    p_non = exp_eta / denom[:, None]
    residuals = counts[:, 1:] - n_per_row[:, None] * p_non
    return ll, p_non, residuals


def negloglike_and_grad(
    beta_all: np.ndarray,
    X: np.ndarray,
    counts: np.ndarray,
    n_per_row: np.ndarray,
) -> tuple[float, np.ndarray]:
    ll, _p_non, residuals = _loglike_pnon_residuals(beta_all, X, counts, n_per_row)
    grad_ll = residuals.T @ X  # (J, p)
    return -ll, -grad_ll.reshape(-1)


def observed_information(
    X: np.ndarray,
    p_non: np.ndarray,
    n_per_row: np.ndarray,
) -> np.ndarray:
    """
    Observed information for the NEGATIVE log-likelihood.

    For outcomes a,b in {1..J}:
      H_ab = X^T diag(n_i * p_ia * (1[a==b] - p_ib)) X
    """
    n, p = X.shape
    J = p_non.shape[1]
    H = np.zeros((J * p, J * p), dtype=float)

    for a in range(J):
        for b in range(J):
            w = n_per_row * p_non[:, a] * (1.0 - p_non[:, a]) if a == b else -n_per_row * p_non[:, a] * p_non[:, b]
            Xw = X * w[:, None]
            Hab = X.T @ Xw

            ra, rb = a * p, (a + 1) * p
            ca, cb = b * p, (b + 1) * p
            H[ra:rb, ca:cb] = Hab

    return H


def null_loglikelihood_intercept_only(counts: np.ndarray) -> float:
    """
    Intercept-only multinomial null (overall shares):
      p_j = N_j / N
      LL0 = Σ_i Σ_j n_ij log(p_j)
    """
    N_j = counts.sum(axis=0).astype(np.float64)
    N = float(N_j.sum())
    if N <= 0:
        _fail("Total modeled household-days is zero after filtering.")
    p = N_j / N
    # Clip for numeric safety; categories with zero mass should be disallowed earlier.
    p = np.clip(p, 1e-300, 1.0)
    return float(np.sum(counts * np.log(p)))


class MultinomialLogitResults:
    """Joint multinomial logit results (non-baseline parameters only)."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        *,
        params_all: np.ndarray,  # (J*p,)
        cov_all: np.ndarray,  # (J*p, J*p)
        outcome_labels: list[str],  # non-baseline labels
        param_names: list[str],  # predictor names incl Intercept
        baseline_label: str,
        n_block_groups: int,
        n_total: int,
        loglike: float,
        loglike_null: float,
        standardized: bool,
        scaler: StandardScaler | None,
    ):
        self.params_all = params_all
        self.cov_all = cov_all
        self.outcome_labels = outcome_labels
        self.param_names = param_names
        self.baseline_label = baseline_label
        self.n_block_groups = int(n_block_groups)
        self.n_total = int(n_total)
        self.loglike = float(loglike)
        self.loglike_null = float(loglike_null)
        self.standardized = bool(standardized)
        self.scaler = scaler

        self.p = len(param_names)
        self.J = len(outcome_labels)
        self.k_outcomes = self.J + 1

        self.k_params = self.J * self.p
        self.k_null_params = self.J * 1
        self.df_llr = self.k_params - self.k_null_params  # (K-1)*(p-1)

        self.llr = 2.0 * (self.loglike - self.loglike_null)
        self.llr_pvalue = float(1.0 - scipy_stats.chi2.cdf(self.llr, self.df_llr))
        self.prsquared = float(1.0 - (self.loglike / self.loglike_null))

        self.aic = float(-2.0 * self.loglike + 2.0 * self.k_params)
        self.bic = float(-2.0 * self.loglike + self.k_params * np.log(max(self.n_total, 1)))

        diag = np.diag(self.cov_all)
        diag = np.where(diag < 0.0, np.nan, diag)
        self.bse_all = np.sqrt(diag)
        self.zvalues_all = self.params_all / self.bse_all
        self.pvalues_all = 2.0 * (1.0 - scipy_stats.norm.cdf(np.abs(self.zvalues_all)))

    def _slice(self, j: int) -> slice:
        return slice(j * self.p, (j + 1) * self.p)

    def outcome_table(self, outcome_label: str) -> pd.DataFrame:  # type: ignore[no-any-unimported]
        j = self.outcome_labels.index(outcome_label)
        sl = self._slice(j)
        coef = self.params_all[sl]
        se = self.bse_all[sl]
        z = self.zvalues_all[sl]
        pv = self.pvalues_all[sl]
        return pd.DataFrame(
            {
                "coef": coef,
                "std err": se,
                "z": z,
                "P>|z|": pv,
                "[0.025": coef - 1.96 * se,
                "0.975]": coef + 1.96 * se,
            },
            index=self.param_names,
        )

    def summary_text(self, outcome_label: str) -> str:
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append(f"Multinomial Logit: log-odds({outcome_label} vs {self.baseline_label})")
        lines.append("=" * 78)
        lines.append("Joint Model Diagnostics:")
        lines.append(f"  No. Block Groups:        {self.n_block_groups:>10,}")
        lines.append(f"  Total Household-Days:    {self.n_total:>10,}")
        lines.append(f"  Parameters (full):       {self.k_params:>10}")
        lines.append(f"  Parameters (null):       {self.k_null_params:>10}")
        lines.append(f"  Df (LLR):                {self.df_llr:>10}")
        lines.append(f"  Log-Likelihood:          {self.loglike:>10.2f}")
        lines.append(f"  LL-Null (intercept):     {self.loglike_null:>10.2f}")
        lines.append(f"  LLR:                     {self.llr:>10.2f}")
        lines.append(f"  Prob (LLR):              {self.llr_pvalue:>10.6f}")
        lines.append(f"  Pseudo R-squared:        {self.prsquared:>10.4f}")
        lines.append(f"  AIC:                     {self.aic:>10.2f}")
        lines.append(f"  BIC:                     {self.bic:>10.2f}")
        lines.append(f"  Standardized predictors: {self.standardized!s:>10}")
        lines.append("=" * 78)
        lines.append(self.outcome_table(outcome_label).to_string())
        lines.append("=" * 78)
        lines.append("Note: z-statistics and p-values use asymptotic normal approximation.")
        return "\n".join(lines)

    def coefficients_long(self) -> pd.DataFrame:  # type: ignore[no-any-unimported]
        rows: list[dict[str, Any]] = []
        for j, out_lab in enumerate(self.outcome_labels):
            sl = self._slice(j)
            for name, coef, se, z, pv in zip(
                self.param_names,
                self.params_all[sl],
                self.bse_all[sl],
                self.zvalues_all[sl],
                self.pvalues_all[sl],
            ):
                rows.append({
                    "cluster": out_lab,
                    "predictor": str(name),
                    "coefficient": float(coef),
                    "std_error": float(se) if np.isfinite(se) else float("nan"),
                    # Back-compat: some downstream expects t_stat; provide both
                    "t_stat": float(z) if np.isfinite(z) else float("nan"),
                    "z_stat": float(z) if np.isfinite(z) else float("nan"),
                    "p_value": float(pv) if np.isfinite(pv) else float("nan"),
                })
        return pd.DataFrame(rows)

    def scaler_info(self, predictors_native: list[str]) -> dict[str, Any] | None:
        if self.scaler is None:
            return None
        means = {p: float(m) for p, m in zip(predictors_native, self.scaler.mean_)}
        scales = {p: float(s) for p, s in zip(predictors_native, self.scaler.scale_)}
        return {"means": means, "scales": scales}


def fit_multinomial_logit(
    df: pl.DataFrame,
    *,
    cluster_labels: list[str],
    baseline_cluster: str,
    predictors: list[str],
    standardize: bool,
    allow_nonconvergence: bool,
    allow_singular_hessian: bool,
    max_hessian_condition_number: float,
) -> tuple[MultinomialLogitResults, dict[str, Any]]:
    """
    Fit joint multinomial logit on aggregated counts.

    Returns:
      results, fit_meta
    """
    df = df.sort("block_group_geoid")

    # Counts in canonical order of cluster_labels
    cluster_cols = [f"cluster_{lab}" for lab in cluster_labels]
    counts = df.select(cluster_cols).to_numpy().astype(np.float64)
    n_per_row = counts.sum(axis=1).astype(np.float64)
    n_total = int(n_per_row.sum())
    n_block_groups = int(df.height)

    # Predictors
    X_raw = df.select(predictors).to_numpy().astype(np.float64)
    scaler: StandardScaler | None = None  # type: ignore[no-any-unimported]
    if standardize:
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(X_raw)

    X_df = pd.DataFrame(X_raw, columns=predictors)
    X_df.insert(0, "Intercept", 1.0)
    X = X_df.values.astype(np.float64)
    param_names = list(X_df.columns)

    k_outcomes = len(cluster_labels)
    if baseline_cluster not in cluster_labels:
        _fail(
            f"Baseline cluster '{baseline_cluster}' not found in cluster labels: {cluster_labels}",
            hint="Use --baseline-cluster only with a label present in the cluster assignments.",
        )
    if k_outcomes < 2:
        _fail("Need at least 2 outcomes for multinomial logit.")

    # Reorder so baseline is first
    baseline_idx = cluster_labels.index(baseline_cluster)
    if baseline_idx != 0:
        counts = np.column_stack([counts[:, baseline_idx], counts[:, :baseline_idx], counts[:, baseline_idx + 1 :]])
        reordered_labels = [baseline_cluster, *cluster_labels[:baseline_idx], *cluster_labels[baseline_idx + 1 :]]
    else:
        reordered_labels = cluster_labels[:]

    non_baseline_labels = reordered_labels[1:]
    J = k_outcomes - 1
    p = X.shape[1]

    # Null LL (intercept-only multinomial)
    ll_null = null_loglikelihood_intercept_only(counts)

    # Initialize small random params
    rng = np.random.default_rng(42)
    beta_init = rng.normal(loc=0.0, scale=0.01, size=(J * p,)).astype(np.float64)

    logger.info(
        "Fitting multinomial logit: K=%d, p=%d (incl intercept), n_bg=%d, n_total=%d",
        k_outcomes,
        p,
        n_block_groups,
        n_total,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        opt = optimize.minimize(
            fun=lambda b: negloglike_and_grad(b, X, counts, n_per_row),
            x0=beta_init,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "disp": False},
        )

    if not opt.success:
        msg = f"Optimization did not converge: {opt.message}"
        if allow_nonconvergence:
            logger.warning(msg)
        else:
            _fail(
                msg,
                hint="Try --standardize, reduce collinearity, or pass --allow-nonconvergence if you must proceed.",
            )

    beta_mle = opt.x.astype(np.float64)
    ll_full, p_non, _resid = _loglike_pnon_residuals(beta_mle, X, counts, n_per_row)

    # Observed information
    logger.info("Computing analytic observed information at MLE...")
    H = observed_information(X, p_non, n_per_row)

    # Condition number gate (prevents silently unstable inference)
    cond = float(np.linalg.cond(H))
    if not np.isfinite(cond):
        msg = "Observed information matrix condition number is non-finite."
        if allow_singular_hessian:
            logger.warning("%s Proceeding due to --allow-singular-hessian.", msg)
        else:
            _fail(msg, hint="This indicates severe multicollinearity or separation. Consider reducing predictors.")

    if np.isfinite(cond) and cond > max_hessian_condition_number:
        msg = f"Observed information matrix is ill-conditioned (cond={cond:.3e} > {max_hessian_condition_number:.3e})."
        if allow_singular_hessian:
            logger.warning("%s Proceeding due to --allow-singular-hessian.", msg)
        else:
            _fail(
                msg,
                hint="Reduce collinearity (drop correlated predictors), increase min_obs_per_bg, or use --allow-singular-hessian.",
            )

    # Invert for covariance
    try:
        cov = np.linalg.inv(H)
        used_pinv = False
    except np.linalg.LinAlgError:
        if allow_singular_hessian:
            logger.warning("Observed information is singular; using pseudo-inverse due to --allow-singular-hessian.")
            cov = np.linalg.pinv(H)
            used_pinv = True
        else:
            _fail(
                "Observed information matrix is singular; cannot compute standard errors.",
                hint="Reduce predictors or pass --allow-singular-hessian to proceed with pseudo-inverse.",
            )

    results = MultinomialLogitResults(
        params_all=beta_mle,
        cov_all=cov,
        outcome_labels=non_baseline_labels,
        param_names=param_names,
        baseline_label=baseline_cluster,
        n_block_groups=n_block_groups,
        n_total=n_total,
        loglike=ll_full,
        loglike_null=ll_null,
        standardized=standardize,
        scaler=scaler,
    )

    fit_meta: dict[str, Any] = {
        "converged": bool(opt.success),
        "optimizer_message": str(opt.message),
        "n_iters": int(getattr(opt, "nit", -1)),
        "n_total": int(n_total),
        "n_block_groups": int(n_block_groups),
        "k_outcomes": int(k_outcomes),
        "p": int(p),
        "k_params": int(results.k_params),
        "k_null_params": int(results.k_null_params),
        "df_llr": int(results.df_llr),
        "loglike": float(results.loglike),
        "loglike_null": float(results.loglike_null),
        "llr": float(results.llr),
        "llr_pvalue": float(results.llr_pvalue),
        "prsquared": float(results.prsquared),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "standardized": bool(standardize),
        "hessian_condition_number": cond,
        "used_pinv_for_covariance": bool(used_pinv),
    }
    scaler_info = results.scaler_info(predictors)
    if scaler_info is not None:
        fit_meta["scaler_info"] = scaler_info

    return results, fit_meta


# ----------------------------
# Outputs
# ----------------------------


def write_coefficients_table_parquet(
    out_path: Path,
    *,
    results: MultinomialLogitResults,
    baseline_cluster: str,
    weight_col: str,
) -> None:
    df_long = results.coefficients_long()
    df_long["pseudo_r_squared"] = results.prsquared
    df_long["nobs"] = results.n_block_groups
    df_long["n_total_household_days"] = results.n_total
    df_long["baseline_cluster"] = str(baseline_cluster)
    df_long["weight_col"] = str(weight_col)
    df_long["standardized"] = bool(results.standardized)
    df_long["loglike"] = float(results.loglike)
    df_long["loglike_null"] = float(results.loglike_null)
    df_long["aic"] = float(results.aic)
    df_long["bic"] = float(results.bic)
    df_long["llr"] = float(results.llr)
    df_long["llr_pvalue"] = float(results.llr_pvalue)
    df_long["df_llr"] = int(results.df_llr)

    pl.from_pandas(df_long).write_parquet(out_path)
    logger.info("Wrote coefficient table: %s", out_path)


def write_diagnostic_plots(out_path: Path, *, results: MultinomialLogitResults) -> None:
    n_out = len(results.outcome_labels)
    fig, axes = plt.subplots(n_out, 2, figsize=(12, max(4, 3.5 * n_out)))
    if n_out == 1:
        axes = np.array([axes])

    for i, out_lab in enumerate(results.outcome_labels):
        ax0, ax1 = axes[i, 0], axes[i, 1]
        tbl = results.outcome_table(out_lab)
        tbl_no_intercept = tbl.drop(index=["Intercept"], errors="ignore")

        y = np.arange(len(tbl_no_intercept))
        coef = tbl_no_intercept["coef"].values
        se = tbl_no_intercept["std err"].values

        ax0.barh(y, coef)
        ax0.errorbar(coef, y, xerr=1.96 * se, fmt="none", color="black", alpha=0.5)
        ax0.set_yticks(y)
        ax0.set_yticklabels(tbl_no_intercept.index, fontsize=8)
        ax0.axvline(0, linestyle="--", linewidth=1)
        ax0.set_title(f"log-odds({out_lab} vs {results.baseline_label}) coefficients ± 95% CI")
        ax0.set_xlabel("Coefficient")

        pvals = tbl_no_intercept["P>|z|"].values
        pvals = pvals[np.isfinite(pvals)]
        ax1.hist(pvals, bins=20, edgecolor="black", alpha=0.7)
        ax1.axvline(0.05, linestyle="--", label="a=0.05")
        ax1.set_title(f"log-odds({out_lab} vs {results.baseline_label}) p-values")
        ax1.set_xlabel("p-value")
        ax1.set_ylabel("Frequency")
        ax1.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote diagnostic plots: %s", out_path)


def write_interpretation_guide(out_path: Path, *, baseline_cluster: str, standardized: bool) -> None:
    unit_note = (
        "Predictors were standardized (z-scored). A one-unit increase means a one standard deviation increase."
        if standardized
        else "Predictors were not standardized. A one-unit increase uses the predictor's native units."
    )
    md = f"""# How to Read Stage 2 Coefficients (Multinomial Logit Regression)

## Model
For each non-baseline cluster q:

log(π_q / π_{{baseline}}) = β0_q + β_q · X

This is multinomial logistic regression fitted on aggregated block-group counts.

Baseline cluster: **{baseline_cluster}**

## Interpretation
For a one-unit increase in predictor X_k:

- The log-odds changes by β_k
- The odds ratio multiplier is exp(β_k)

## Standardization
{unit_note}

## Statistical Method
- Multinomial likelihood on aggregated block-group counts (no expansion to household-day rows)
- No Laplace smoothing (zeros handled naturally by likelihood)
- Standard errors from analytic observed information at the MLE
- Inference uses asymptotic normal (z) statistics
"""
    out_path.write_text(md + "\n", encoding="utf-8")
    logger.info("Wrote interpretation guide: %s", out_path)


# ----------------------------
# CLI
# ----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: Block-group-level multinomial logit regression using household-day units.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--clusters", type=Path, required=True, help="cluster_assignments.parquet")
    parser.add_argument("--crosswalk", type=Path, required=True, help="ZIP+4 → block-group crosswalk")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for Stage 2 artifacts")

    parser.add_argument("--census-cache", type=Path, default=None, help="Per-run Census cache path")
    parser.add_argument("--fetch-census", action="store_true", help="Force re-fetch Census data")
    parser.add_argument("--state-fips", default="17")
    parser.add_argument("--acs-year", type=int, default=2023)

    # Upstream filters
    parser.add_argument("--min-obs-per-bg", type=int, default=50, help="Minimum household-days per block group")
    parser.add_argument(
        "--min-block-groups-for-model",
        type=int,
        default=100,
        help="Fail if fewer than this many block groups remain after filtering",
    )

    # Fail-fast gates (explicit thresholds)
    parser.add_argument(
        "--max-drop-missing-crosswalk-pct",
        type=float,
        default=0.05,
        help="Fail if pct of household-days dropped due to missing crosswalk exceeds this (default 0.05)",
    )
    parser.add_argument(
        "--min-census-match-pct",
        type=float,
        default=0.98,
        help="Fail if pct of modeled BGs with a census match falls below this (default 0.98)",
    )
    parser.add_argument(
        "--max-drop-complete-case-pct",
        type=float,
        default=0.20,
        help="Fail if pct of BGs dropped by complete-case filtering exceeds this (default 0.20)",
    )

    # Model options / safety overrides
    parser.add_argument("--standardize", action="store_true", help="Standardize predictors before regression")
    parser.add_argument("--baseline-cluster", type=str, default=None, help="Baseline cluster label")
    parser.add_argument("--predictors-from", type=str, default=None, help="Path to predictor list file (one per line)")
    parser.add_argument(
        "--allow-nonconvergence", action="store_true", help="Proceed even if optimizer does not converge"
    )
    parser.add_argument(
        "--allow-singular-hessian",
        action="store_true",
        help="Proceed if observed information is singular/ill-conditioned (uses pseudo-inverse)",
    )
    parser.add_argument(
        "--max-hessian-cond",
        type=float,
        default=1e12,
        help="Fail if Hessian condition number exceeds this unless --allow-singular-hessian is set",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = _configure_logging(args.output_dir)
    logger.info("Command: %s", " ".join(sys.argv))

    diagnostics_partial: dict[str, Any] = {
        "status": "started",
        "created_utc": _utc_now_iso(),
        "command": " ".join(sys.argv),
        "parameters": {
            "min_obs_per_bg": args.min_obs_per_bg,
            "min_block_groups_for_model": args.min_block_groups_for_model,
            "max_drop_missing_crosswalk_pct": args.max_drop_missing_crosswalk_pct,
            "min_census_match_pct": args.min_census_match_pct,
            "max_drop_complete_case_pct": args.max_drop_complete_case_pct,
            "standardize": bool(args.standardize),
            "allow_nonconvergence": bool(args.allow_nonconvergence),
            "allow_singular_hessian": bool(args.allow_singular_hessian),
            "max_hessian_cond": float(args.max_hessian_cond),
        },
        "paths": {
            "clusters": str(args.clusters),
            "crosswalk": str(args.crosswalk),
            "output_dir": str(args.output_dir),
            "run_log": str(run_log_path),
        },
    }

    try:
        if not args.clusters.exists():
            _fail(f"Cluster assignments not found: {args.clusters}")
        if not args.crosswalk.exists():
            _fail(f"Crosswalk not found: {args.crosswalk}")

        census_cache = args.census_cache
        if census_cache is None:
            census_cache = args.output_dir / f"census_cache_{args.state_fips}_{args.acs_year}.parquet"

        # ---- Load clusters ----
        household_days_lf, _stats, dominance_stats = load_household_day_clusters(args.clusters)
        cluster_labels = list_cluster_labels(household_days_lf)
        diagnostics_partial["cluster_labels"] = cluster_labels

        baseline_cluster = args.baseline_cluster or choose_baseline_cluster(household_days_lf)
        if baseline_cluster not in cluster_labels:
            _fail(
                f"Baseline cluster '{baseline_cluster}' not in detected labels {cluster_labels}",
                hint="If you specified --baseline-cluster, ensure it matches Stage 1 labels.",
            )
        logger.info("Baseline cluster: %s", baseline_cluster)

        # ---- Crosswalk ----
        zip_codes_lf = household_days_lf.select(pl.col("zip_code")).unique()
        crosswalk_lf, crosswalk_metrics = load_crosswalk_one_to_one(args.crosswalk, zip_codes_lf=zip_codes_lf)
        household_days_bg_lf, join_metrics = attach_block_groups(household_days_lf, crosswalk_lf)

        diagnostics_partial["crosswalk_metrics"] = crosswalk_metrics
        diagnostics_partial["join_metrics"] = join_metrics

        # Fail-fast: too much crosswalk drop
        drop_pct = float(join_metrics["pct_dropped_missing_crosswalk"])
        if drop_pct > float(args.max_drop_missing_crosswalk_pct):
            _fail(
                f"Crosswalk drop too high: {drop_pct:.2%} household-days missing BG mapping "
                f"(threshold {args.max_drop_missing_crosswalk_pct:.2%}).",
                hint="Verify crosswalk version/year matches data and ZIP+4 formatting matches Stage 1 outputs.",
            )

        # ---- Aggregate (BG) ----
        bg_comp = aggregate_blockgroup_composition(household_days_bg_lf, cluster_labels=cluster_labels)

        # Upstream BG filter (BEFORE census join)
        bg0 = int(bg_comp.height)
        bg_comp = bg_comp.filter(pl.col("total_obs") >= int(args.min_obs_per_bg))
        bg1 = int(bg_comp.height)

        if bg1 < int(args.min_block_groups_for_model):
            _fail(
                f"Too few block groups after min_obs_per_bg filter: {bg1} (min required {args.min_block_groups_for_model}).",
                hint="Lower --min-obs-per-bg or verify the upstream join/aggregation did not drop too much data.",
            )

        diagnostics_partial["bg_counts"] = {
            "block_groups_before_min_obs_filter": bg0,
            "block_groups_after_min_obs_filter": bg1,
            "rows_dropped_min_obs_filter": bg0 - bg1,
        }
        logger.info("BG filter (min_obs_per_bg=%d): %d -> %d BGs", args.min_obs_per_bg, bg0, bg1)

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
            else:
                _fail("Census data missing 'block_group_geoid' or 'GEOID'")

        # Add explicit join indicator to measure census coverage precisely
        census_df = census_df.with_columns(pl.lit(1).alias("_census_joined"))

        demo_df = bg_comp.join(census_df, on="block_group_geoid", how="left")
        bg_total_modeled = int(demo_df.height)
        bg_census_matched = int(demo_df.select(pl.col("_census_joined").is_not_null().sum()).item())
        census_match_pct = float(bg_census_matched / max(bg_total_modeled, 1))

        diagnostics_partial["census_join"] = {
            "bg_rows_modeled_pre_complete_case": bg_total_modeled,
            "bg_rows_with_census_match": bg_census_matched,
            "census_match_pct": census_match_pct,
            "census_cache_path": str(census_cache),
        }

        if census_match_pct < float(args.min_census_match_pct):
            _fail(
                f"Census join coverage too low: {census_match_pct:.2%} (threshold {args.min_census_match_pct:.2%}).",
                hint="Confirm census GEOIDs match the crosswalk BG GEOID format (12-digit block group GEOID).",
            )

        # ---- Predictors ----
        if args.predictors_from is not None:
            p_path = Path(args.predictors_from)
            if not p_path.exists():
                _fail(f"--predictors-from file not found: {p_path}")
            predictors = [ln.strip() for ln in p_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            excluded_all_null: list[str] = []
        else:
            predictors, excluded_all_null = detect_predictors(demo_df)

        # ---- Final regression filtering (complete-case) ----
        reg_df, predictors_used, filter_metrics, missingness_pre = prepare_regression_df(
            demo_df,
            predictors,
            min_obs_per_bg=int(args.min_obs_per_bg),
        )

        # Remove join indicator from regression df before writing/export (but keep for diagnostics)
        if "_census_joined" in reg_df.columns:
            reg_df = reg_df.drop("_census_joined")

        bg_after_complete_case = int(reg_df.height)
        drop_complete_case = int(filter_metrics["rows_dropped_complete_case"])
        denom_cc = max(int(filter_metrics["rows_after_min_obs_filter"]), 1)
        drop_complete_case_pct = float(drop_complete_case / denom_cc)

        diagnostics_partial["filter_metrics"] = filter_metrics
        diagnostics_partial["predictors_used"] = predictors_used
        diagnostics_partial["predictors_excluded_all_null"] = excluded_all_null
        diagnostics_partial["missingness_pre_complete_case"] = missingness_pre[:25]  # cap size

        # Fail-fast: too much complete-case drop
        if drop_complete_case_pct > float(args.max_drop_complete_case_pct):
            top = missingness_pre[:10]
            top_str = "; ".join([f"{r['predictor']}={r['null_pct']:.1%}" for r in top])
            _fail(
                f"Complete-case filtering dropped too many BGs: {drop_complete_case_pct:.2%} "
                f"(threshold {args.max_drop_complete_case_pct:.2%}). Top missing predictors: {top_str}",
                hint="Consider adjusting predictors, improving census join coverage, or increasing allowable missingness via workflow changes.",
            )

        if bg_after_complete_case < int(args.min_block_groups_for_model):
            _fail(
                f"Too few block groups after complete-case filtering: {bg_after_complete_case} "
                f"(min required {args.min_block_groups_for_model}).",
                hint="Reduce predictor set, lower min_obs_per_bg, or address missing census fields.",
            )

        # Fail-fast: ensure each outcome has positive mass after filtering
        outcome_totals = {}
        for lab in cluster_labels:
            col = f"cluster_{lab}"
            if col not in reg_df.columns:
                _fail(f"Missing expected cluster count column in regression df: {col}")
            outcome_totals[lab] = int(reg_df.select(pl.col(col).sum()).item())
        zero_outcomes = [k for k, v in outcome_totals.items() if v <= 0]
        if zero_outcomes:
            _fail(
                f"One or more clusters have zero total counts after filtering: {zero_outcomes}",
                hint="This can happen if filtering removes all BGs that contain certain clusters. Lower min_obs_per_bg or adjust filters.",
            )

        diagnostics_partial["outcome_totals_after_filters"] = outcome_totals

        # ---- Write regression data ----
        reg_df_path = args.output_dir / "regression_data_blockgroups_wide.parquet"
        reg_df.write_parquet(reg_df_path)
        logger.info("Wrote regression data: %s", reg_df_path)

        # ---- Fit ----
        results, fit_meta = fit_multinomial_logit(
            reg_df,
            cluster_labels=cluster_labels,
            baseline_cluster=baseline_cluster,
            predictors=predictors_used,
            standardize=bool(args.standardize),
            allow_nonconvergence=bool(args.allow_nonconvergence),
            allow_singular_hessian=bool(args.allow_singular_hessian),
            max_hessian_condition_number=float(args.max_hessian_cond),
        )
        diagnostics_partial["fit"] = fit_meta

        # ---- Outputs ----
        coef_path = args.output_dir / "regression_results.parquet"
        write_coefficients_table_parquet(
            coef_path,
            results=results,
            baseline_cluster=baseline_cluster,
            weight_col="total_obs",
        )

        (args.output_dir / "regression_summaries.txt").write_text(
            "\n\n".join([results.summary_text(out_lab) for out_lab in results.outcome_labels]) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote model summaries")

        diag_plot_path = args.output_dir / "regression_diagnostics.png"
        write_diagnostic_plots(diag_plot_path, results=results)

        write_interpretation_guide(
            args.output_dir / "interpretation_guide.md",
            baseline_cluster=baseline_cluster,
            standardized=bool(args.standardize),
        )

        # ---- Diagnostics JSON ----
        diagnostics: dict[str, Any] = {
            "status": "ok",
            "method": "multinomial_logit_aggregated",
            "created_utc": _utc_now_iso(),
            "command": " ".join(sys.argv),
            "cluster_labels": cluster_labels,
            "baseline_cluster": str(baseline_cluster),
            "dominance_stats": dominance_stats,
            "crosswalk_metrics": crosswalk_metrics,
            "join_metrics": join_metrics,
            "bg_counts": diagnostics_partial.get("bg_counts", {}),
            "census_join": diagnostics_partial.get("census_join", {}),
            "predictors_used": predictors_used,
            "predictors_excluded_all_null": excluded_all_null,
            "missingness_pre_complete_case": missingness_pre[:25],
            "filter_metrics": filter_metrics,
            "complete_case_drop_pct": drop_complete_case_pct,
            "outcome_totals_after_filters": outcome_totals,
            "fit": fit_meta,
            "thresholds": {
                "max_drop_missing_crosswalk_pct": float(args.max_drop_missing_crosswalk_pct),
                "min_census_match_pct": float(args.min_census_match_pct),
                "max_drop_complete_case_pct": float(args.max_drop_complete_case_pct),
                "min_block_groups_for_model": int(args.min_block_groups_for_model),
            },
        }
        (args.output_dir / "regression_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # ---- Metadata JSON ----
        stage2_meta: dict[str, Any] = {
            "created_utc": _utc_now_iso(),
            "command": " ".join(sys.argv),
            "method": "multinomial_logit_aggregated",
            "inputs": {
                "clusters_path": str(args.clusters),
                "crosswalk_path": str(args.crosswalk),
                "census_cache_path": str(census_cache),
            },
            "parameters": {
                "baseline_cluster": str(baseline_cluster),
                "min_obs_per_bg": int(args.min_obs_per_bg),
                "standardize": bool(args.standardize),
                "thresholds": diagnostics["thresholds"],
            },
            "sizes": {
                "k_total": len(cluster_labels),
                "n_household_days_before_crosswalk": int(join_metrics["household_days_before_crosswalk"]),
                "n_household_days_after_crosswalk_nonnull": int(join_metrics["household_days_after_crosswalk_nonnull"]),
                "n_household_days_modeled": int(results.n_total),
                "n_block_groups_after_min_obs": int(
                    diagnostics_partial["bg_counts"]["block_groups_after_min_obs_filter"]
                ),
                "n_block_groups_after_complete_case": int(bg_after_complete_case),
            },
            "outputs": {
                "regression_results_parquet": str(coef_path),
                "regression_diagnostics_json": str(args.output_dir / "regression_diagnostics.json"),
                "regression_diagnostics_png": str(diag_plot_path),
                "interpretation_guide_md": str(args.output_dir / "interpretation_guide.md"),
                "regression_data_wide_parquet": str(reg_df_path),
                "run_log": str(run_log_path),
            },
        }
        scaler_info = fit_meta.get("scaler_info")
        if scaler_info is not None:
            stage2_meta["fit_scaler_info"] = scaler_info

        (args.output_dir / "stage2_metadata.json").write_text(
            json.dumps(stage2_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # ---- Report ----
        cluster_dist = (
            household_days_lf.group_by("cluster")
            .agg(pl.len().alias("n_obs"))
            .collect(streaming=True)  # type: ignore[call-overload]
            .sort("cluster")
            .with_columns((pl.col("n_obs") / pl.col("n_obs").sum() * 100).alias("pct"))
        )

        report_lines: list[str] = []
        report_lines.append("STAGE 2 MULTINOMIAL LOGIT REGRESSION REPORT")
        report_lines.append("")
        report_lines.append("Method: Multinomial logit on aggregated block-group counts")
        report_lines.append("")
        report_lines.append("Dominance stats (household-level):")
        for k, v in dominance_stats.items():
            report_lines.append(f"  {k}: {v}")
        report_lines.append("")
        report_lines.append("Cluster distribution (household-day):")
        for row in cluster_dist.to_dicts():
            report_lines.append(f"  cluster={row['cluster']} n_obs={row['n_obs']} pct={row['pct']:.2f}")
        report_lines.append("")
        report_lines.append("Joint fit statistics:")
        report_lines.append(f"  loglike_full: {results.loglike:.2f}")
        report_lines.append(f"  loglike_null: {results.loglike_null:.2f}")
        report_lines.append(f"  pseudo_r2:    {results.prsquared:.4f}")
        report_lines.append(f"  aic:          {results.aic:.2f}")
        report_lines.append(f"  bic:          {results.bic:.2f}")
        report_lines.append(f"  llr:          {results.llr:.2f}")
        report_lines.append(f"  llr_pvalue:   {results.llr_pvalue:.6f}")
        report_lines.append(f"  df_llr:       {results.df_llr}")
        report_lines.append("")

        for out_lab in results.outcome_labels:
            report_lines.append("=" * 80)
            report_lines.append(results.summary_text(out_lab))
            report_lines.append("")

        report_path = args.output_dir / "regression_report_multinomial_blockgroups.txt"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        logger.info("Wrote report: %s", report_path)

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
            alpha=0.0,  # Not used in multinomial logit
            weight_column="total_obs",
            predictors_detected=len(predictors),
            predictors_used=predictors_used,
            predictors_excluded_all_null=excluded_all_null,
            block_groups_total=int(diagnostics_partial["bg_counts"]["block_groups_before_min_obs_filter"]),
            block_groups_after_min_obs=int(diagnostics_partial["bg_counts"]["block_groups_after_min_obs_filter"]),
            block_groups_after_drop_null_predictors=int(bg_after_complete_case),
            regression_data_path=reg_df_path,
            regression_report_path=report_path,
            run_log_path=run_log_path,
        )

        diagnostics_partial["status"] = "ok"
        logger.info("Stage 2 multinomial logit complete: outputs in %s", args.output_dir)
        return 0

    except Exception as e:
        logger.exception("Stage 2 failed with error.")
        diagnostics_partial["status"] = "failed"
        diagnostics_partial["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }
        _write_failure_json(args.output_dir, diagnostics_partial)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
