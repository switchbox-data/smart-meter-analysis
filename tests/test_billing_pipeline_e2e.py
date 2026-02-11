#!/usr/bin/env python3
"""End-to-end tests for the multi-month billing pipeline orchestrator.

Runs the full pipeline on existing sample artifacts (no synthetic data
generation) and validates:

1. Orchestrator produces expected directory layout
2. Per-month outputs exist and have non-empty row counts
3. Annual aggregate exists and equals sum of monthly totals per household
4. Join coverage: no null prices, no row inflation
5. Required columns exist in outputs
6. Regression outputs exist and contain both models + predictor list + R-sq
7. Manifest contains git SHA, parameters, month-by-month counts

All outputs go to tmp_path; no writes into tracked data directories.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

# ── Sample data paths (committed reference data) ────────────────────────
INTERVAL_DATA = Path("data/processed/comed_202308.parquet")
TARIFF_A = Path("data/reference/comed_flat_hourly_prices_2023.parquet")
TARIFF_B = Path("data/reference/comed_stou_hourly_prices_2023.parquet")
CROSSWALK = Path("data/reference/comed_bg_zip4_crosswalk.txt")
CENSUS = Path("data/reference/census_17_2023.parquet")

# Skip entire module if sample data is missing
pytestmark = pytest.mark.skipif(
    not INTERVAL_DATA.exists() or not TARIFF_A.exists() or not TARIFF_B.exists(),
    reason="Sample data files not found; skipping E2E pipeline tests.",
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the orchestrator once for the module; return paths + metadata."""
    out_dir = tmp_path_factory.mktemp("billing_e2e")
    run_name = "test_e2e"
    run_dir = out_dir / run_name

    cmd = [
        sys.executable,
        "scripts/run_billing_pipeline.py",
        "--months",
        "202308",
        "--interval-pattern",
        str(INTERVAL_DATA.parent / "comed_{yyyymm}.parquet"),
        "--tariff-a",
        str(TARIFF_A),
        "--tariff-b",
        str(TARIFF_B),
        "--crosswalk",
        str(CROSSWALK),
        "--census",
        str(CENSUS),
        "--predictors",
        "median_household_income",
        "--min-obs-per-bg",
        "1",
        "--max-crosswalk-drop-pct",
        "100",
        "--run-name",
        run_name,
        "--output-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    return {
        "run_dir": run_dir,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "month": "202308",
    }


@pytest.fixture(scope="module")
def pipeline_skip_regression(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run orchestrator with --skip-regression for faster validation."""
    out_dir = tmp_path_factory.mktemp("billing_skip_reg")
    run_name = "test_skip_reg"
    run_dir = out_dir / run_name

    cmd = [
        sys.executable,
        "scripts/run_billing_pipeline.py",
        "--months",
        "202308",
        "--interval-pattern",
        str(INTERVAL_DATA.parent / "comed_{yyyymm}.parquet"),
        "--tariff-a",
        str(TARIFF_A),
        "--tariff-b",
        str(TARIFF_B),
        "--skip-regression",
        "--run-name",
        run_name,
        "--output-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    return {
        "run_dir": run_dir,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Directory layout
# ═══════════════════════════════════════════════════════════════════════════


class TestDirectoryLayout:
    """Verify that the orchestrator creates the expected directory tree."""

    def test_pipeline_succeeds(self, pipeline_skip_regression: dict) -> None:
        r = pipeline_skip_regression
        assert r["returncode"] == 0, (
            f"Pipeline exited {r['returncode']}.\nstdout: {r['stdout'][-2000:]}\nstderr: {r['stderr'][-2000:]}"
        )

    def test_run_dir_exists(self, pipeline_skip_regression: dict) -> None:
        assert pipeline_skip_regression["run_dir"].is_dir()

    def test_tmp_dir_exists(self, pipeline_skip_regression: dict) -> None:
        tmp = pipeline_skip_regression["run_dir"] / "_tmp"
        assert tmp.is_dir(), "Missing _tmp directory for hourly loads"

    def test_hourly_loads_exists(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "_tmp" / "month=202308" / "hourly_loads.parquet"
        assert path.exists(), f"Missing hourly loads: {path}"

    def test_monthly_bills_exist(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "month=202308" / "household_bills.parquet"
        assert path.exists(), f"Missing monthly bills: {path}"

    def test_annual_aggregate_exists(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "annual_household_aggregate.parquet"
        assert path.exists(), f"Missing annual aggregate: {path}"

    def test_manifest_exists(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        assert path.exists(), f"Missing manifest: {path}"

    def test_log_exists(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "pipeline.log"
        assert path.exists(), f"Missing pipeline log: {path}"

    def test_regression_dir_absent_when_skipped(self, pipeline_skip_regression: dict) -> None:
        reg = pipeline_skip_regression["run_dir"] / "regression"
        # Regression dir should not be created when --skip-regression
        assert not reg.exists() or not any(reg.iterdir()), (
            "Regression artifacts should not exist with --skip-regression"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-month outputs: non-empty row counts
# ═══════════════════════════════════════════════════════════════════════════


class TestPerMonthOutputs:
    """Verify per-month outputs have non-empty row counts."""

    def test_hourly_loads_not_empty(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "_tmp" / "month=202308" / "hourly_loads.parquet"
        df = pl.read_parquet(path)
        assert df.height > 0, "Hourly loads parquet is empty"

    def test_monthly_bills_not_empty(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "month=202308" / "household_bills.parquet"
        df = pl.read_parquet(path)
        assert df.height > 0, "Monthly bills parquet is empty"

    def test_hourly_loads_required_columns(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "_tmp" / "month=202308" / "hourly_loads.parquet"
        df = pl.read_parquet(path)
        required = {"account_identifier", "zip_code", "hour_chicago", "kwh_hour"}
        missing = required - set(df.columns)
        assert not missing, f"Hourly loads missing columns: {missing}"

    def test_monthly_bills_required_columns(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "month=202308" / "household_bills.parquet"
        df = pl.read_parquet(path)
        required = {
            "account_identifier",
            "zip_code",
            "total_kwh",
            "bill_a_dollars",
            "bill_b_dollars",
            "bill_diff_dollars",
            "pct_savings",
            "net_bill_diff_dollars",
            "net_pct_savings",
        }
        missing = required - set(df.columns)
        assert not missing, f"Monthly bills missing columns: {missing}"

    def test_no_null_prices_in_bills(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "month=202308" / "household_bills.parquet"
        df = pl.read_parquet(path)
        for col in ("bill_a_dollars", "bill_b_dollars"):
            n_null = df.select(pl.col(col).is_null().sum()).item()
            assert n_null == 0, f"Column {col} has {n_null} null values (join coverage failure)"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Annual aggregate correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestAnnualAggregate:
    """Verify annual aggregate equals sum of monthly totals per household."""

    def test_annual_not_empty(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "annual_household_aggregate.parquet"
        df = pl.read_parquet(path)
        assert df.height > 0, "Annual aggregate is empty"

    def test_annual_has_required_columns(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "annual_household_aggregate.parquet"
        df = pl.read_parquet(path)
        required = {
            "account_identifier",
            "total_kwh",
            "bill_a_dollars",
            "bill_b_dollars",
            "bill_diff_dollars",
            "pct_savings",
            "net_pct_savings",
        }
        missing = required - set(df.columns)
        assert not missing, f"Annual aggregate missing columns: {missing}"

    def test_annual_matches_monthly_totals(self, pipeline_skip_regression: dict) -> None:
        """For single-month run, annual should equal monthly exactly."""
        run_dir = pipeline_skip_regression["run_dir"]
        monthly = pl.read_parquet(run_dir / "month=202308" / "household_bills.parquet")
        annual = pl.read_parquet(run_dir / "annual_household_aggregate.parquet")

        # For a single-month run, the sums should match
        sum_cols = ["total_kwh", "bill_a_dollars", "bill_b_dollars", "bill_diff_dollars"]
        for col in sum_cols:
            monthly_total = monthly[col].sum()
            annual_total = annual[col].sum()
            assert abs(monthly_total - annual_total) < 0.01, (
                f"Column {col}: monthly sum={monthly_total:.4f} != annual sum={annual_total:.4f}"
            )

    def test_annual_household_count_matches(self, pipeline_skip_regression: dict) -> None:
        """Annual should have same number of unique households as monthly."""
        run_dir = pipeline_skip_regression["run_dir"]
        monthly = pl.read_parquet(run_dir / "month=202308" / "household_bills.parquet")
        annual = pl.read_parquet(run_dir / "annual_household_aggregate.parquet")

        assert monthly["account_identifier"].n_unique() == annual.height, (
            f"Household count mismatch: monthly unique={monthly['account_identifier'].n_unique()}, "
            f"annual rows={annual.height}"
        )

    def test_pct_savings_recomputed(self, pipeline_skip_regression: dict) -> None:
        """pct_savings should equal bill_diff / bill_a * 100."""
        path = pipeline_skip_regression["run_dir"] / "annual_household_aggregate.parquet"
        df = pl.read_parquet(path)
        # Filter to rows with positive bill_a to avoid division issues
        df_valid = df.filter(pl.col("bill_a_dollars") > 0)
        expected = df_valid["bill_diff_dollars"] / df_valid["bill_a_dollars"] * 100
        actual = df_valid["pct_savings"]
        diff = (expected - actual).abs().max()
        assert diff < 0.001, f"pct_savings recomputation mismatch: max diff = {diff}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Join coverage: no row inflation
# ═══════════════════════════════════════════════════════════════════════════


class TestJoinCoverage:
    """Verify joins don't inflate or drop rows unexpectedly."""

    def test_no_row_inflation_in_bills(self, pipeline_skip_regression: dict) -> None:
        """Each account should appear exactly once in monthly bills."""
        path = pipeline_skip_regression["run_dir"] / "month=202308" / "household_bills.parquet"
        df = pl.read_parquet(path)
        dupes = df.group_by("account_identifier").len().filter(pl.col("len") > 1)
        assert dupes.height == 0, f"Duplicate accounts in monthly bills: {dupes.height}"

    def test_loads_accounts_preserved_in_bills(self, pipeline_skip_regression: dict) -> None:
        """All accounts in loads should appear in bills (no silent drops)."""
        run_dir = pipeline_skip_regression["run_dir"]
        loads = pl.read_parquet(run_dir / "_tmp" / "month=202308" / "hourly_loads.parquet")
        bills = pl.read_parquet(run_dir / "month=202308" / "household_bills.parquet")

        loads_accounts = set(loads["account_identifier"].unique().to_list())
        bills_accounts = set(bills["account_identifier"].unique().to_list())
        dropped = loads_accounts - bills_accounts
        assert not dropped, f"Accounts in loads but not in bills: {len(dropped)}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Regression outputs (with relaxed thresholds for small sample)
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionOutputs:
    """Verify regression outputs when pipeline runs with --predictors core."""

    def test_full_pipeline_succeeds(self, pipeline_run: dict) -> None:
        r = pipeline_run
        assert r["returncode"] == 0, (
            f"Full pipeline exited {r['returncode']}.\nstdout: {r['stdout'][-2000:]}\nstderr: {r['stderr'][-2000:]}"
        )

    def test_regression_dir_exists(self, pipeline_run: dict) -> None:
        reg_dir = pipeline_run["run_dir"] / "regression"
        assert reg_dir.is_dir(), f"Regression directory missing: {reg_dir}"

    def test_regression_dataset_exists(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_dataset_bg.parquet"
        assert path.exists(), f"Missing regression dataset: {path}"

    def test_regression_results_json(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_results.json"
        assert path.exists(), f"Missing regression results: {path}"
        with open(path) as f:
            results = json.load(f)

        # Should have two models
        assert "model_1_savings" in results, "Missing model_1_savings in regression results"
        assert "model_2_bill_diff" in results, "Missing model_2_bill_diff in regression results"

        # Each model should have R-squared
        for model_key in ("model_1_savings", "model_2_bill_diff"):
            m = results[model_key]
            assert "r_squared" in m, f"{model_key} missing r_squared"
            assert "adj_r_squared" in m, f"{model_key} missing adj_r_squared"
            assert "f_statistic" in m, f"{model_key} missing f_statistic"
            assert "coefficients" in m, f"{model_key} missing coefficients"
            assert "n_obs" in m, f"{model_key} missing n_obs"
            assert m["n_obs"] > 0, f"{model_key} has 0 observations"

    def test_regression_summary_text(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_summary.txt"
        assert path.exists(), f"Missing regression summary: {path}"
        text = path.read_text()
        assert len(text) > 100, "Regression summary text suspiciously short"
        assert "R-squared" in text, "Regression summary should contain R-squared"

    def test_regression_metadata_json(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        assert path.exists(), f"Missing regression metadata: {path}"
        with open(path) as f:
            meta = json.load(f)

        assert "predictors_used" in meta, "Missing predictors_used in metadata"
        assert len(meta["predictors_used"]) > 0, "Empty predictor list"
        assert "join_metrics" in meta, "Missing join_metrics in metadata"
        assert "crosswalk_metrics" in meta, "Missing crosswalk_metrics in metadata"

    def test_regression_predictors_match_explicit(self, pipeline_run: dict) -> None:
        """--predictors median_household_income should use exactly that predictor."""
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        assert meta["predictors_used"] == ["median_household_income"]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Manifest correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestManifest:
    """Verify the run manifest JSON has complete provenance."""

    def test_manifest_required_fields(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)

        for key in ("run_id", "created_utc", "months", "month_summary", "inputs", "parameters", "steps_completed"):
            assert key in m, f"Manifest missing key: {key}"

    def test_manifest_git_sha(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)

        sha = m.get("git_sha")
        assert sha is not None, "Manifest missing git_sha"
        assert len(sha) == 40, f"git_sha should be 40 hex chars, got: {sha!r}"

    def test_manifest_months(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)

        assert m["months"] == ["202308"], f"Expected months=['202308'], got {m['months']}"

    def test_manifest_month_summary_counts(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)

        summary = m["month_summary"]
        assert "202308" in summary, "Month 202308 not in month_summary"
        counts = summary["202308"]
        assert counts["rows_hourly_loads"] > 0, "rows_hourly_loads should be > 0"
        assert counts["rows_bills"] > 0, "rows_bills should be > 0"

    def test_manifest_steps_completed(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)

        steps = m["steps_completed"]
        assert "202308" in steps, "Month 202308 not in steps_completed"
        assert "annual_aggregate" in steps, "annual_aggregate not in steps_completed"

    def test_manifest_parameters(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)

        params = m["parameters"]
        assert "capacity_rate" in params, "Missing capacity_rate in parameters"
        assert "admin_fee" in params, "Missing admin_fee in parameters"
        assert params["skip_regression"] is True, "skip_regression should be True"

    def test_manifest_inputs(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)

        inputs = m["inputs"]
        assert "interval_pattern" in inputs, "Missing interval_pattern in inputs"
        assert "tariff_a" in inputs, "Missing tariff_a in inputs"
        assert "tariff_b" in inputs, "Missing tariff_b in inputs"
