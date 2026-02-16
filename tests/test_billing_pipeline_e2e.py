#!/usr/bin/env python3
"""End-to-end tests for the multi-month billing pipeline orchestrator.

Runs the full pipeline on existing sample artifacts (no synthetic data
generation) and validates:

1. Orchestrator produces expected directory layout
2. Per-month outputs exist and have non-empty row counts
3. All-months concatenated bills have correct schema and month tagging
4. Join coverage: no null prices, no row inflation
5. Required columns exist in all outputs
6. Regression outputs exist with correct schemas and mathematical invariants
7. Manifest contains git SHA, parameters, month-by-month counts, steps
8. Skip-regression mode produces no regression artifacts
9. Both regression modes (annual, bg_month) produce consistent outputs

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
_RAW_INTERVAL_DATA = Path("data/processed/comed_202308.parquet")
TARIFF_A = Path("data/reference/comed_flat_hourly_prices_2023.parquet")
TARIFF_B = Path("data/reference/comed_stou_hourly_prices_2023.parquet")
CROSSWALK = Path("data/reference/comed_bg_zip4_crosswalk.txt")
CENSUS = Path("data/reference/census_17_2023.parquet")

# Skip entire module if sample data is missing. Module-level skip keeps
# test collection clean in CI environments without the full data checkout.
pytestmark = pytest.mark.skipif(
    not _RAW_INTERVAL_DATA.exists() or not TARIFF_A.exists() or not TARIFF_B.exists(),
    reason="Sample data files not found; skipping E2E pipeline tests.",
)

# Regression model keys produced by build_regression_dataset.py
MODEL_KEY_1 = "model_1_pct_savings_weighted"
MODEL_KEY_2 = "model_2_sum_bill_diff"

# Regression parquet artifacts (under regression/)
REGRESSION_PARQUETS = [
    "bg_month_outcomes.parquet",
    "bg_annual_outcomes.parquet",
    "bg_season_outcomes.parquet",
    "regression_dataset_bg.parquet",
]
REGRESSION_JSONS = [
    "regression_results.json",
    "regression_metadata.json",
]
REGRESSION_TEXTS = [
    "regression_summary.txt",
]
ALL_REGRESSION_FILES = REGRESSION_PARQUETS + REGRESSION_JSONS + REGRESSION_TEXTS


# ── Helper to prepare test data ─────────────────────────────────────────
# The refactored pipeline expects the canonical column name ``energy_kwh``
# but the committed sample parquet uses the legacy name ``kwh``.  Rather
# than modifying the core pipeline or the committed data, we create a
# lightweight temp copy with the column renamed once per test session.
#
# The committed sample data maps to only 2 Census block groups—too few for
# OLS regression (needs n_obs >= n_params + 2).  We synthesize a handful
# of extra households at diverse ZIP+4 values so that the regression step
# can run with ≥ 6 BGs.  The synthetic rows reuse real hourly timestamps
# so that tariff joins succeed without gaps.

# ZIP+4 values that map to distinct block groups with non-null census
# predictors, chosen from the crosswalk.  We only need a few extra BGs
# beyond the 2 already in the committed test data.
_SYNTHETIC_ZIP4S = [
    "60068-5766",
    "60430-1950",
    "60946-0128",
    "62401-4817",
    "62650-6533",
    "61048-0270",
]


def _prepare_interval_data(tmp_dir: Path) -> Path:
    """Copy interval parquet to *tmp_dir* with fixes for the current pipeline.

    1. Rename ``kwh`` → ``energy_kwh`` if needed (canonical column name).
    2. Append synthetic households at diverse ZIP+4s so the crosswalk join
       yields enough block groups for OLS regression (≥ 6 BGs).

    Returns the **pattern** path suitable for ``--interval-pattern``.
    """
    dest = tmp_dir / "comed_202308.parquet"
    if dest.exists():
        return tmp_dir / "comed_{yyyymm}.parquet"

    df = pl.read_parquet(_RAW_INTERVAL_DATA)
    if "energy_kwh" not in df.columns and "kwh" in df.columns:
        df = df.rename({"kwh": "energy_kwh"})

    # Take one real household's hourly timestamps as a template so
    # tariff joins match.  Each synthetic household gets identical
    # timestamps but a different zip_code and account_identifier.
    template_acct = df["account_identifier"][0]
    template = df.filter(pl.col("account_identifier") == template_acct)

    synth_frames = []
    for i, zip4 in enumerate(_SYNTHETIC_ZIP4S):
        synth = template.with_columns(
            pl.lit(f"SYNTH_{i:04d}").alias("account_identifier"),
            pl.lit(zip4).alias("zip_code"),
        )
        synth_frames.append(synth)

    augmented = pl.concat([df, *synth_frames], how="diagonal_relaxed")
    augmented.write_parquet(dest)
    return tmp_dir / "comed_{yyyymm}.parquet"


# ── Helper to build pipeline command ────────────────────────────────────


def _build_pipeline_cmd(
    *,
    months: str,
    interval_pattern: Path,
    out_dir: Path,
    run_name: str,
    skip_regression: bool = False,
    regression_level: str = "annual",
    include_crosswalk: bool = True,
) -> list[str]:
    """Build a CLI command list for the billing pipeline orchestrator."""
    cmd = [
        sys.executable,
        "scripts/run_billing_pipeline.py",
        "--months",
        months,
        "--interval-pattern",
        str(interval_pattern),
        "--tariff-a",
        str(TARIFF_A),
        "--tariff-b",
        str(TARIFF_B),
        "--run-name",
        run_name,
        "--output-dir",
        str(out_dir),
    ]
    if skip_regression:
        cmd.append("--skip-regression")
    else:
        # Regression requires crosswalk + census
        if include_crosswalk:
            cmd.extend(["--crosswalk", str(CROSSWALK)])
            cmd.extend(["--census", str(CENSUS)])
        # Relaxed thresholds for small test fixture
        cmd.extend(["--predictors", "median_household_income"])
        cmd.extend(["--min-obs-per-bg", "1"])
        cmd.extend(["--max-crosswalk-drop-pct", "100"])
        cmd.extend(["--regression-level", regression_level])
    return cmd


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def interval_pattern(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepare interval data with canonical column names (session-shared)."""
    data_dir = tmp_path_factory.mktemp("interval_data")
    return _prepare_interval_data(data_dir)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory, interval_pattern: Path) -> dict:
    """Run the orchestrator with regression (annual, default level)."""
    out_dir = tmp_path_factory.mktemp("billing_e2e")
    run_name = "test_e2e"
    cmd = _build_pipeline_cmd(
        months="202308",
        interval_pattern=interval_pattern,
        out_dir=out_dir,
        run_name=run_name,
        regression_level="annual",
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return {
        "run_dir": out_dir / run_name,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "month": "202308",
    }


@pytest.fixture(scope="module")
def pipeline_run_bg_month(tmp_path_factory: pytest.TempPathFactory, interval_pattern: Path) -> dict:
    """Run the orchestrator with --regression-level bg_month."""
    out_dir = tmp_path_factory.mktemp("billing_bgmonth")
    run_name = "test_bgmonth"
    cmd = _build_pipeline_cmd(
        months="202308",
        interval_pattern=interval_pattern,
        out_dir=out_dir,
        run_name=run_name,
        regression_level="bg_month",
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return {
        "run_dir": out_dir / run_name,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "month": "202308",
    }


@pytest.fixture(scope="module")
def pipeline_skip_regression(tmp_path_factory: pytest.TempPathFactory, interval_pattern: Path) -> dict:
    """Run orchestrator with --skip-regression for faster validation."""
    out_dir = tmp_path_factory.mktemp("billing_skip_reg")
    run_name = "test_skip_reg"
    cmd = _build_pipeline_cmd(
        months="202308",
        interval_pattern=interval_pattern,
        out_dir=out_dir,
        run_name=run_name,
        skip_regression=True,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return {
        "run_dir": out_dir / run_name,
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

    def test_all_months_bills_exists(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        assert path.exists(), f"Missing all-months bills: {path}"

    def test_manifest_exists(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        assert path.exists(), f"Missing manifest: {path}"

    def test_log_exists(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "pipeline.log"
        assert path.exists(), f"Missing pipeline log: {path}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-month outputs: non-empty row counts & required columns
# ═══════════════════════════════════════════════════════════════════════════


class TestPerMonthOutputs:
    """Verify per-month outputs have non-empty row counts and correct schema."""

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
# 3. All-months household bills (replaces annual aggregate)
# ═══════════════════════════════════════════════════════════════════════════


class TestAllMonthsBills:
    """Validate the concatenated all_months_household_bills.parquet artifact."""

    def test_not_empty(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        df = pl.read_parquet(path)
        assert df.height > 0, "All-months bills is empty"

    def test_readable_by_polars(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        df = pl.read_parquet(path)
        assert isinstance(df, pl.DataFrame)

    def test_required_columns(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        df = pl.read_parquet(path)
        required = {
            "account_identifier",
            "zip_code",
            "month",
            "total_kwh",
            "bill_a_dollars",
            "bill_diff_dollars",
        }
        missing = required - set(df.columns)
        assert not missing, f"All-months bills missing columns: {missing}"

    def test_month_column_format(self, pipeline_skip_regression: dict) -> None:
        """month must be a 6-digit YYYYMM string for all rows."""
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        df = pl.read_parquet(path)
        # Check dtype is string
        assert df.schema["month"] == pl.Utf8, f"month dtype should be Utf8, got {df.schema['month']}"
        # Every value must be exactly 6 digits
        bad = df.filter(~pl.col("month").str.contains(r"^\d{6}$"))
        assert bad.height == 0, (
            f"{bad.height} rows have non-YYYYMM month values: {bad['month'].unique().to_list()[:10]}"
        )

    def test_month_column_no_nulls(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        df = pl.read_parquet(path)
        n_null = df.select(pl.col("month").is_null().sum()).item()
        assert n_null == 0, f"month column has {n_null} nulls"

    def test_month_values_match_requested(self, pipeline_skip_regression: dict) -> None:
        """Set of months in output should exactly match requested months."""
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        df = pl.read_parquet(path)
        actual_months = set(df["month"].unique().to_list())
        expected_months = {"202308"}
        assert actual_months == expected_months, (
            f"Month set mismatch: actual={actual_months}, expected={expected_months}"
        )

    def test_row_count_matches_monthly(self, pipeline_skip_regression: dict) -> None:
        """For single-month run, all-months rows should equal monthly rows."""
        run_dir = pipeline_skip_regression["run_dir"]
        monthly = pl.read_parquet(run_dir / "month=202308" / "household_bills.parquet")
        all_months = pl.read_parquet(run_dir / "all_months_household_bills.parquet")
        assert all_months.height == monthly.height, (
            f"Row count mismatch: all_months={all_months.height}, monthly={monthly.height}"
        )

    def test_household_count_matches(self, pipeline_skip_regression: dict) -> None:
        """Unique households should match between monthly and all-months."""
        run_dir = pipeline_skip_regression["run_dir"]
        monthly = pl.read_parquet(run_dir / "month=202308" / "household_bills.parquet")
        all_months = pl.read_parquet(run_dir / "all_months_household_bills.parquet")
        assert monthly["account_identifier"].n_unique() == all_months["account_identifier"].n_unique()

    def test_additive_totals_match_monthly(self, pipeline_skip_regression: dict) -> None:
        """For single-month run, sum of additive columns should match exactly."""
        run_dir = pipeline_skip_regression["run_dir"]
        monthly = pl.read_parquet(run_dir / "month=202308" / "household_bills.parquet")
        all_months = pl.read_parquet(run_dir / "all_months_household_bills.parquet")
        sum_cols = ["total_kwh", "bill_a_dollars", "bill_b_dollars", "bill_diff_dollars"]
        for col in sum_cols:
            monthly_total = monthly[col].sum()
            all_months_total = all_months[col].sum()
            assert abs(monthly_total - all_months_total) < 0.01, (
                f"Column {col}: monthly sum={monthly_total:.4f} != all_months sum={all_months_total:.4f}"
            )


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
# 5. Regression artifacts: existence and schemas
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionArtifacts:
    """Verify all regression outputs exist and have correct schemas."""

    def test_full_pipeline_succeeds(self, pipeline_run: dict) -> None:
        r = pipeline_run
        assert r["returncode"] == 0, (
            f"Full pipeline exited {r['returncode']}.\nstdout: {r['stdout'][-2000:]}\nstderr: {r['stderr'][-2000:]}"
        )

    def test_regression_dir_exists(self, pipeline_run: dict) -> None:
        reg_dir = pipeline_run["run_dir"] / "regression"
        assert reg_dir.is_dir(), f"Regression directory missing: {reg_dir}"

    @pytest.mark.parametrize("filename", ALL_REGRESSION_FILES)
    def test_regression_file_exists(self, pipeline_run: dict, filename: str) -> None:
        path = pipeline_run["run_dir"] / "regression" / filename
        assert path.exists(), f"Missing regression artifact: {path}"

    @pytest.mark.parametrize("filename", REGRESSION_PARQUETS)
    def test_regression_parquet_readable(self, pipeline_run: dict, filename: str) -> None:
        path = pipeline_run["run_dir"] / "regression" / filename
        df = pl.read_parquet(path)
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0, f"{filename} is empty"

    def test_bg_month_outcomes_schema(self, pipeline_run: dict) -> None:
        df = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_month_outcomes.parquet")
        required = {
            "block_group_geoid",
            "month",
            "sum_total_kwh",
            "sum_bill_a_dollars",
            "sum_bill_diff_dollars",
            "pct_savings_weighted",
            "n_household_months",
        }
        missing = required - set(df.columns)
        assert not missing, f"bg_month_outcomes missing columns: {missing}"

    def test_bg_annual_outcomes_schema(self, pipeline_run: dict) -> None:
        df = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_annual_outcomes.parquet")
        required = {
            "block_group_geoid",
            "sum_total_kwh",
            "sum_bill_a_dollars",
            "sum_bill_diff_dollars",
            "pct_savings_weighted",
            "n_household_months",
        }
        missing = required - set(df.columns)
        assert not missing, f"bg_annual_outcomes missing columns: {missing}"

    def test_bg_season_outcomes_schema(self, pipeline_run: dict) -> None:
        df = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_season_outcomes.parquet")
        required = {
            "block_group_geoid",
            "season",
            "sum_total_kwh",
            "sum_bill_a_dollars",
            "sum_bill_diff_dollars",
            "pct_savings_weighted",
            "n_household_months",
        }
        missing = required - set(df.columns)
        assert not missing, f"bg_season_outcomes missing columns: {missing}"

    def test_bg_annual_n_household_months_definition(self, pipeline_run: dict) -> None:
        """n_household_months is household-months, not unique households."""
        bg_month = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_month_outcomes.parquet")
        bg_annual = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_annual_outcomes.parquet")
        # For single-month run, the values should match (no multi-month aggregation)
        expected = bg_month.group_by("block_group_geoid").agg(pl.col("n_household_months").sum())
        check = expected.join(
            bg_annual.select("block_group_geoid", "n_household_months"),
            on="block_group_geoid",
            suffix="_annual",
        )
        diff = (check["n_household_months"] - check["n_household_months_annual"]).abs().max()
        assert diff == 0, f"n_household_months mismatch between month and annual: max diff={diff}"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Mathematical invariants (strong correctness checks)
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionMathInvariants:
    """Verify mathematical relationships across regression outputs."""

    def test_pct_savings_weighted_definition_annual(self, pipeline_run: dict) -> None:
        """pct_savings_weighted ≈ sum_diff / sum_bill_a * 100 on annual table."""
        reg_dir = pipeline_run["run_dir"] / "regression"
        # Read metadata to know which diff column the pipeline resolved
        with open(reg_dir / "regression_metadata.json") as f:
            meta = json.load(f)
        diff_col = meta["bill_diff_column_used"]
        sum_diff_col = f"sum_{diff_col}"

        df = pl.read_parquet(reg_dir / "bg_annual_outcomes.parquet")
        valid = df.filter(pl.col("sum_bill_a_dollars") > 0)
        assert valid.height > 0, "No rows with positive sum_bill_a_dollars"

        expected = valid[sum_diff_col] / valid["sum_bill_a_dollars"] * 100
        actual = valid["pct_savings_weighted"]
        max_diff = (expected - actual).abs().max()
        assert max_diff < 1e-6, f"pct_savings_weighted invariant violated on annual table: max diff={max_diff}"

    def test_pct_savings_weighted_null_when_zero_denom(self, pipeline_run: dict) -> None:
        """When sum_bill_a_dollars <= 0, pct_savings_weighted should be null."""
        df = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_annual_outcomes.parquet")
        zero_denom = df.filter(pl.col("sum_bill_a_dollars") <= 0)
        if zero_denom.height > 0:
            non_null = zero_denom.filter(pl.col("pct_savings_weighted").is_not_null())
            assert non_null.height == 0, (
                f"{non_null.height} rows have pct_savings_weighted despite zero/negative bill_a"
            )

    def test_annual_rollup_equals_sum_of_months(self, pipeline_run: dict) -> None:
        """Annual additive sums should equal sum over monthly BG outcomes."""
        reg_dir = pipeline_run["run_dir"] / "regression"
        bg_month = pl.read_parquet(reg_dir / "bg_month_outcomes.parquet")
        bg_annual = pl.read_parquet(reg_dir / "bg_annual_outcomes.parquet")

        additive_cols = [c for c in bg_month.columns if c.startswith("sum_")]
        expected = bg_month.group_by("block_group_geoid").agg(
            [pl.col(c).sum() for c in additive_cols] + [pl.col("n_household_months").sum()]
        )

        check = expected.join(bg_annual, on="block_group_geoid", suffix="_annual")
        for col in additive_cols:
            diff = (check[col] - check[f"{col}_annual"]).abs().max()
            assert diff < 1e-6, f"Annual rollup mismatch for {col}: max diff={diff}"
        nhm_diff = (check["n_household_months"] - check["n_household_months_annual"]).abs().max()
        assert nhm_diff == 0, f"n_household_months mismatch: max diff={nhm_diff}"

    def test_annual_pct_recomputed_from_sums(self, pipeline_run: dict) -> None:
        """Annual pct_savings_weighted should be recomputed from rolled-up sums."""
        reg_dir = pipeline_run["run_dir"] / "regression"
        with open(reg_dir / "regression_metadata.json") as f:
            meta = json.load(f)
        diff_col = meta["bill_diff_column_used"]
        sum_diff_col = f"sum_{diff_col}"

        bg_month = pl.read_parquet(reg_dir / "bg_month_outcomes.parquet")
        bg_annual = pl.read_parquet(reg_dir / "bg_annual_outcomes.parquet")

        # Compute expected annual pct from rolled-up month sums
        rolled = bg_month.group_by("block_group_geoid").agg(
            pl.col(sum_diff_col).sum(),
            pl.col("sum_bill_a_dollars").sum(),
        )
        rolled = rolled.with_columns(
            pl.when(pl.col("sum_bill_a_dollars") > 0)
            .then(pl.col(sum_diff_col) / pl.col("sum_bill_a_dollars") * 100)
            .otherwise(None)
            .alias("expected_pct"),
        )
        check = rolled.join(
            bg_annual.select("block_group_geoid", "pct_savings_weighted"),
            on="block_group_geoid",
        )
        valid = check.filter(pl.col("expected_pct").is_not_null())
        max_diff = (valid["expected_pct"] - valid["pct_savings_weighted"]).abs().max()
        assert max_diff < 1e-6, f"Annual pct not recomputed from sums: max diff={max_diff}"

    def test_season_values_valid(self, pipeline_run: dict) -> None:
        """Seasons must be from the canonical set."""
        df = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_season_outcomes.parquet")
        valid_seasons = {"Winter", "Spring", "Summer", "Fall"}
        actual_seasons = set(df["season"].unique().to_list())
        assert actual_seasons <= valid_seasons, f"Invalid seasons: {actual_seasons - valid_seasons}"

    def test_season_mapping_consistent_for_sample(self, pipeline_run: dict) -> None:
        """August (202308) should map to Summer."""
        df = pl.read_parquet(pipeline_run["run_dir"] / "regression" / "bg_season_outcomes.parquet")
        # With single month 202308, only Summer should appear
        actual = set(df["season"].unique().to_list())
        assert actual == {"Summer"}, f"Expected only Summer for month 202308, got {actual}"

    def test_season_rollup_matches_months(self, pipeline_run: dict) -> None:
        """Season additive sums should equal sum of constituent months."""
        reg_dir = pipeline_run["run_dir"] / "regression"
        bg_month = pl.read_parquet(reg_dir / "bg_month_outcomes.parquet")
        bg_season = pl.read_parquet(reg_dir / "bg_season_outcomes.parquet")

        # Derive season from month (mirrors _derive_season_expr logic)
        mm = bg_month["month"].str.slice(4, 2)
        bg_month_with_season = bg_month.with_columns(
            pl.when(mm.is_in(["12", "01", "02"]))
            .then(pl.lit("Winter"))
            .when(mm.is_in(["03", "04", "05"]))
            .then(pl.lit("Spring"))
            .when(mm.is_in(["06", "07", "08"]))
            .then(pl.lit("Summer"))
            .when(mm.is_in(["09", "10", "11"]))
            .then(pl.lit("Fall"))
            .otherwise(pl.lit(None))
            .alias("season")
        )

        additive_cols = [c for c in bg_month.columns if c.startswith("sum_")]
        expected = bg_month_with_season.group_by(["block_group_geoid", "season"]).agg(
            [pl.col(c).sum() for c in additive_cols] + [pl.col("n_household_months").sum()]
        )

        check = expected.join(bg_season, on=["block_group_geoid", "season"], suffix="_season")
        for col in additive_cols:
            diff = (check[col] - check[f"{col}_season"]).abs().max()
            assert diff < 1e-6, f"Season rollup mismatch for {col}: max diff={diff}"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Crosswalk coverage & regression metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionCrosswalkAndMetadata:
    """Validate crosswalk coverage and regression metadata provenance."""

    def test_crosswalk_metrics_in_metadata(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        cw = meta["crosswalk_metrics"]
        assert "n_zip4" in cw, "Missing n_zip4 in crosswalk_metrics"
        assert "n_bg" in cw, "Missing n_bg in crosswalk_metrics"
        assert "n_zip4_multi_bg" in cw, "Missing n_zip4_multi_bg in crosswalk_metrics"
        assert cw["n_zip4"] > 0, "n_zip4 should be positive"
        assert cw["n_bg"] > 0, "n_bg should be positive"

    def test_join_metrics_in_metadata(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        jm = meta["join_metrics"]
        assert "pct_dropped" in jm, "Missing pct_dropped in join_metrics"
        assert "households_before_crosswalk" in jm
        assert "households_matched" in jm
        assert "households_dropped" in jm

    def test_pct_dropped_within_threshold(self, pipeline_run: dict) -> None:
        """pct_dropped should be <= max_crosswalk_drop_pct and not NaN."""
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        pct = meta["join_metrics"]["pct_dropped"]
        max_pct = meta["max_crosswalk_drop_pct"]
        assert isinstance(pct, (int, float)), f"pct_dropped is not numeric: {pct!r}"
        assert pct == pct, "pct_dropped is NaN"  # NaN != NaN
        assert pct <= max_pct, f"pct_dropped={pct:.2f}% exceeds max_crosswalk_drop_pct={max_pct}%"

    def test_predictors_match_explicit(self, pipeline_run: dict) -> None:
        """--predictors median_household_income should use exactly that predictor."""
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        assert meta["predictors_used"] == ["median_household_income"]

    def test_regression_level_recorded(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        assert meta["regression_level"] == "annual"

    def test_months_included_recorded(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        assert "months_included" in meta
        assert "202308" in meta["months_included"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Regression results JSON: schema & model structure
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionResults:
    """Validate regression_results.json structure and content."""

    def test_model_keys_exist(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_results.json"
        with open(path) as f:
            results = json.load(f)
        assert MODEL_KEY_1 in results, f"Missing {MODEL_KEY_1} in regression results"
        assert MODEL_KEY_2 in results, f"Missing {MODEL_KEY_2} in regression results"

    @pytest.mark.parametrize("model_key", [MODEL_KEY_1, MODEL_KEY_2])
    def test_model_required_fields(self, pipeline_run: dict, model_key: str) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_results.json"
        with open(path) as f:
            results = json.load(f)
        m = results[model_key]
        for field in ("r_squared", "adj_r_squared", "n_obs", "n_predictors"):
            assert field in m, f"{model_key} missing {field}"
        assert m["n_obs"] > 0, f"{model_key} has 0 observations"
        assert "coefficients" in m, f"{model_key} missing coefficients"

    @pytest.mark.parametrize("model_key", [MODEL_KEY_1, MODEL_KEY_2])
    def test_model_coefficients_include_const(self, pipeline_run: dict, model_key: str) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_results.json"
        with open(path) as f:
            results = json.load(f)
        coeffs = results[model_key]["coefficients"]
        assert "const" in coeffs, f"{model_key} coefficients missing 'const'"

    @pytest.mark.parametrize("model_key", [MODEL_KEY_1, MODEL_KEY_2])
    def test_model_has_predictor_coefficients(self, pipeline_run: dict, model_key: str) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_results.json"
        with open(path) as f:
            results = json.load(f)
        coeffs = results[model_key]["coefficients"]
        non_const = {k for k in coeffs if k != "const"}
        assert len(non_const) >= 1, f"{model_key} has no predictor coefficients (only const)"

    def test_regression_summary_text(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "regression" / "regression_summary.txt"
        assert path.exists(), f"Missing regression summary: {path}"
        text = path.read_text()
        assert len(text) > 100, "Regression summary text suspiciously short"
        assert "R-squared" in text, "Regression summary should contain R-squared"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Regression: bg_month mode
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionBgMonth:
    """Tests specific to --regression-level bg_month."""

    def test_bg_month_pipeline_succeeds(self, pipeline_run_bg_month: dict) -> None:
        r = pipeline_run_bg_month
        assert r["returncode"] == 0, (
            f"bg_month pipeline exited {r['returncode']}.\nstdout: {r['stdout'][-2000:]}\nstderr: {r['stderr'][-2000:]}"
        )

    @pytest.mark.parametrize("filename", ALL_REGRESSION_FILES)
    def test_bg_month_regression_artifacts_exist(self, pipeline_run_bg_month: dict, filename: str) -> None:
        path = pipeline_run_bg_month["run_dir"] / "regression" / filename
        assert path.exists(), f"Missing bg_month regression artifact: {path}"

    def test_bg_month_metadata_records_level(self, pipeline_run_bg_month: dict) -> None:
        path = pipeline_run_bg_month["run_dir"] / "regression" / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        assert meta["regression_level"] == "bg_month"

    def test_bg_month_month_fixed_effects_in_results(self, pipeline_run_bg_month: dict) -> None:
        """In bg_month mode, model results should record month FE metadata.

        With a single-month fixture and drop_first=True, n_month_fe will be 0
        (degenerate case), but the structural fields must still be present.
        """
        path = pipeline_run_bg_month["run_dir"] / "regression" / "regression_results.json"
        with open(path) as f:
            results = json.load(f)
        for model_key in (MODEL_KEY_1, MODEL_KEY_2):
            m = results[model_key]
            assert "n_month_fe" in m, f"{model_key} missing n_month_fe"
            assert "month_fe_columns" in m, f"{model_key} missing month_fe_columns"
            assert isinstance(m["month_fe_columns"], list)

    @pytest.mark.parametrize(
        "filename",
        # regression_dataset_bg.parquet is excluded: bg_month mode joins to
        # the BG x month table (which has a ``month`` column), while annual
        # mode joins to the BG annual table (no ``month``).
        [f for f in REGRESSION_PARQUETS if f != "regression_dataset_bg.parquet"],
    )
    def test_bg_month_outcome_parquets_consistent_schema(
        self, pipeline_run: dict, pipeline_run_bg_month: dict, filename: str
    ) -> None:
        """Both modes should produce the same outcome parquet schemas."""
        annual_df = pl.read_parquet(pipeline_run["run_dir"] / "regression" / filename)
        bgm_df = pl.read_parquet(pipeline_run_bg_month["run_dir"] / "regression" / filename)
        assert set(annual_df.columns) == set(bgm_df.columns), (
            f"Schema mismatch in {filename}: annual={sorted(annual_df.columns)}, bg_month={sorted(bgm_df.columns)}"
        )

    def test_bg_month_dataset_has_month_column(self, pipeline_run_bg_month: dict) -> None:
        """In bg_month mode, regression_dataset_bg.parquet should include month."""
        df = pl.read_parquet(pipeline_run_bg_month["run_dir"] / "regression" / "regression_dataset_bg.parquet")
        assert "month" in df.columns, "regression_dataset_bg.parquet should include month in bg_month mode"

    def test_bg_month_manifest_records_level(self, pipeline_run_bg_month: dict) -> None:
        path = pipeline_run_bg_month["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        assert m["parameters"]["regression_level"] == "bg_month"


# ═══════════════════════════════════════════════════════════════════════════
# 10. Manifest correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestManifest:
    """Verify the run manifest JSON has complete provenance."""

    def test_manifest_required_fields(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        for key in (
            "run_id",
            "created_utc",
            "months",
            "month_summary",
            "inputs",
            "parameters",
            "steps_completed",
            "all_months_bills_rows",
        ):
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

    def test_manifest_all_months_bills_rows(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        assert m["all_months_bills_rows"] > 0, "all_months_bills_rows should be > 0"

    def test_manifest_all_months_bills_rows_equals_sum(self, pipeline_skip_regression: dict) -> None:
        """all_months_bills_rows should equal sum of per-month rows_bills."""
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        total = sum(v["rows_bills"] for v in m["month_summary"].values())
        assert m["all_months_bills_rows"] == total, (
            f"all_months_bills_rows={m['all_months_bills_rows']} != sum(rows_bills)={total}"
        )

    def test_manifest_month_summary_counts(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        summary = m["month_summary"]
        assert "202308" in summary, "Month 202308 not in month_summary"
        counts = summary["202308"]
        assert counts["rows_hourly_loads"] > 0, "rows_hourly_loads should be > 0"
        assert counts["rows_bills"] > 0, "rows_bills should be > 0"

    def test_manifest_steps_completed_skip_regression(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        steps = m["steps_completed"]
        assert "202308" in steps, "Month 202308 not in steps_completed"
        assert "all_months_bills" in steps, "all_months_bills not in steps_completed"
        assert "regression" not in steps, "regression should NOT be in steps_completed when skipped"

    def test_manifest_steps_completed_with_regression(self, pipeline_run: dict) -> None:
        path = pipeline_run["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        steps = m["steps_completed"]
        assert "all_months_bills" in steps, "all_months_bills not in steps_completed"
        assert "regression" in steps, "regression not in steps_completed"

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


# ═══════════════════════════════════════════════════════════════════════════
# 11. Skip-regression mode
# ═══════════════════════════════════════════════════════════════════════════


class TestSkipRegression:
    """Verify behavior when --skip-regression is set."""

    def test_skip_pipeline_succeeds(self, pipeline_skip_regression: dict) -> None:
        r = pipeline_skip_regression
        assert r["returncode"] == 0, (
            f"Skip-reg pipeline exited {r['returncode']}.\nstdout: {r['stdout'][-2000:]}\nstderr: {r['stderr'][-2000:]}"
        )

    def test_regression_dir_absent(self, pipeline_skip_regression: dict) -> None:
        """Regression directory should not exist or should be empty."""
        reg = pipeline_skip_regression["run_dir"] / "regression"
        assert not reg.exists() or not any(reg.iterdir()), (
            "Regression artifacts should not exist with --skip-regression"
        )

    def test_manifest_skip_regression_true(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        assert m["parameters"]["skip_regression"] is True

    def test_steps_completed_no_regression(self, pipeline_skip_regression: dict) -> None:
        path = pipeline_skip_regression["run_dir"] / "run_manifest.json"
        with open(path) as f:
            m = json.load(f)
        assert "regression" not in m["steps_completed"]

    def test_all_months_bills_still_produced(self, pipeline_skip_regression: dict) -> None:
        """Even with skip-regression, all_months_household_bills should exist."""
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        assert path.exists(), "all_months_household_bills.parquet missing with skip-regression"
        df = pl.read_parquet(path)
        assert df.height > 0, "all_months_household_bills is empty with skip-regression"

    def test_all_months_bills_has_month_column(self, pipeline_skip_regression: dict) -> None:
        """month column must be present even when regression is skipped."""
        path = pipeline_skip_regression["run_dir"] / "all_months_household_bills.parquet"
        df = pl.read_parquet(path)
        assert "month" in df.columns, "month column missing from all_months_household_bills"
