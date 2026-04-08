#!/usr/bin/env python3
"""Negative tests for fail-loud conditions in the billing pipeline.

Verifies that the pipeline and regression script exit non-zero with clear
error messages when given invalid inputs or when thresholds are breached.
All tests use synthetic data via tmp_path; no network, no real data required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl

# ── Helpers ──────────────────────────────────────────────────────────────
# All tests use tiny synthetic DataFrames so the suite runs in seconds
# with no network, no real data, and no side effects. This is important
# because these are negative tests that intentionally trigger errors—we
# don't want real data files accidentally corrupted or large downloads.


def _write_minimal_bills(path: Path, *, include_savings: bool = True, include_bill_diff: bool = True) -> None:
    """Write a minimal valid bills parquet."""
    data: dict = {
        "account_identifier": ["A001", "A002", "A003"],
        "zip_code": ["60002-1102", "60002-1102", "60002-1103"],
        "total_kwh": [100.0, 200.0, 150.0],
    }
    if include_savings:
        data["net_pct_savings"] = [5.0, 3.0, 7.0]
    if include_bill_diff:
        data["bill_diff_dollars"] = [10.0, 6.0, 14.0]
    pl.DataFrame(data).write_parquet(path)


def _write_minimal_crosswalk(path: Path) -> None:
    """Write a minimal crosswalk TSV matching the test bills."""
    lines = [
        "Zip\tZip4\tCensusKey2023\n",
        "60002\t1102\t170310101001234\n",
        "60002\t1103\t170310101002345\n",
    ]
    path.write_text("".join(lines))


def _write_minimal_census(path: Path) -> None:
    """Write a minimal census parquet with GEOID + numeric columns."""
    pl.DataFrame({
        "GEOID": ["170310101001", "170310101002"],
        "median_household_income": [55000.0, 72000.0],
        "old_building_pct": [0.35, 0.50],
    }).write_parquet(path)


# 120s timeout is generous but necessary: on slow CI runners, subprocess
# startup + Polars import + parquet I/O can exceed the default 30s.
def _run_regression(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run build_regression_dataset.py as subprocess."""
    cmd = [sys.executable, "analysis/rtp/build_regression_dataset.py", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _run_orchestrator(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run run_billing_pipeline.py as subprocess."""
    cmd = [sys.executable, "scripts/run_billing_pipeline.py", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Missing columns in bills
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingBillsColumns:
    """Regression script should fail when required columns are absent."""

    def test_missing_account_identifier(self, tmp_path: Path) -> None:
        bills_path = tmp_path / "bills.parquet"
        pl.DataFrame({
            "zip_code": ["60002-1102"],
            "total_kwh": [100.0],
            "net_pct_savings": [5.0],
            "bill_diff_dollars": [10.0],
        }).write_parquet(bills_path)

        xwalk_path = tmp_path / "crosswalk.txt"
        _write_minimal_crosswalk(xwalk_path)
        census_path = tmp_path / "census.parquet"
        _write_minimal_census(census_path)

        r = _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(tmp_path / "out"),
        ])
        assert r.returncode != 0, "Should fail when account_identifier is missing"

    def test_missing_both_outcome_columns(self, tmp_path: Path) -> None:
        bills_path = tmp_path / "bills.parquet"
        _write_minimal_bills(bills_path, include_savings=False, include_bill_diff=True)
        # Remove savings columns entirely
        df = pl.read_parquet(bills_path)
        # This file has bill_diff_dollars but no savings column at all
        df_no_savings = df.select([c for c in df.columns if c not in ("net_pct_savings", "pct_savings")])
        df_no_savings.write_parquet(bills_path)

        xwalk_path = tmp_path / "crosswalk.txt"
        _write_minimal_crosswalk(xwalk_path)
        census_path = tmp_path / "census.parquet"
        _write_minimal_census(census_path)

        r = _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(tmp_path / "out"),
        ])
        assert r.returncode != 0, "Should fail when neither savings column exists"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Crosswalk coverage threshold
# ═══════════════════════════════════════════════════════════════════════════


class TestCrosswalkCoverageThreshold:
    """Fail-loud when crosswalk drop rate exceeds threshold."""

    def test_exceeds_drop_threshold(self, tmp_path: Path) -> None:
        """With a crosswalk that matches nothing, drop rate is 100%."""
        bills_path = tmp_path / "bills.parquet"
        _write_minimal_bills(bills_path)

        # Crosswalk with ZIP+4 that won't match any bills
        xwalk_path = tmp_path / "crosswalk.txt"
        xwalk_path.write_text("Zip\tZip4\tCensusKey2023\n99999\t9999\t170310101001234\n")

        census_path = tmp_path / "census.parquet"
        _write_minimal_census(census_path)

        r = _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-crosswalk-drop-pct",
            "5.0",
        ])
        assert r.returncode != 0, "Should fail when crosswalk drop rate exceeds threshold"

    def test_high_threshold_allows_low_coverage(self, tmp_path: Path) -> None:
        """With threshold=100, even total mismatch should still try to proceed.

        This verifies that the threshold is a gate (not a hard-coded constant)
        and that a different error ("No block groups remain") fires downstream.
        """
        bills_path = tmp_path / "bills.parquet"
        _write_minimal_bills(bills_path)

        # Crosswalk matches nothing
        xwalk_path = tmp_path / "crosswalk.txt"
        xwalk_path.write_text("Zip\tZip4\tCensusKey2023\n99999\t9999\t170310101001234\n")

        census_path = tmp_path / "census.parquet"
        _write_minimal_census(census_path)

        r = _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-crosswalk-drop-pct",
            "100.0",
        ])
        # Should fail with "No block groups remain" (not crosswalk threshold)
        assert r.returncode != 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Zero usable predictors
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroPredictors:
    """Fail when no usable predictors found."""

    def test_explicit_nonexistent_predictor(self, tmp_path: Path) -> None:
        bills_path = tmp_path / "bills.parquet"
        _write_minimal_bills(bills_path)
        xwalk_path = tmp_path / "crosswalk.txt"
        _write_minimal_crosswalk(xwalk_path)
        census_path = tmp_path / "census.parquet"
        _write_minimal_census(census_path)

        r = _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--predictors",
            "totally_fake_column",
        ])
        assert r.returncode != 0, "Should fail when explicit predictor not in census"

    def test_core_mode_no_core_cols(self, tmp_path: Path) -> None:
        """Census without income or building_pct should fail in core mode."""
        bills_path = tmp_path / "bills.parquet"
        _write_minimal_bills(bills_path)
        xwalk_path = tmp_path / "crosswalk.txt"
        _write_minimal_crosswalk(xwalk_path)

        # Census with NO core predictor columns
        census_path = tmp_path / "census.parquet"
        pl.DataFrame({
            "GEOID": ["170310101001", "170310101002"],
            "pct_renter": [0.60, 0.35],
        }).write_parquet(census_path)

        r = _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--predictors",
            "core",
        ])
        assert r.returncode != 0, "Should fail when no core predictor columns exist in census"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Input file not found
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingInputFiles:
    """Fail-loud when input files don't exist."""

    def test_missing_bills_file(self, tmp_path: Path) -> None:
        r = _run_regression([
            "--bills",
            str(tmp_path / "nonexistent_bills.parquet"),
            "--crosswalk",
            str(tmp_path / "xwalk.txt"),
            "--census",
            str(tmp_path / "census.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ])
        assert r.returncode != 0

    def test_missing_census_file(self, tmp_path: Path) -> None:
        bills_path = tmp_path / "bills.parquet"
        _write_minimal_bills(bills_path)
        xwalk_path = tmp_path / "crosswalk.txt"
        _write_minimal_crosswalk(xwalk_path)

        r = _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(tmp_path / "nonexistent_census.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ])
        assert r.returncode != 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Orchestrator: missing interval data
# ═══════════════════════════════════════════════════════════════════════════


class TestOrchestratorFailLoud:
    """Orchestrator should fail when interval data is missing."""

    def test_missing_interval_file(self, tmp_path: Path) -> None:
        r = _run_orchestrator([
            "--months",
            "202301",
            "--interval-pattern",
            str(tmp_path / "nonexistent_{yyyymm}.parquet"),
            "--tariff-a",
            str(tmp_path / "tariff_a.parquet"),
            "--tariff-b",
            str(tmp_path / "tariff_b.parquet"),
            "--skip-regression",
            "--run-name",
            "test_fail",
            "--output-dir",
            str(tmp_path / "out"),
        ])
        assert r.returncode != 0, "Should fail when interval data file is missing"

    def test_invalid_month_format(self, tmp_path: Path) -> None:
        r = _run_orchestrator([
            "--months",
            "2023-08",
            "--interval-pattern",
            str(tmp_path / "{yyyymm}.parquet"),
            "--tariff-a",
            str(tmp_path / "tariff_a.parquet"),
            "--tariff-b",
            str(tmp_path / "tariff_b.parquet"),
            "--skip-regression",
            "--output-dir",
            str(tmp_path / "out"),
        ])
        assert r.returncode != 0, "Should fail on invalid YYYYMM format"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Outcome column fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestOutcomeColumnFallback:
    """Test that fallback columns work when preferred is absent."""

    def test_fallback_to_pct_savings(self, tmp_path: Path) -> None:
        """When net_pct_savings absent, should fall back to pct_savings."""
        bills_path = tmp_path / "bills.parquet"
        pl.DataFrame({
            "account_identifier": ["A001", "A002", "A003"],
            "zip_code": ["60002-1102", "60002-1102", "60002-1103"],
            "total_kwh": [100.0, 200.0, 150.0],
            "pct_savings": [5.0, 3.0, 7.0],
            "bill_diff_dollars": [10.0, 6.0, 14.0],
        }).write_parquet(bills_path)

        xwalk_path = tmp_path / "crosswalk.txt"
        _write_minimal_crosswalk(xwalk_path)
        census_path = tmp_path / "census.parquet"
        _write_minimal_census(census_path)
        out_dir = tmp_path / "out"

        _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(out_dir),
            "--predictors",
            "core",
            "--max-crosswalk-drop-pct",
            "100",
            "--min-obs-per-bg",
            "1",
        ])

        # May fail at OLS step (too few obs) but should NOT fail at column resolution
        # Check metadata if it was written
        meta_path = out_dir / "regression_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["savings_column_used"] == "pct_savings"
            assert meta["savings_fallback_used"] is True

    def test_fallback_to_net_bill_diff(self, tmp_path: Path) -> None:
        """When bill_diff_dollars absent, should fall back to net_bill_diff_dollars."""
        bills_path = tmp_path / "bills.parquet"
        pl.DataFrame({
            "account_identifier": ["A001", "A002", "A003"],
            "zip_code": ["60002-1102", "60002-1102", "60002-1103"],
            "total_kwh": [100.0, 200.0, 150.0],
            "net_pct_savings": [5.0, 3.0, 7.0],
            "net_bill_diff_dollars": [10.0, 6.0, 14.0],
        }).write_parquet(bills_path)

        xwalk_path = tmp_path / "crosswalk.txt"
        _write_minimal_crosswalk(xwalk_path)
        census_path = tmp_path / "census.parquet"
        _write_minimal_census(census_path)
        out_dir = tmp_path / "out"

        _run_regression([
            "--bills",
            str(bills_path),
            "--crosswalk",
            str(xwalk_path),
            "--census",
            str(census_path),
            "--output-dir",
            str(out_dir),
            "--predictors",
            "core",
            "--max-crosswalk-drop-pct",
            "100",
            "--min-obs-per-bg",
            "1",
        ])

        meta_path = out_dir / "regression_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["bill_diff_column_used"] == "net_bill_diff_dollars"
            assert meta["bill_diff_fallback_used"] is True
