#!/usr/bin/env python3
"""Unit tests for predictor selection modes in build_regression_dataset.py.

Tests the three predictor modes (auto / core / explicit comma-separated) and
their edge cases, without running the full subprocess pipeline.  Uses synthetic
DataFrames that mirror the Census schema so tests are fast and deterministic.
"""

from __future__ import annotations

import polars as pl
import pytest

# Direct imports (not subprocess) so we can test individual functions in
# isolation without the overhead and fragility of CLI round-trips. The
# fail-loud tests in test_fail_loud_conditions.py cover subprocess behavior.
from analysis.rtp.build_regression_dataset import (
    EXCLUDE_COLS,
    _normalize_zip4_expr,
    _resolve_col,
    detect_predictors,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


# Fixture includes both CORE_PREDICTORS (income, building_pct), additional
# numeric columns, an all-null column, and an EXCLUDE_COLS member (NAME).
# This exercises every branch in detect_predictors with a single fixture.
@pytest.fixture()
def census_df() -> pl.DataFrame:
    """Synthetic census DataFrame with known columns and types."""
    return pl.DataFrame({
        "block_group_geoid": ["170310101001", "170310101002", "170310101003"],
        "NAME": ["BG 1", "BG 2", "BG 3"],
        "median_household_income": [55000.0, 72000.0, 43000.0],
        "old_building_pct": [0.35, 0.50, 0.20],
        "pct_college_degree": [0.40, 0.65, 0.30],
        "pct_renter_occupied": [0.60, 0.35, 0.55],
        "total_population": [1500, 2200, 900],
        "all_null_col": [None, None, None],
    }).cast({
        "median_household_income": pl.Float64,
        "old_building_pct": pl.Float64,
        "pct_college_degree": pl.Float64,
        "pct_renter_occupied": pl.Float64,
        "total_population": pl.Int64,
        "all_null_col": pl.Float64,
    })


@pytest.fixture()
def census_no_core() -> pl.DataFrame:
    """Census DataFrame without any CORE_PREDICTORS columns."""
    return pl.DataFrame({
        "block_group_geoid": ["170310101001", "170310101002"],
        "pct_college_degree": [0.40, 0.65],
        "total_population": [1500, 2200],
    }).cast({
        "pct_college_degree": pl.Float64,
        "total_population": pl.Int64,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 1. Auto mode
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoMode:
    """--predictors auto: infer all numeric cols minus excluded/all-null."""

    def test_auto_includes_numeric(self, census_df: pl.DataFrame) -> None:
        preds, _ = detect_predictors(census_df, mode="auto")
        # Should include all numeric columns except EXCLUDE_COLS and all-null
        assert "median_household_income" in preds
        assert "old_building_pct" in preds
        assert "pct_college_degree" in preds
        assert "total_population" in preds

    def test_auto_excludes_id_columns(self, census_df: pl.DataFrame) -> None:
        preds, _ = detect_predictors(census_df, mode="auto")
        for col in EXCLUDE_COLS:
            assert col not in preds, f"Auto mode should exclude {col}"
        assert "NAME" not in preds

    def test_auto_excludes_all_null(self, census_df: pl.DataFrame) -> None:
        preds, excluded = detect_predictors(census_df, mode="auto")
        assert "all_null_col" not in preds
        assert "all_null_col" in excluded

    def test_auto_returns_sorted(self, census_df: pl.DataFrame) -> None:
        preds, _ = detect_predictors(census_df, mode="auto")
        assert preds == sorted(preds), "Auto predictors should be sorted"

    def test_auto_excluded_sorted(self, census_df: pl.DataFrame) -> None:
        _, excluded = detect_predictors(census_df, mode="auto")
        assert excluded == sorted(excluded), "Excluded columns should be sorted"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Core mode
# ═══════════════════════════════════════════════════════════════════════════


class TestCoreMode:
    """--predictors core: just median_household_income + old_building_pct."""

    def test_core_returns_both(self, census_df: pl.DataFrame) -> None:
        preds, excluded = detect_predictors(census_df, mode="core")
        assert set(preds) == {"median_household_income", "old_building_pct"}
        assert excluded == []

    def test_core_partial_match(self) -> None:
        """If only one core predictor exists, return just that one."""
        df = pl.DataFrame({
            "block_group_geoid": ["170310101001"],
            "median_household_income": [55000.0],
        })
        preds, _ = detect_predictors(df, mode="core")
        assert preds == ["median_household_income"]

    def test_core_fails_when_none_present(self, census_no_core: pl.DataFrame) -> None:
        with pytest.raises(RuntimeError, match="--predictors core requested but none of"):
            detect_predictors(census_no_core, mode="core")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Explicit mode (comma-separated)
# ═══════════════════════════════════════════════════════════════════════════


class TestExplicitMode:
    """--predictors col1,col2: validate against census columns."""

    def test_explicit_valid_columns(self, census_df: pl.DataFrame) -> None:
        preds, excluded = detect_predictors(census_df, mode="pct_college_degree,total_population")
        assert preds == ["pct_college_degree", "total_population"]
        assert excluded == []

    def test_explicit_single_column(self, census_df: pl.DataFrame) -> None:
        preds, _ = detect_predictors(census_df, mode="median_household_income")
        assert preds == ["median_household_income"]

    def test_explicit_with_spaces(self, census_df: pl.DataFrame) -> None:
        preds, _ = detect_predictors(census_df, mode=" median_household_income , old_building_pct ")
        assert preds == ["median_household_income", "old_building_pct"]

    def test_explicit_missing_column(self, census_df: pl.DataFrame) -> None:
        with pytest.raises(RuntimeError, match="Requested predictors not found in census"):
            detect_predictors(census_df, mode="nonexistent_col")

    def test_explicit_partial_missing(self, census_df: pl.DataFrame) -> None:
        with pytest.raises(RuntimeError, match="Requested predictors not found"):
            detect_predictors(census_df, mode="median_household_income,fake_column")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ZIP+4 normalization
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeZip4:
    """Test _normalize_zip4_expr for various input formats."""

    def test_already_formatted(self) -> None:
        df = pl.DataFrame({"zip_code": ["60002-1102"]})
        result = df.with_columns(_normalize_zip4_expr())
        assert result["zip4"][0] == "60002-1102"

    def test_nine_digit_no_dash(self) -> None:
        df = pl.DataFrame({"zip_code": ["600021102"]})
        result = df.with_columns(_normalize_zip4_expr())
        assert result["zip4"][0] == "60002-1102"

    def test_five_digit_returns_null(self) -> None:
        df = pl.DataFrame({"zip_code": ["60002"]})
        result = df.with_columns(_normalize_zip4_expr())
        assert result["zip4"][0] is None

    def test_empty_string_returns_null(self) -> None:
        df = pl.DataFrame({"zip_code": [""]})
        result = df.with_columns(_normalize_zip4_expr())
        assert result["zip4"][0] is None

    def test_whitespace_stripped(self) -> None:
        df = pl.DataFrame({"zip_code": [" 60002-1102 "]})
        result = df.with_columns(_normalize_zip4_expr())
        assert result["zip4"][0] == "60002-1102"

    def test_mixed_batch(self) -> None:
        df = pl.DataFrame({"zip_code": ["60002-1102", "600021102", "60002", None]})
        result = df.with_columns(_normalize_zip4_expr())
        expected = ["60002-1102", "60002-1102", None, None]
        assert result["zip4"].to_list() == expected


# ═══════════════════════════════════════════════════════════════════════════
# 5. Outcome column resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveCol:
    """Test _resolve_col fallback logic."""

    def test_preferred_found(self) -> None:
        col, fallback = _resolve_col({"net_pct_savings", "pct_savings"}, "net_pct_savings", "pct_savings", "test")
        assert col == "net_pct_savings"
        assert fallback is False

    def test_fallback_used(self) -> None:
        col, fallback = _resolve_col({"pct_savings"}, "net_pct_savings", "pct_savings", "test")
        assert col == "pct_savings"
        assert fallback is True

    def test_neither_raises(self) -> None:
        with pytest.raises(RuntimeError, match="neither 'net_pct_savings' nor 'pct_savings'"):
            _resolve_col({"other_col"}, "net_pct_savings", "pct_savings", "test")
