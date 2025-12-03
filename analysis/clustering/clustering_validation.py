"""
Validation utilities for clustering pipeline.

Validates data quality at each stage of the DTW clustering analysis. Designed
to work with both enriched data (with census variables) and raw processed data
(without census variables) to support flexible pipeline configurations.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class ClusteringDataValidator:
    """Validates data at each stage of clustering pipeline."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def reset(self):
        """Reset errors and warnings for a new validation run."""
        self.errors = []
        self.warnings = []

    def validate_enriched_data(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Validate interval-level data before aggregation.

        Works with both enriched data (with census variables) and raw processed
        data (without census variables). Geographic validation is skipped if
        census columns are not present.

        Args:
            df: Interval-level energy data

        Returns:
            Validation results with status, errors, warnings, and statistics
        """
        self.reset()
        logger.info("Validating interval data...")

        # Required energy columns (must be present)
        required_energy_cols = ["zip_code", "account_identifier", "datetime", "kwh"]

        # Required time columns (must be present)
        required_time_cols = ["date", "hour", "weekday", "is_weekend"]

        # Geographic columns (optional - only validated if present)
        geo_cols = ["block_group_geoid"]

        missing_energy = [c for c in required_energy_cols if c not in df.columns]
        missing_time = [c for c in required_time_cols if c not in df.columns]

        if missing_energy:
            self.errors.append(f"Missing energy columns: {missing_energy}")
        if missing_time:
            self.errors.append(f"Missing time columns: {missing_time}")

        # Check data completeness for critical columns
        critical_cols = ["zip_code", "account_identifier", "datetime", "kwh"]
        for col in critical_cols:
            if col not in df.columns:
                continue
            null_count = df[col].null_count()
            null_pct = (null_count / len(df)) * 100
            if null_pct > 5:
                self.errors.append(f"{col}: {null_pct:.1f}% null values (>5%)")
            elif null_pct > 0:
                self.warnings.append(f"{col}: {null_pct:.1f}% null values")

        # Geographic coverage - only check if column exists
        match_rate = None
        if "block_group_geoid" in df.columns:
            total_rows = len(df)
            matched_rows = df.filter(pl.col("block_group_geoid").is_not_null()).height
            match_rate = (matched_rows / total_rows) * 100

            if match_rate < 90:
                self.errors.append(f"Low geographic match rate: {match_rate:.1f}%")
            elif match_rate < 95:
                self.warnings.append(f"Geographic match rate below 95%: {match_rate:.1f}%")
        else:
            self.warnings.append("No geographic columns - running without census enrichment")

        # Check time features
        if "hour" in df.columns:
            hour_min, hour_max = df["hour"].min(), df["hour"].max()
            if hour_min < 0 or hour_max > 23:
                self.errors.append(f"Invalid hour values: {hour_min} to {hour_max}")

        if "weekday" in df.columns:
            weekday_min, weekday_max = df["weekday"].min(), df["weekday"].max()
            if weekday_min < 1 or weekday_max > 7:
                self.errors.append(f"Invalid weekday values: {weekday_min} to {weekday_max}")

        # Check energy values
        if "kwh" in df.columns:
            kwh_stats = df.select([
                pl.col("kwh").min().alias("min"),
                pl.col("kwh").max().alias("max"),
                pl.col("kwh").mean().alias("mean"),
            ]).to_dicts()[0]

            if kwh_stats["min"] is not None and kwh_stats["min"] < 0:
                self.warnings.append(f"Negative kWh values: min={kwh_stats['min']:.4f}")

            if kwh_stats["max"] is not None and kwh_stats["max"] > 100:
                self.warnings.append(f"Very high kWh values: max={kwh_stats['max']:.2f}")

        self._print_results("INTERVAL DATA VALIDATION")

        return {
            "status": "PASS" if not self.errors else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": {
                "n_rows": len(df),
                "n_accounts": df["account_identifier"].n_unique() if "account_identifier" in df.columns else None,
                "n_zip4s": df["zip_code"].n_unique() if "zip_code" in df.columns else None,
                "geographic_match_rate": match_rate,
                "has_census_data": "block_group_geoid" in df.columns,
            },
        }

    def validate_daily_profiles(self, df: pl.DataFrame, expected_intervals: int = 48) -> dict[str, Any]:
        """
        Validate daily load profiles after aggregation.

        Ensures profiles have the expected 48-point structure and reasonable values.

        Args:
            df: Daily profiles with 'profile' list column
            expected_intervals: Expected intervals per profile (default: 48)

        Returns:
            Validation results with status, errors, warnings, and statistics
        """
        self.reset()
        logger.info("Validating daily profiles...")

        # Check required columns
        required_cols = ["zip_code", "date", "profile"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
            self._print_results("PROFILE VALIDATION")
            return {"status": "FAIL", "errors": self.errors, "warnings": self.warnings}

        # Check profile completeness
        if "num_intervals" in df.columns:
            incomplete = df.filter(
                (pl.col("num_intervals") < expected_intervals - 1) | (pl.col("num_intervals") > expected_intervals)
            )
            if incomplete.height > 0:
                pct = (incomplete.height / len(df)) * 100
                self.warnings.append(f"{incomplete.height} profiles ({pct:.1f}%) have incorrect interval count")

        # Check profile array lengths
        profile_lengths = df.select(pl.col("profile").list.len().alias("len")).unique()
        unique_lengths = profile_lengths["len"].to_list()

        if len(unique_lengths) > 1:
            self.warnings.append(f"Inconsistent profile lengths: {unique_lengths}")
        elif unique_lengths[0] != expected_intervals:
            self.errors.append(f"Profile length {unique_lengths[0]} != expected {expected_intervals}")

        # Check for null profiles
        null_profiles = df.filter(pl.col("profile").is_null()).height
        if null_profiles > 0:
            self.errors.append(f"{null_profiles} null profiles found")

        # Check date coverage per ZIP+4
        dates_per_zip = df.group_by("zip_code").agg([
            pl.col("date").n_unique().alias("n_dates"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
        ])

        # Flag ZIP+4s with very few days
        sparse_zips = dates_per_zip.filter(pl.col("n_dates") < 5)
        if sparse_zips.height > 0:
            self.warnings.append(f"{sparse_zips.height} ZIP+4s have fewer than 5 days of data")

        # Check for reasonable values in profiles
        if df.height > 0:
            value_stats = (
                df.select(pl.col("profile").list.explode().alias("value"))
                .select([
                    pl.col("value").min().alias("min"),
                    pl.col("value").max().alias("max"),
                ])
                .to_dicts()[0]
            )

            if value_stats["min"] is not None and value_stats["min"] < 0:
                self.warnings.append(f"Negative values in profiles: min={value_stats['min']:.2f}")

            if value_stats["max"] is not None and value_stats["max"] > 10000:
                self.warnings.append(f"Very high values in profiles: max={value_stats['max']:.2f}")

        self._print_results("DAILY PROFILES VALIDATION")

        return {
            "status": "PASS" if not self.errors else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": {
                "n_profiles": len(df),
                "n_zip4s": df["zip_code"].n_unique(),
                "n_dates": df["date"].n_unique(),
                "profile_length": unique_lengths[0] if unique_lengths else None,
            },
        }

    def validate_demographics(self, df: pl.DataFrame, required_zip4s: set[str] | None = None) -> dict[str, Any]:
        """
        Validate census demographics data.

        Args:
            df: Demographics data with ZIP+4 codes
            required_zip4s: Set of ZIP+4 codes that must have demographics

        Returns:
            Validation results with status, errors, warnings, and statistics
        """
        self.reset()
        logger.info("Validating demographics data...")

        # Handle empty demographics (valid when running without census data)
        if df is None or df.height == 0:
            self.warnings.append("No demographics data - running without census enrichment")
            self._print_results("DEMOGRAPHICS VALIDATION")
            return {
                "status": "PASS",
                "errors": [],
                "warnings": self.warnings,
                "stats": {"n_zip4s": 0, "n_demo_vars": 0},
            }

        # Check required columns
        required_cols = ["zip_code"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")

        # Check GEOID format if present
        if "block_group_geoid" in df.columns:
            non_null_geoids = df.filter(pl.col("block_group_geoid").is_not_null())

            if non_null_geoids.height > 0:
                geoid_lengths = non_null_geoids["block_group_geoid"].str.len_chars().unique().to_list()
                if geoid_lengths and 12 not in geoid_lengths:
                    self.errors.append(f"Block Group GEOIDs should be 12 digits, got: {geoid_lengths}")

                # Check Illinois state FIPS
                non_il = non_null_geoids.filter(~pl.col("block_group_geoid").str.starts_with("17")).height
                if non_il > 0:
                    self.warnings.append(f"{non_il} GEOIDs don't start with '17' (Illinois)")

        # Check coverage of required ZIP+4s
        if required_zip4s:
            present_zips = set(df["zip_code"].to_list())
            missing_zips = required_zip4s - present_zips
            if missing_zips:
                self.warnings.append(f"{len(missing_zips)} required ZIP+4s missing demographics")

        # Count demographic columns
        demo_cols = [c for c in df.columns if c not in ["zip_code", "block_group_geoid", "Urban_Rural_Classification"]]

        # Check for excessive nulls in demographic columns
        high_null_cols = []
        for col in demo_cols:
            null_pct = (df[col].null_count() / len(df)) * 100
            if null_pct > 50:
                high_null_cols.append(f"{col} ({null_pct:.1f}%)")

        if high_null_cols:
            self.warnings.append(f"{len(high_null_cols)} columns with >50% nulls")

        self._print_results("DEMOGRAPHICS VALIDATION")

        return {
            "status": "PASS" if not self.errors else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": {
                "n_zip4s": len(df),
                "n_demo_vars": len(demo_cols),
                "high_null_vars": len(high_null_cols),
            },
        }

    def _print_results(self, title: str):
        """Print validation results summary."""
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print("=" * 80)

        if self.errors:
            print(f"\n❌ FAILED with {len(self.errors)} error(s):")
            for err in self.errors:
                print(f"  - {err}")
        else:
            print("\n✅ PASSED all critical checks")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warning(s):")
            for warn in self.warnings:
                print(f"  - {warn}")


def validate_interval_completeness(
    df: pl.DataFrame, account_col: str = "account_identifier", date_col: str = "date", expected_intervals: int = 48
) -> pl.DataFrame:
    """
    Check interval completeness for each account-date combination.

    Args:
        df: Interval-level data
        account_col: Account identifier column name
        date_col: Date column name
        expected_intervals: Expected intervals per day (default: 48)

    Returns:
        DataFrame with completeness statistics per account-date
    """
    completeness = (
        df.group_by([account_col, date_col])
        .agg([
            pl.len().alias("n_intervals"),
            pl.col("kwh").is_null().sum().alias("n_null_kwh"),
        ])
        .with_columns([
            (pl.col("n_intervals") == expected_intervals).alias("is_complete"),
            ((pl.col("n_intervals") - pl.col("n_null_kwh")) / expected_intervals * 100).alias("completeness_pct"),
        ])
    )

    return completeness


def check_for_duplicates(df: pl.DataFrame, key_cols: list[str]) -> tuple[int, pl.DataFrame | None]:
    """
    Check for duplicate records based on key columns.

    Args:
        df: DataFrame to check
        key_cols: Columns that should be unique together

    Returns:
        Tuple of (duplicate_count, duplicate_records_df)
    """
    duplicates = df.group_by(key_cols).agg(pl.len().alias("count")).filter(pl.col("count") > 1)

    n_dups = duplicates.height

    if n_dups > 0:
        dup_keys = duplicates.select(key_cols)
        dup_records = df.join(dup_keys, on=key_cols, how="inner")
        return n_dups, dup_records

    return 0, None


def validate_time_series_array(
    profiles: list[list[float]], expected_length: int = 48, max_profiles: int = 5000
) -> dict[str, Any]:
    """
    Validate time series arrays for clustering.

    Args:
        profiles: List of time series profiles
        expected_length: Expected length of each profile
        max_profiles: Maximum profiles to validate (samples if exceeded)

    Returns:
        Validation results dictionary
    """
    import numpy as np

    issues = []
    warnings = []

    if len(profiles) > max_profiles:
        logger.info(f"Sampling {max_profiles} of {len(profiles)} profiles for validation")
        profiles = profiles[:max_profiles]

    # Check lengths
    lengths = [len(p) for p in profiles]
    unique_lengths = set(lengths)

    if len(unique_lengths) > 1:
        issues.append(f"Inconsistent lengths: {unique_lengths}")
    elif list(unique_lengths)[0] != expected_length:
        issues.append(f"Expected length {expected_length}, got {list(unique_lengths)[0]}")

    # Check for NaN/inf values
    arr = np.array(profiles, dtype=np.float32)
    if np.any(np.isnan(arr)):
        issues.append("NaN values detected in profiles")
    if np.any(np.isinf(arr)):
        issues.append("Infinite values detected in profiles")

    # Check value ranges
    if np.any(arr < 0):
        warnings.append(f"Negative values detected: min={arr.min():.2f}")

    if arr.max() > 10000:
        warnings.append(f"Very high values detected: max={arr.max():.2f}")

    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "n_profiles": len(profiles),
            "profile_length": list(unique_lengths)[0] if len(unique_lengths) == 1 else None,
            "min_value": float(arr.min()),
            "max_value": float(arr.max()),
            "mean_value": float(arr.mean()),
        },
    }
