"""
Validation script for enriched ComEd data.

Performs data quality checks on enriched parquet files to ensure:
- Schema correctness
- Data completeness
- Geographic coverage
- Demographic data integrity
- Time feature preservation

Usage:
    python tests/validate_enriched_data.py data/analysis/202308/enriched.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


class EnrichedDataValidator:
    """Validates enriched energy data quality."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.df: pl.DataFrame | None = None
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def load_data(self) -> bool:
        """Load the parquet file."""
        try:
            self.df = pl.read_parquet(self.filepath)
            print(f"✓ Loaded {self.filepath}")
            print(f"  Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            return True
        except Exception as e:
            self.errors.append(f"Failed to load file: {e}")
            return False

    def check_required_columns(self) -> bool:
        """Verify all required columns are present."""
        required_energy_cols = [
            "zip_code",
            "account_identifier",
            "delivery_service_class",
            "delivery_service_name",
            "datetime",
            "kwh",
        ]

        required_time_cols = ["date", "hour", "weekday", "is_weekend"]

        required_geo_cols = ["block_group_geoid"]

        missing_energy = [c for c in required_energy_cols if c not in self.df.columns]
        missing_time = [c for c in required_time_cols if c not in self.df.columns]
        missing_geo = [c for c in required_geo_cols if c not in self.df.columns]

        if missing_energy:
            self.errors.append(f"Missing energy columns: {missing_energy}")

        if missing_time:
            self.errors.append(f"Missing time columns: {missing_time}")

        if missing_geo:
            self.errors.append(f"Missing geographic columns: {missing_geo}")

        if not (missing_energy or missing_time or missing_geo):
            print("✓ All required columns present")
            return True
        return False

    def check_data_completeness(self) -> bool:
        """Check for null values in critical columns."""
        critical_cols = ["zip_code", "account_identifier", "datetime", "kwh"]

        for col in critical_cols:
            if col not in self.df.columns:
                continue

            null_count = self.df[col].null_count()
            null_pct = (null_count / len(self.df)) * 100

            if null_pct > 5:
                self.errors.append(f"{col}: {null_pct:.1f}% null values (critical)")
            elif null_pct > 0:
                self.warnings.append(f"{col}: {null_pct:.1f}% null values")

        if not self.errors:
            print("✓ Data completeness check passed")
            return True
        return False

    def check_geographic_coverage(self) -> bool:
        """Verify geographic enrichment coverage."""
        total_rows = len(self.df)
        matched_rows = self.df.filter(pl.col("block_group_geoid").is_not_null()).shape[0]
        match_rate = (matched_rows / total_rows) * 100

        print(f"✓ Geographic coverage: {match_rate:.1f}%")
        print(f"  Matched: {matched_rows:,} / {total_rows:,} records")

        if match_rate < 90:
            self.errors.append(f"Low geographic match rate: {match_rate:.1f}%")
            return False
        elif match_rate < 95:
            self.warnings.append(f"Geographic match rate below 95%: {match_rate:.1f}%")

        # Check unique block groups
        n_block_groups = self.df["block_group_geoid"].n_unique()
        print(f"  Block groups: {n_block_groups:,}")

        return True

    def check_demographic_data(self) -> bool:
        """Verify demographic variables are present and valid."""
        expected_demo_vars = ["Total_Households", "Median_Household_Income", "Avg_Household_Size"]

        present_vars = [v for v in expected_demo_vars if v in self.df.columns]

        if not present_vars:
            self.errors.append("No demographic variables found")
            return False

        # Count total demographic columns
        excluded_cols = {
            "zip_code",
            "account_identifier",
            "datetime",
            "kwh",
            "date",
            "hour",
            "weekday",
            "is_weekend",
            "block_group_geoid",
            "delivery_service_class",
            "delivery_service_name",
            "is_spring_forward_day",
            "is_fall_back_day",
            "is_dst_day",
        }
        demo_cols = [c for c in self.df.columns if c not in excluded_cols]

        print(f"✓ Demographic variables: {len(demo_cols)}")
        print(f"  Examples: {', '.join(present_vars)}")

        return True

    def check_time_features(self) -> bool:
        """Verify time features are correctly populated."""
        checks_passed = True

        # Check hour range
        if "hour" in self.df.columns:
            hour_min = self.df["hour"].min()
            hour_max = self.df["hour"].max()
            if hour_min < 0 or hour_max > 23:
                self.errors.append(f"Invalid hour values: {hour_min} to {hour_max}")
                checks_passed = False
            else:
                print(f"✓ Hour range: {hour_min}-{hour_max}")

        # Check weekday range
        if "weekday" in self.df.columns:
            weekday_min = self.df["weekday"].min()
            weekday_max = self.df["weekday"].max()
            if weekday_min < 1 or weekday_max > 7:
                self.errors.append(f"Invalid weekday values: {weekday_min} to {weekday_max}")
                checks_passed = False
            else:
                print(f"✓ Weekday range: {weekday_min}-{weekday_max}")

        return checks_passed

    def check_energy_data(self) -> bool:
        """Validate energy usage values."""
        if "kwh" not in self.df.columns:
            return True

        print("✓ Energy data statistics:")
        print(f"  Mean: {self.df['kwh'].mean():.3f} kWh")
        print(f"  Median: {self.df['kwh'].median():.3f} kWh")
        print(f"  Max: {self.df['kwh'].max():.3f} kWh")

        # Check for negative values
        negative_count = self.df.filter(pl.col("kwh") < 0).shape[0]
        if negative_count > 0:
            self.warnings.append(f"Negative kWh values: {negative_count} records")

        # Check for unreasonably high values (>100 kWh per 30-min interval = 200 kW)
        high_count = self.df.filter(pl.col("kwh") > 100).shape[0]
        if high_count > 0:
            self.warnings.append(f"Very high kWh values (>100): {high_count} records")

        return True

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("\n" + "=" * 80)
        print(f"VALIDATING: {self.filepath.name}")
        print("=" * 80 + "\n")

        if not self.load_data():
            return False

        print()
        self.check_required_columns()
        print()
        self.check_data_completeness()
        print()
        self.check_geographic_coverage()
        print()
        self.check_demographic_data()
        print()
        self.check_time_features()
        print()
        self.check_energy_data()

        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
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

        print()

        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate enriched ComEd data quality")
    parser.add_argument("filepath", type=Path, help="Path to enriched parquet file to validate")

    args = parser.parse_args()

    if not args.filepath.exists():
        print(f"ERROR: File not found: {args.filepath}")
        sys.exit(1)

    validator = EnrichedDataValidator(args.filepath)
    success = validator.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
