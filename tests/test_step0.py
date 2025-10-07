"""Tests for Step 0: AWS ingestion and transformation pipeline"""

import polars as pl
import pytest

from smart_meter_analysis.step0_transform import (
    daily_interval_qc,
    dst_transition_dates,
)

# Check if AWS credentials are available
try:
    import boto3

    s3 = boto3.client("s3")
    s3.list_buckets()
    HAS_AWS_CREDENTIALS = True
except Exception:
    HAS_AWS_CREDENTIALS = False

# Skip marker for AWS tests
requires_aws = pytest.mark.skipif(not HAS_AWS_CREDENTIALS, reason="AWS credentials not available (expected in CI)")


# Only import AWS functions if credentials available
if HAS_AWS_CREDENTIALS:
    from smart_meter_analysis.step0_aws import (
        list_s3_files,
        process_month_batch,
        process_single_csv,
    )


@pytest.fixture(scope="module")
def s3_files():
    """Get list of S3 files for testing"""
    if not HAS_AWS_CREDENTIALS:
        pytest.skip("AWS credentials not available")
    return list_s3_files(year_month="202308", max_files=3)


@pytest.fixture(scope="module")
def sample_df(s3_files):
    """Process a single CSV for testing"""
    if not HAS_AWS_CREDENTIALS:
        pytest.skip("AWS credentials not available")
    return process_single_csv(s3_files[0])


@requires_aws
def test_aws_listing():
    """Test that we can list files from S3"""
    files = list_s3_files(year_month="202308", max_files=5)
    assert len(files) > 0
    assert all(f.endswith(".csv") for f in files)
    assert all("202308" in f for f in files)


@requires_aws
def test_single_file_processing(sample_df):
    """Test processing a single CSV file"""
    # Check shape
    assert sample_df.height > 0

    # Check required columns exist
    required_cols = ["zip_code", "account_identifier", "datetime", "kwh", "date", "hour", "weekday"]
    for col in required_cols:
        assert col in sample_df.columns

    # Check data types
    assert sample_df["datetime"].dtype == pl.Datetime
    assert sample_df["kwh"].dtype == pl.Float64
    assert sample_df["hour"].dtype == pl.Int8

    # Check data quality
    assert sample_df["account_identifier"].n_unique() > 0
    assert sample_df["date"].min() is not None
    assert sample_df["date"].max() is not None


@requires_aws
def test_qc_checks(sample_df):
    """Test quality control checks"""
    qc = daily_interval_qc(sample_df)

    # Check QC columns exist
    assert "day_type" in qc.columns
    assert "n_intervals" in qc.columns
    assert "is_dst_transition" in qc.columns

    # Check day types are valid
    valid_day_types = {"normal", "spring_forward", "fall_back", "odd"}
    assert set(qc["day_type"].unique()) <= valid_day_types

    # Most August days should be normal (48 intervals)
    normal_days = qc.filter(pl.col("day_type") == "normal").height
    total_days = qc.height
    assert normal_days > total_days * 0.5  # At least 50% normal days


@requires_aws
def test_batch_processing(tmp_path):
    """Test batch processing multiple files"""
    output_path = tmp_path / "test_batch.parquet"

    process_month_batch(
        year_month="202308",
        output_path=output_path,
        max_files=2,  # Just 2 files for speed
    )

    # Check file was created
    assert output_path.exists()

    # Read and verify
    df = pl.read_parquet(output_path)
    assert df.height > 0
    assert "account_identifier" in df.columns
    assert df["account_identifier"].n_unique() > 0


@requires_aws
def test_dst_detection(sample_df):
    """Test DST transition detection"""
    dst_dates = dst_transition_dates(sample_df)

    # August should have no DST transitions
    # But if there are any odd interval counts, they might be flagged
    # Just check the function runs without error
    assert "date" in dst_dates.columns
    assert "day_type" in dst_dates.columns


# Non-AWS tests that should always run
def test_imports():
    """Test that core modules can be imported"""
    from smart_meter_analysis import step0_aws, step0_transform

    assert step0_transform is not None
    assert step0_aws is not None
