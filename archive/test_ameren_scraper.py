"""Tests for ameren_scraper.py functions."""

from scripts.data_collection.ameren_scraper import extract_date_from_filename


def test_extract_date_from_filename_valid():
    """Test extracting valid YYYYMM dates from filenames."""
    # Standard format
    assert extract_date_from_filename("usage_202401_data.csv") == "202401"
    assert extract_date_from_filename("202312_usage.csv") == "202312"

    # With multiple numbers - should get the valid date
    assert extract_date_from_filename("file_123456_202405_data.csv") == "202405"

    # Edge cases - valid months
    assert extract_date_from_filename("data_202001.csv") == "202001"  # January
    assert extract_date_from_filename("data_202012.csv") == "202012"  # December


def test_extract_date_from_filename_invalid():
    """Test handling of invalid dates."""
    # Invalid month
    assert extract_date_from_filename("usage_202013.csv") is None  # Month 13
    assert extract_date_from_filename("usage_202400.csv") is None  # Month 0

    # Invalid year
    assert extract_date_from_filename("usage_190001.csv") is None  # Year 1900
    assert extract_date_from_filename("usage_210001.csv") is None  # Year 2100

    # No date at all
    assert extract_date_from_filename("usage_data.csv") is None
    assert extract_date_from_filename("random_file.csv") is None

    # Wrong format (5 or 7 digits)
    assert extract_date_from_filename("usage_20241.csv") is None


def test_extract_date_from_filename_with_extension():
    """Test that .csv extension is properly handled."""
    assert extract_date_from_filename("usage_202403.csv") == "202403"
    assert extract_date_from_filename("usage_202403") == "202403"
