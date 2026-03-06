#!/usr/bin/env python3
"""Unit tests for analysis/rtp/compute_hourly_loads.py.

Tests the compute_hourly_loads() function directly using synthetic DataFrames
so that the suite requires no large data files and runs in milliseconds.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from analysis.rtp.compute_hourly_loads import compute_hourly_loads

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_interval_df(
    accounts: list[str],
    start: dt.datetime,
    n_intervals: int,
    *,
    kwh_per_interval: float = 0.5,
    zip_code: str = "60601-1234",
) -> pl.DataFrame:
    """Build a minimal 30-minute interval DataFrame for testing."""
    rows = []
    for acct in accounts:
        for i in range(n_intervals):
            ts = start + dt.timedelta(minutes=30 * i)
            rows.append({
                "account_identifier": acct,
                "zip_code": zip_code,
                "datetime": ts,
                "energy_kwh": kwh_per_interval,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("datetime").cast(pl.Datetime("us")),
        pl.col("energy_kwh").cast(pl.Float64),
    )


def _write_parquet(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. Basic aggregation
# ═══════════════════════════════════════════════════════════════════════════


class TestBasicAggregation:
    """Core summation behaviour: 30-min intervals → hourly kWh."""

    def test_two_intervals_sum_to_one_hour(self, tmp_path: Path) -> None:
        """Two 0.5 kWh 30-min intervals within the same hour → kwh_hour = 1.0."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001"], start, n_intervals=2, kwh_per_interval=0.5)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        assert result.height == 1
        assert result["kwh_hour"][0] == pytest.approx(1.0)

    def test_48_intervals_produce_24_hours(self, tmp_path: Path) -> None:
        """One full day of 30-min data for a single account → 24 hourly rows."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001"], start, n_intervals=48, kwh_per_interval=1.0)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        assert result.height == 24

    def test_each_hour_sums_two_intervals(self, tmp_path: Path) -> None:
        """Each hour must equal exactly 2 x kwh_per_interval = 2.0."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001"], start, n_intervals=48, kwh_per_interval=1.0)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        for v in result["kwh_hour"].to_list():
            assert v == pytest.approx(2.0)

    def test_multiple_accounts_independent(self, tmp_path: Path) -> None:
        """Three accounts each get their own 24 hourly rows — no cross-contamination."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001", "A002", "A003"], start, n_intervals=48)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        assert result.height == 3 * 24
        for acct in ["A001", "A002", "A003"]:
            acct_rows = result.filter(pl.col("account_identifier") == acct)
            assert acct_rows.height == 24

    def test_hour_chicago_truncated_to_hour(self, tmp_path: Path) -> None:
        """hour_chicago must have zero minutes/seconds — truncated to the hour."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001"], start, n_intervals=4)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        minutes = result["hour_chicago"].dt.minute().to_list()
        assert all(m == 0 for m in minutes)
        seconds = result["hour_chicago"].dt.second().to_list()
        assert all(s == 0 for s in seconds)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Output schema
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputSchema:
    """Validate the columns emitted by compute_hourly_loads."""

    def test_required_columns_present(self, tmp_path: Path) -> None:
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001"], start, n_intervals=2)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        required = {"account_identifier", "zip_code", "hour_chicago", "kwh_hour"}
        assert required <= set(result.columns)

    def test_no_null_values_in_key_columns(self, tmp_path: Path) -> None:
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001", "A002"], start, n_intervals=4)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        for col in ("account_identifier", "zip_code", "hour_chicago", "kwh_hour"):
            n_null = result[col].null_count()
            assert n_null == 0, f"Column {col} has {n_null} nulls"

    def test_kwh_hour_non_negative(self, tmp_path: Path) -> None:
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001"], start, n_intervals=4, kwh_per_interval=0.25)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        assert all(v >= 0 for v in result["kwh_hour"].to_list())


# ═══════════════════════════════════════════════════════════════════════════
# 3. DST fall-back: duplicate naive timestamps aggregate correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestDSTFallback:
    """
    During fall-back, two meter reads at e.g. 01:00 CDT and 01:00 CST share
    the same naive local time.  compute_hourly_loads must sum all intervals
    within the same naive hour — including duplicates — without dropping rows.
    """

    def test_fallback_intervals_both_summed(self, tmp_path: Path) -> None:
        """Four 30-min reads at the same naive hour (fall-back) → kwh_hour = 2.0."""
        # 2023-11-05: fall-back at 2 AM CDT → 1 AM CST
        # Both the CDT 01:00/01:30 and CST 01:00/01:30 map to the same naive hour.
        ts_01_00 = dt.datetime(2023, 11, 5, 1, 0)
        ts_01_30 = dt.datetime(2023, 11, 5, 1, 30)

        # Simulate 4 intervals that all truncate to 01:00 (the CDT set + CST set)
        rows = [
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "datetime": ts_01_00,
                "energy_kwh": 0.5,
            },
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "datetime": ts_01_30,
                "energy_kwh": 0.5,
            },
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "datetime": ts_01_00,
                "energy_kwh": 0.5,
            },
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "datetime": ts_01_30,
                "energy_kwh": 0.5,
            },
        ]
        df = pl.DataFrame(rows).with_columns(
            pl.col("datetime").cast(pl.Datetime("us")),
            pl.col("energy_kwh").cast(pl.Float64),
        )
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=False)

        result = pl.read_parquet(out)
        # Two distinct naive hours: 01:00 and 01:30 (after truncation to hour,
        # ts_01_30 truncates to 01:00 too — so actually all 4 truncate to 01:00)
        # All four 0.5 kWh intervals truncate to 01:00 → kwh_hour = 2.0
        assert result.height == 1
        assert result["kwh_hour"][0] == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Sort output
# ═══════════════════════════════════════════════════════════════════════════


class TestSortOutput:
    """Verify sort_output=True produces (zip_code, account_identifier, hour_chicago) order."""

    def test_sorted_output_is_ordered(self, tmp_path: Path) -> None:
        start = dt.datetime(2023, 7, 1, 0, 0)
        # Deliberately put accounts in reverse order so unsorted would fail
        df = _make_interval_df(["Z999", "A001", "M500"], start, n_intervals=4)
        inp = _write_parquet(df, tmp_path / "interval.parquet")
        out = tmp_path / "loads.parquet"

        compute_hourly_loads(inp, None, out, sort_output=True)

        result = pl.read_parquet(out)
        expected_order = result.sort(["zip_code", "account_identifier", "hour_chicago"])
        assert result.rows() == expected_order.rows()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Multi-file chunked processing
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkedMultiFile:
    """When input is a directory with multiple files, chunked aggregation must
    produce the same result as a single-file run."""

    def test_multi_file_matches_single_file(self, tmp_path: Path) -> None:
        """Splitting data across 3 files must give identical output to one file."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001", "A002"], start, n_intervals=48)

        # Single-file reference
        single_dir = tmp_path / "single"
        _write_parquet(df, single_dir / "all.parquet")
        out_single = tmp_path / "out_single.parquet"
        compute_hourly_loads(single_dir / "all.parquet", None, out_single, sort_output=True)

        # Split across 3 files in a directory (triggers chunked path)
        multi_dir = tmp_path / "multi"
        multi_dir.mkdir()
        chunks = [df[:32], df[32:64], df[64:]]
        for i, chunk in enumerate(chunks):
            chunk.write_parquet(multi_dir / f"part_{i}.parquet")
        out_multi = tmp_path / "out_multi.parquet"
        compute_hourly_loads(multi_dir, None, out_multi, sort_output=True)

        ref = pl.read_parquet(out_single)
        got = pl.read_parquet(out_multi)
        assert ref.shape == got.shape
        assert ref.rows() == got.rows()

    def test_multi_file_re_aggregates_split_keys(self, tmp_path: Path) -> None:
        """Same (account, zip, hour) key split across two files must sum correctly."""
        ts = dt.datetime(2023, 7, 1, 10, 0)  # 10:00
        row_a = {
            "account_identifier": "A001",
            "zip_code": "60601-1234",
            "datetime": ts,
            "energy_kwh": 0.3,
        }
        row_b = {**row_a, "energy_kwh": 0.7}

        df_a = pl.DataFrame([row_a]).with_columns(
            pl.col("datetime").cast(pl.Datetime("us")),
            pl.col("energy_kwh").cast(pl.Float64),
        )
        df_b = pl.DataFrame([row_b]).with_columns(
            pl.col("datetime").cast(pl.Datetime("us")),
            pl.col("energy_kwh").cast(pl.Float64),
        )

        multi_dir = tmp_path / "multi"
        multi_dir.mkdir()
        df_a.write_parquet(multi_dir / "part_0.parquet")
        df_b.write_parquet(multi_dir / "part_1.parquet")

        out = tmp_path / "loads.parquet"
        compute_hourly_loads(multi_dir, None, out, sort_output=False)

        result = pl.read_parquet(out)
        assert result.height == 1
        assert result["kwh_hour"][0] == pytest.approx(1.0)

    def test_multi_file_with_cluster_filter(self, tmp_path: Path) -> None:
        """Cluster assignment filtering works with chunked multi-file path."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001", "A002", "A003"], start, n_intervals=4)

        multi_dir = tmp_path / "multi"
        multi_dir.mkdir()
        df[:6].write_parquet(multi_dir / "part_0.parquet")
        df[6:].write_parquet(multi_dir / "part_1.parquet")

        assignments = pl.DataFrame({"account_identifier": ["A001"]})
        asgn_path = tmp_path / "assignments.parquet"
        assignments.write_parquet(asgn_path)

        out = tmp_path / "loads.parquet"
        compute_hourly_loads(multi_dir, asgn_path, out, sort_output=False)

        result = pl.read_parquet(out)
        assert set(result["account_identifier"].unique().to_list()) == {"A001"}

    def test_multi_file_sorted(self, tmp_path: Path) -> None:
        """sort_output=True works with chunked multi-file path."""
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["Z999", "A001"], start, n_intervals=4)

        multi_dir = tmp_path / "multi"
        multi_dir.mkdir()
        df[:4].write_parquet(multi_dir / "part_0.parquet")
        df[4:].write_parquet(multi_dir / "part_1.parquet")

        out = tmp_path / "loads.parquet"
        compute_hourly_loads(multi_dir, None, out, sort_output=True)

        result = pl.read_parquet(out)
        expected_order = result.sort(["zip_code", "account_identifier", "hour_chicago"])
        assert result.rows() == expected_order.rows()


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cluster assignment filtering
# ═══════════════════════════════════════════════════════════════════════════


class TestClusterAssignmentFilter:
    """When cluster_assignments is given, only those accounts are retained."""

    def test_semi_join_restricts_accounts(self, tmp_path: Path) -> None:
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001", "A002", "A003"], start, n_intervals=2)
        inp = _write_parquet(df, tmp_path / "interval.parquet")

        # Only A001 in cluster assignments
        assignments = pl.DataFrame({"account_identifier": ["A001"]})
        asgn_path = tmp_path / "assignments.parquet"
        assignments.write_parquet(asgn_path)

        out = tmp_path / "loads.parquet"
        compute_hourly_loads(inp, asgn_path, out, sort_output=False)

        result = pl.read_parquet(out)
        accounts = set(result["account_identifier"].unique().to_list())
        assert accounts == {"A001"}

    def test_no_rows_when_no_accounts_match(self, tmp_path: Path) -> None:
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001", "A002"], start, n_intervals=2)
        inp = _write_parquet(df, tmp_path / "interval.parquet")

        assignments = pl.DataFrame({"account_identifier": ["ZZZZ"]})
        asgn_path = tmp_path / "assignments.parquet"
        assignments.write_parquet(asgn_path)

        out = tmp_path / "loads.parquet"
        compute_hourly_loads(inp, asgn_path, out, sort_output=False)

        result = pl.read_parquet(out)
        assert result.height == 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Fail-loud conditions
# ═══════════════════════════════════════════════════════════════════════════


class TestFailLoud:
    """compute_hourly_loads must raise clearly on bad inputs."""

    def test_missing_input_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_hourly_loads(
                tmp_path / "nonexistent.parquet",
                None,
                tmp_path / "out.parquet",
                sort_output=False,
            )

    def test_missing_energy_kwh_column_raises(self, tmp_path: Path) -> None:
        df = pl.DataFrame({
            "account_identifier": ["A001"],
            "zip_code": ["60601-0001"],
            "datetime": [dt.datetime(2023, 7, 1, 0, 0)],
            # energy_kwh deliberately absent
        })
        inp = _write_parquet(df, tmp_path / "interval.parquet")

        with pytest.raises(ValueError, match="missing required columns"):
            compute_hourly_loads(inp, None, tmp_path / "out.parquet", sort_output=False)

    def test_missing_datetime_column_raises(self, tmp_path: Path) -> None:
        df = pl.DataFrame({
            "account_identifier": ["A001"],
            "zip_code": ["60601-0001"],
            "energy_kwh": [0.5],
            # datetime deliberately absent
        })
        inp = _write_parquet(df, tmp_path / "interval.parquet")

        with pytest.raises(ValueError, match="missing required columns"):
            compute_hourly_loads(inp, None, tmp_path / "out.parquet", sort_output=False)

    def test_missing_cluster_assignments_file_raises(self, tmp_path: Path) -> None:
        start = dt.datetime(2023, 7, 1, 0, 0)
        df = _make_interval_df(["A001"], start, n_intervals=2)
        inp = _write_parquet(df, tmp_path / "interval.parquet")

        with pytest.raises(FileNotFoundError):
            compute_hourly_loads(
                inp,
                tmp_path / "nonexistent_assignments.parquet",
                tmp_path / "out.parquet",
                sort_output=False,
            )
