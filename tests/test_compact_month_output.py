#!/usr/bin/env python3
"""Tests for _stream_write_chunks and related constants in compact_month_output.py.

Covers:
- ROWS_PER_ROW_GROUP constant value
- compaction_plan.json contains rows_per_row_group and estimated_n_output_files keys
- _stream_write_chunks: single small batch -> one file
- _stream_write_chunks: multi-row-group packing into a single file
- _stream_write_chunks: file rollover when target_size_bytes is reached
- _stream_write_chunks: total row count preserved across all output files
- _stream_write_chunks: sort order within a single output file
- _stream_write_chunks: sort order preserved across rolled-over files
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import pytest

# The scripts/ directory is not an installed package; add project root to path
# so pytest can resolve the namespace package on both local and CI runs.
sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.csv_to_parquet.compact_month_output import (
    DEFAULT_COMPACT_TARGET_SIZE_BYTES,
    ROWS_PER_ROW_GROUP,
    SORT_KEYS,
    CompactionConfig,
    _stream_write_chunks,
    run_compaction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

YEAR_MONTH = "202307"
RUN_ID = "test_run_001"


def _canonical_df(n_rows: int = 20) -> pl.DataFrame:
    """Return a minimal DataFrame matching the canonical 10-column schema."""
    base_ts = dt.datetime(2023, 7, 1, 0, 0, 0)
    return pl.DataFrame({
        "zip_code": ["60601"] * n_rows,
        "delivery_service_class": ["DS1"] * n_rows,
        "delivery_service_name": ["Residential"] * n_rows,
        "account_identifier": [f"ACCT{i:04d}" for i in range(n_rows)],
        "datetime": [base_ts + dt.timedelta(hours=i) for i in range(n_rows)],
        "energy_kwh": [float(i) * 0.5 for i in range(n_rows)],
        "plc_value": [0.0] * n_rows,
        "nspl_value": [0.0] * n_rows,
        "year": [2023] * n_rows,
        "month": [7] * n_rows,
    }).with_columns(
        pl.col("zip_code").cast(pl.Utf8),
        pl.col("delivery_service_class").cast(pl.Categorical),
        pl.col("delivery_service_name").cast(pl.Categorical),
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("datetime").cast(pl.Datetime("us")),
        pl.col("energy_kwh").cast(pl.Float64),
        pl.col("plc_value").cast(pl.Float64),
        pl.col("nspl_value").cast(pl.Float64),
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int8),
    )


def _write_batch(df: pl.DataFrame, month_dir: Path, name: str = "batch_0000.parquet") -> Path:
    month_dir.mkdir(parents=True, exist_ok=True)
    path = month_dir / name
    df.write_parquet(path)
    return path


def _make_cfg(tmp_path: Path, target_size_bytes: int, dry_run: bool = True) -> CompactionConfig:
    return CompactionConfig(
        year_month=YEAR_MONTH,
        run_id=RUN_ID,
        out_root=tmp_path / "out",
        run_dir=tmp_path / "_runs" / YEAR_MONTH / RUN_ID,
        target_size_bytes=target_size_bytes,
        max_files=None,
        overwrite=False,
        dry_run=dry_run,
        no_swap=False,
    )


class _SilentLogger:
    """No-op logger compatible with run_compaction's logger.log(dict) protocol."""

    def log(self, event: dict[str, Any]) -> None:
        pass


def _setup_month_dir(tmp_path: Path) -> Path:
    """Write one batch_0000.parquet to the canonical month directory."""
    month_dir = tmp_path / "out" / "2023" / "07"
    _write_batch(_canonical_df(), month_dir)
    return month_dir


# ---------------------------------------------------------------------------
# Unit tests — constants and plan keys
# ---------------------------------------------------------------------------


def test_rows_per_row_group_constant() -> None:
    """ROWS_PER_ROW_GROUP must be exactly 50_000_000."""
    assert ROWS_PER_ROW_GROUP == 50_000_000


def test_plan_contains_all_new_keys(tmp_path: pytest.TempPathFactory) -> None:
    """compaction_plan.json must contain rows_per_row_group and estimated_n_output_files."""
    _setup_month_dir(tmp_path)
    cfg = _make_cfg(tmp_path, target_size_bytes=DEFAULT_COMPACT_TARGET_SIZE_BYTES)
    run_compaction(cfg, _SilentLogger())

    plan_path = tmp_path / "_runs" / YEAR_MONTH / RUN_ID / "compaction" / "compaction_plan.json"
    plan = json.loads(plan_path.read_text())

    assert "rows_per_row_group" in plan, "Missing key 'rows_per_row_group' in compaction_plan.json"
    assert "estimated_n_output_files" in plan, "Missing key 'estimated_n_output_files' in compaction_plan.json"
    assert plan["rows_per_row_group"] == ROWS_PER_ROW_GROUP
    assert plan["estimated_n_output_files"] >= 1


# ---------------------------------------------------------------------------
# Unit tests — _stream_write_chunks directly
# ---------------------------------------------------------------------------


def test_single_small_batch_one_file(tmp_path: Path) -> None:
    """5 rows, rows_per_row_group=10, huge target -> 1 output file with 1 row group."""
    staging = tmp_path / "staging"
    batch_path = _write_batch(_canonical_df(5), tmp_path / "input", "batch_0000.parquet")

    output_files = _stream_write_chunks(
        sorted_input_files=[batch_path],
        staging_month_dir=staging,
        rows_per_row_group=10,
        target_size_bytes=10 * 1024**3,  # 10 GiB — never triggers rollover
        max_files=None,
        logger=_SilentLogger(),
        log_ctx={},
    )

    assert len(output_files) == 1
    meta = pq.read_metadata(str(output_files[0]))
    assert meta.num_row_groups == 1
    assert meta.num_rows == 5


def test_multi_row_group_single_file(tmp_path: Path) -> None:
    """25 rows, rows_per_row_group=10, huge target -> 1 file with 3 row groups (10+10+5)."""
    staging = tmp_path / "staging"
    batch_path = _write_batch(_canonical_df(25), tmp_path / "input", "batch_0000.parquet")

    output_files = _stream_write_chunks(
        sorted_input_files=[batch_path],
        staging_month_dir=staging,
        rows_per_row_group=10,
        target_size_bytes=10 * 1024**3,  # 10 GiB — never triggers rollover
        max_files=None,
        logger=_SilentLogger(),
        log_ctx={},
    )

    assert len(output_files) == 1
    meta = pq.read_metadata(str(output_files[0]))
    assert meta.num_row_groups == 3
    assert meta.num_rows == 25


def test_file_rollover_at_target_size(tmp_path: Path) -> None:
    """30 rows, rows_per_row_group=10, target_size_bytes=1 -> rollover after each rg -> 3 files."""
    staging = tmp_path / "staging"
    batch_path = _write_batch(_canonical_df(30), tmp_path / "input", "batch_0000.parquet")

    output_files = _stream_write_chunks(
        sorted_input_files=[batch_path],
        staging_month_dir=staging,
        rows_per_row_group=10,
        target_size_bytes=1,  # always triggers rollover
        max_files=None,
        logger=_SilentLogger(),
        log_ctx={},
    )

    assert len(output_files) == 3
    for f in output_files:
        meta = pq.read_metadata(str(f))
        assert meta.num_row_groups == 1
        assert meta.num_rows == 10


def test_row_count_preserved(tmp_path: Path) -> None:
    """2 batch files of 15 rows each -> total 30 rows across all output files."""
    staging = tmp_path / "staging"
    input_dir = tmp_path / "input"
    b0 = _write_batch(_canonical_df(15), input_dir, "batch_0000.parquet")
    # Second batch: distinct account_identifiers to avoid duplicate key issues
    df2 = _canonical_df(15).with_columns(pl.Series("account_identifier", [f"ACCT{i:04d}" for i in range(15, 30)]))
    b1 = _write_batch(df2, input_dir, "batch_0001.parquet")

    output_files = _stream_write_chunks(
        sorted_input_files=[b0, b1],
        staging_month_dir=staging,
        rows_per_row_group=10,
        target_size_bytes=10 * 1024**3,
        max_files=None,
        logger=_SilentLogger(),
        log_ctx={},
    )

    total = sum(pq.read_metadata(str(f)).num_rows for f in output_files)
    assert total == 30


def test_sort_order_within_file(tmp_path: Path) -> None:
    """Rows read back from a single output file are sorted by SORT_KEYS."""
    staging = tmp_path / "staging"
    input_dir = tmp_path / "input"
    b0 = _write_batch(_canonical_df(15), input_dir, "batch_0000.parquet")
    df2 = _canonical_df(15).with_columns(pl.Series("account_identifier", [f"ACCT{i:04d}" for i in range(15, 30)]))
    b1 = _write_batch(df2, input_dir, "batch_0001.parquet")

    output_files = _stream_write_chunks(
        sorted_input_files=[b0, b1],
        staging_month_dir=staging,
        rows_per_row_group=10,
        target_size_bytes=10 * 1024**3,
        max_files=None,
        logger=_SilentLogger(),
        log_ctx={},
    )

    for f in output_files:
        df = pl.read_parquet(str(f))
        sort_keys = list(SORT_KEYS)
        sorted_df = df.sort(sort_keys)
        assert df.select(sort_keys).equals(sorted_df.select(sort_keys)), (
            f"Rows in {f.name} are not sorted by {sort_keys}"
        )


def test_sort_order_across_files(tmp_path: Path) -> None:
    """Last sort key of file N < first sort key of file N+1 when rollover occurs."""
    staging = tmp_path / "staging"
    input_dir = tmp_path / "input"
    b0 = _write_batch(_canonical_df(15), input_dir, "batch_0000.parquet")
    df2 = _canonical_df(15).with_columns(pl.Series("account_identifier", [f"ACCT{i:04d}" for i in range(15, 30)]))
    b1 = _write_batch(df2, input_dir, "batch_0001.parquet")

    output_files = _stream_write_chunks(
        sorted_input_files=[b0, b1],
        staging_month_dir=staging,
        rows_per_row_group=10,
        target_size_bytes=1,  # rollover after each row group -> multiple files
        max_files=None,
        logger=_SilentLogger(),
        log_ctx={},
    )

    assert len(output_files) > 1, "Expected multiple output files with target_size_bytes=1"
    sort_keys = list(SORT_KEYS)
    prev_last: tuple[Any, ...] | None = None
    for f in output_files:
        df = pl.read_parquet(str(f))
        first = tuple(df.select(sort_keys).row(0))
        last = tuple(df.select(sort_keys).row(-1))
        if prev_last is not None:
            assert prev_last <= first, (
                f"Sort order broken across files: last of prev={prev_last} >= first of next={first}"
            )
        prev_last = last
