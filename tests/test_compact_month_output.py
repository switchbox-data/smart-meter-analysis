#!/usr/bin/env python3
"""Tests for the MAX_ROWS_PER_CHUNK cap in compact_month_output.py.

Covers:
- MAX_ROWS_PER_CHUNK constant value
- rows_per_chunk capped at MAX_ROWS_PER_CHUNK when bytes_per_row is very small
  (e.g. 202508-style dense month with high Parquet compression)
- rows_per_chunk passes through unchanged when rows_per_chunk_raw <= MAX_ROWS_PER_CHUNK
  (normal month)
- compaction_plan.json contains the three new audit keys
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest

# The scripts/ directory is not an installed package; add project root to path
# so pytest can resolve the namespace package on both local and CI runs.
sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.csv_to_parquet.compact_month_output import (
    DEFAULT_COMPACT_TARGET_SIZE_BYTES,
    MAX_ROWS_PER_CHUNK,
    CompactionConfig,
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
    month_dir = tmp_path / "out" / "year=2023" / "month=07"
    _write_batch(_canonical_df(), month_dir)
    return month_dir


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_max_rows_per_chunk_constant() -> None:
    """MAX_ROWS_PER_CHUNK must be exactly 50_000_000."""
    assert MAX_ROWS_PER_CHUNK == 50_000_000


def test_rows_per_chunk_capped_for_dense_month(tmp_path: pytest.TempPathFactory) -> None:
    """When bytes_per_row is tiny (high compression), rows_per_chunk is capped.

    Uses a huge target_size_bytes to force rows_per_chunk_raw >> MAX_ROWS_PER_CHUNK.
    """
    month_dir = _setup_month_dir(tmp_path)
    batch_file = month_dir / "batch_0000.parquet"
    file_bytes = batch_file.stat().st_size  # small: ~1-5 KB
    # bytes_per_row ≈ file_bytes / 20 (n_rows).  Choose target so that
    # rows_per_chunk_raw = round(target / bytes_per_row) >> 50M.
    # bytes_per_row_estimate ≈ file_bytes / 20.
    # We want: target / (file_bytes/20) > 50_000_000
    # → target > 50_000_000 * file_bytes / 20 = 2_500_000 * file_bytes
    target = 2_500_000 * file_bytes * 10  # 25x the threshold: safely above cap

    cfg = _make_cfg(tmp_path, target_size_bytes=target)
    result = run_compaction(cfg, _SilentLogger())

    # dry_run result carries no compacted files, but plan.json is written
    plan_path = tmp_path / "_runs" / YEAR_MONTH / RUN_ID / "compaction" / "compaction_plan.json"
    assert plan_path.exists(), "compaction_plan.json not written"
    plan = json.loads(plan_path.read_text())

    assert plan["max_rows_per_chunk"] == MAX_ROWS_PER_CHUNK
    assert plan["rows_per_chunk_raw"] > MAX_ROWS_PER_CHUNK, (
        f"Expected rows_per_chunk_raw > {MAX_ROWS_PER_CHUNK}, got {plan['rows_per_chunk_raw']}"
    )
    assert plan["rows_per_chunk_capped"] is True
    assert plan["rows_per_chunk"] == MAX_ROWS_PER_CHUNK

    # Returned summary dict must also carry the capped value
    assert result["rows_per_chunk"] == MAX_ROWS_PER_CHUNK


def test_rows_per_chunk_not_capped_for_normal_month(tmp_path: pytest.TempPathFactory) -> None:
    """When bytes_per_row is large (low compression), rows_per_chunk passes through unchanged."""
    _setup_month_dir(tmp_path)
    # DEFAULT_COMPACT_TARGET_SIZE_BYTES (1 GiB) with a multi-KB batch file keeps
    # rows_per_chunk_raw well below 50M because bytes_per_row is large.
    cfg = _make_cfg(tmp_path, target_size_bytes=DEFAULT_COMPACT_TARGET_SIZE_BYTES)
    result = run_compaction(cfg, _SilentLogger())

    plan_path = tmp_path / "_runs" / YEAR_MONTH / RUN_ID / "compaction" / "compaction_plan.json"
    plan = json.loads(plan_path.read_text())

    assert plan["max_rows_per_chunk"] == MAX_ROWS_PER_CHUNK
    assert plan["rows_per_chunk_raw"] <= MAX_ROWS_PER_CHUNK, (
        f"Expected rows_per_chunk_raw <= {MAX_ROWS_PER_CHUNK}, got {plan['rows_per_chunk_raw']}"
    )
    assert plan["rows_per_chunk_capped"] is False
    # rows_per_chunk should equal rows_per_chunk_raw when not capped
    assert plan["rows_per_chunk"] == plan["rows_per_chunk_raw"]

    assert result["rows_per_chunk"] == plan["rows_per_chunk"]


def test_plan_contains_all_new_keys(tmp_path: pytest.TempPathFactory) -> None:
    """compaction_plan.json must contain all three new audit keys."""
    _setup_month_dir(tmp_path)
    cfg = _make_cfg(tmp_path, target_size_bytes=DEFAULT_COMPACT_TARGET_SIZE_BYTES)
    run_compaction(cfg, _SilentLogger())

    plan_path = tmp_path / "_runs" / YEAR_MONTH / RUN_ID / "compaction" / "compaction_plan.json"
    plan = json.loads(plan_path.read_text())

    for key in ("rows_per_chunk_raw", "rows_per_chunk_capped", "max_rows_per_chunk"):
        assert key in plan, f"Missing key '{key}' in compaction_plan.json"
