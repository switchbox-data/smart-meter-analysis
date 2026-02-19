from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from smart_meter_analysis.wide_to_long import transform_wide_to_long_lf


def _standard_interval_cols() -> list[str]:
    cols: list[str] = []
    # 0030..2400 inclusive, step 30 minutes
    for end_minutes in range(30, 1441, 30):
        hh = end_minutes // 60
        mm = end_minutes % 60
        hhmm = f"{hh:02d}{mm:02d}"
        cols.append(f"INTERVAL_HR{hhmm}_ENERGY_QTY")
    return cols


def _add_filler_columns_to_reach_59(df: pl.DataFrame) -> pl.DataFrame:
    """Add deterministic non-contract filler columns so total wide schema is exactly 59 columns.

    Important: Filler columns MUST NOT match the interval column regex and MUST NOT overlap required columns.
    """
    target = 59
    current = len(df.columns)
    if current > target:
        raise ValueError(f"Test fixture bug: wide df already has {current} columns (>59).")

    needed = target - current
    if needed == 0:
        return df

    existing = set(df.columns)
    # Find the next filler index after any existing FILLER_XX columns to avoid overwrites.
    max_idx = 0
    for c in existing:
        if c.startswith("FILLER_"):
            try:
                max_idx = max(max_idx, int(c.split("_", 1)[1]))
            except ValueError:
                continue

    added = 0
    i = max_idx + 1
    while added < needed:
        name = f"FILLER_{i:02d}"
        if name not in existing:
            df = df.with_columns(pl.lit(f"filler_{i:02d}").alias(name))
            existing.add(name)
            added += 1
        i += 1

    assert len(df.columns) == 59
    return df


def _minimal_wide_lf(n: int = 3, *, with_dst_extras: bool) -> pl.LazyFrame:
    reading_date = "07/01/2023"

    base = pl.DataFrame({
        "ZIP_CODE": ["60601"] * n,
        "DELIVERY_SERVICE_CLASS": ["RES"] * n,
        "DELIVERY_SERVICE_NAME": ["COMED"] * n,
        "ACCOUNT_IDENTIFIER": [f"acct_{i:03d}" for i in range(n)],
        "INTERVAL_READING_DATE": [reading_date] * n,
        # Keep as integer dtype to satisfy strict INTERVAL_LENGTH contract.
        "INTERVAL_LENGTH": [1800] * n,
        "PLC_VALUE": [1.0] * n,
        "NSPL_VALUE": [2.0] * n,
    })

    # Standard 48 interval columns: default 0.0
    for c in _standard_interval_cols():
        base = base.with_columns(pl.lit(0.0).alias(c))

    if with_dst_extras:
        # Provide extras (2430/2500) and set specific values for fold-in testing.
        base = base.with_columns([
            pl.lit(2.0).alias("INTERVAL_HR2430_ENERGY_QTY"),
            pl.lit(4.0).alias("INTERVAL_HR2500_ENERGY_QTY"),
        ])
        # Set base values for 2330/2400 so we can assert sums:
        base = base.with_columns([
            pl.lit(1.0).alias("INTERVAL_HR2330_ENERGY_QTY"),
            pl.lit(3.0).alias("INTERVAL_HR2400_ENERGY_QTY"),
        ])

    # Enforce exact 59-column wide schema required by strict mode.
    base = _add_filler_columns_to_reach_59(base)
    return base.lazy()


def _assert_schema_contract(out: pl.DataFrame) -> None:
    assert out.columns == [
        "zip_code",
        "delivery_service_class",
        "delivery_service_name",
        "account_identifier",
        "datetime",
        "energy_kwh",
        "plc_value",
        "nspl_value",
        "year",
        "month",
    ]

    sch = out.schema
    assert sch["zip_code"] == pl.Utf8
    assert sch["delivery_service_class"] == pl.Categorical
    assert sch["delivery_service_name"] == pl.Categorical
    assert sch["account_identifier"] == pl.Utf8
    assert sch["datetime"] == pl.Datetime("us")
    assert sch["energy_kwh"] == pl.Float64
    assert sch["plc_value"] == pl.Float64
    assert sch["nspl_value"] == pl.Float64
    assert sch["year"] == pl.Int32
    assert sch["month"] == pl.Int8


def test_rows_out_equals_rows_wide_times_48() -> None:
    lf = _minimal_wide_lf(n=3, with_dst_extras=True)
    out = transform_wide_to_long_lf(lf, strict=True).collect()
    assert out.height == 3 * 48


def test_dst_option_b_foldin_and_no_extras_representation() -> None:
    lf = _minimal_wide_lf(n=1, with_dst_extras=True)
    out = transform_wide_to_long_lf(lf, strict=True).collect()

    # Strong invariant: max datetime is 23:30 and there are exactly 48 unique datetimes for the day.
    assert out.select(pl.col("datetime").max()).item() == dt.datetime(2023, 7, 1, 23, 30)
    assert out.select(pl.col("datetime").n_unique()).item() == 48

    # Fold-in checks:
    # HR2330 label corresponds to interval start at 23:00
    v_2300 = out.filter(pl.col("datetime") == dt.datetime(2023, 7, 1, 23, 0)).select("energy_kwh").item()
    assert v_2300 == pytest.approx(3.0)

    # HR2400 label corresponds to interval start at 23:30
    v_2330 = out.filter(pl.col("datetime") == dt.datetime(2023, 7, 1, 23, 30)).select("energy_kwh").item()
    assert v_2330 == pytest.approx(7.0)


def test_output_schema_and_dtypes() -> None:
    lf = _minimal_wide_lf(n=2, with_dst_extras=False)
    out = transform_wide_to_long_lf(lf, strict=True).collect()
    _assert_schema_contract(out)


def test_deterministic_sort_order_default_true() -> None:
    # Default behavior must remain deterministic (sorted).
    lf = _minimal_wide_lf(n=5, with_dst_extras=False)
    out = transform_wide_to_long_lf(lf, strict=True).collect()
    out_sorted = out.sort(["zip_code", "account_identifier", "datetime"])
    assert out.rows() == out_sorted.rows()


def test_sort_output_false_preserves_semantics_and_schema() -> None:
    # No-sort mode is intended for month-scale validation in constrained environments.
    lf = _minimal_wide_lf(n=4, with_dst_extras=True)
    out = transform_wide_to_long_lf(lf, strict=True, sort_output=False).collect()

    # Schema/dtypes remain contractually identical.
    _assert_schema_contract(out)

    # Row multiplier holds.
    assert out.height == 4 * 48

    # Datetime invariants: max is 23:30, no nulls, 48 unique per day (since all rows are same day).
    assert out.select(pl.col("datetime").max()).item() == dt.datetime(2023, 7, 1, 23, 30)
    assert out.select(pl.col("datetime").is_null().any()).item() is False
    assert out.select(pl.col("datetime").n_unique()).item() == 48

    # IMPORTANT: do not assert ordering in no-sort mode.


def test_strict_mode_rejects_bad_interval_length() -> None:
    df = _minimal_wide_lf(n=1, with_dst_extras=False).collect()

    # Keep total columns == 59 by replacing the existing column (no add/drop).
    df = df.with_columns(pl.lit(3600).cast(pl.Int32).alias("INTERVAL_LENGTH"))
    with pytest.raises(ValueError, match="INTERVAL_LENGTH"):
        transform_wide_to_long_lf(df.lazy(), strict=True).collect()


def test_strict_mode_rejects_bad_date_format() -> None:
    df = _minimal_wide_lf(n=1, with_dst_extras=False).collect()
    # Replace only; keep 59 columns invariant.
    df = df.with_columns(pl.lit("2023-07-01").alias("INTERVAL_READING_DATE"))
    with pytest.raises(ValueError, match="Failed to parse INTERVAL_READING_DATE"):
        transform_wide_to_long_lf(df.lazy(), strict=True).collect()


def test_strict_mode_rejects_missing_standard_interval() -> None:
    df = _minimal_wide_lf(n=1, with_dst_extras=False).collect()

    # Dropping a required standard interval reduces column count.
    # To keep 59 columns invariant, drop one filler column as well and then add back a filler column.
    df = df.drop(["INTERVAL_HR0030_ENERGY_QTY", "FILLER_01"])
    df = _add_filler_columns_to_reach_59(df)

    with pytest.raises(ValueError, match="missing standard"):
        transform_wide_to_long_lf(df.lazy(), strict=True).collect()


def test_strict_mode_rejects_unexpected_extra_interval() -> None:
    df = _minimal_wide_lf(n=1, with_dst_extras=False).collect()

    # Adding a bogus interval would make 60 columns; keep invariant by dropping a filler column first.
    df = df.drop(["FILLER_01"])
    df = df.with_columns(pl.lit(0.0).alias("INTERVAL_HR2415_ENERGY_QTY"))
    assert len(df.columns) == 59

    with pytest.raises(ValueError, match="invalid interval column minutes"):
        transform_wide_to_long_lf(df.lazy(), strict=True).collect()


def test_forbid_0000_interval_label() -> None:
    df = _minimal_wide_lf(n=1, with_dst_extras=False).collect()

    # Adding 0000 interval would make 60 columns; keep invariant by dropping a filler column first.
    df = df.drop(["FILLER_01"])
    df = df.with_columns(pl.lit(0.0).alias("INTERVAL_HR0000_ENERGY_QTY"))
    assert len(df.columns) == 59

    with pytest.raises(ValueError, match="ending at 0000"):
        transform_wide_to_long_lf(df.lazy(), strict=True).collect()
