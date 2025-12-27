# smart_meter_analysis/transformation.py
"""Data transformation utilities for ComEd smart meter data.

Supports both eager (DataFrame) and lazy (LazyFrame) processing paths.

Core behaviors:
- Converts wide-format ComEd interval columns into long format.
- Builds a timestamp per interval using the service date + interval "hour/minute" encoding.
- Preserves ComEd DST anomalies: some days include 24:30 and 25:00 columns (2430, 2500).
- First interval column is expected to be 00:30 (0030), not 00:00.

Day attribution modes:
- "calendar": 00:00 readings belong to the new day (default)
- "billing":  00:00 readings belong to the previous day
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as _date

import polars as pl

__all__ = [
    "COMED_INTERVAL_COLUMNS",
    "add_time_columns",
    "transform_wide_to_long",
    "transform_wide_to_long_lf",
]

# Canonical interval columns:
# - Standard: 0030..2400 (48 columns)
# - DST fall-back may add: 2430, 2500
# We include the DST extensions explicitly; for non-DST days these columns are simply absent.
COMED_INTERVAL_COLUMNS: list[str] = [
    f"INTERVAL_HR{m // 60:02d}{m % 60:02d}_ENERGY_QTY" for m in range(30, 24 * 60 + 1, 30)
] + [
    "INTERVAL_HR2430_ENERGY_QTY",
    "INTERVAL_HR2500_ENERGY_QTY",
]


def _present_interval_cols(cols: Sequence[str]) -> list[str]:
    return [c for c in COMED_INTERVAL_COLUMNS if c in cols]


def _present_id_cols(cols: Sequence[str]) -> list[str]:
    id_cols = [
        "ZIP_CODE",
        "DELIVERY_SERVICE_CLASS",
        "DELIVERY_SERVICE_NAME",
        "ACCOUNT_IDENTIFIER",
        "INTERVAL_READING_DATE",
        "INTERVAL_LENGTH",
        "TOTAL_REGISTERED_ENERGY",
        "PLC_VALUE",
        "NSPL_VALUE",
    ]
    return [c for c in id_cols if c in cols]


def _validate_required_cols(cols: Sequence[str], *, date_col: str) -> None:
    required = ["ZIP_CODE", "ACCOUNT_IDENTIFIER", date_col]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns for transform_wide_to_long: {missing}")


def _service_date_expr(date_col: str) -> pl.Expr:
    """Normalize INTERVAL_READING_DATE to pl.Date.

    Handles common cases:
    - Utf8 like '07/31/2023'
    - Date (already correct type)
    - Datetime (cast to date)

    Strategy: Try string parsing first, fall back to direct casting.
    Works in both eager and lazy contexts.
    """
    col = pl.col(date_col)

    # Try parsing as string first (most common case for ComEd CSVs)
    # If it fails, the column is likely already Date or Datetime
    return (
        col.str.strptime(pl.Date, format="%m/%d/%Y", strict=False)
        .fill_null(col.cast(pl.Date, strict=False))
        .alias("service_date")
    )


def transform_wide_to_long(df: pl.DataFrame, date_col: str = "INTERVAL_READING_DATE") -> pl.DataFrame:
    """Eager wrapper for converting wide-format ComEd data into long format.

    Returns one row per account x interval with:
        zip_code, delivery_service_class, delivery_service_name,
        account_identifier, datetime, kwh
    """
    return transform_wide_to_long_lf(df.lazy(), date_col=date_col).collect()


def transform_wide_to_long_lf(lf: pl.LazyFrame, date_col: str = "INTERVAL_READING_DATE") -> pl.LazyFrame:
    """Convert wide-format ComEd data into long format (lazy).

    Notes:
    - Interval columns are detected from the input schema and may include DST extensions
      (INTERVAL_HR2430_ENERGY_QTY, INTERVAL_HR2500_ENERGY_QTY).
    - First interval is expected to be 0030.

    """
    schema = lf.collect_schema()
    cols = schema.names()

    _validate_required_cols(cols, date_col=date_col)

    id_cols = _present_id_cols(cols)
    interval_cols = _present_interval_cols(cols)
    if not interval_cols:
        raise ValueError("No ComEd interval columns found (expected INTERVAL_HR####_ENERGY_QTY).")

    # Step 1: Unpivot to long format
    out = (
        lf.select(id_cols + interval_cols)
        .unpivot(
            index=id_cols,
            on=interval_cols,
            variable_name="interval_col",
            value_name="kwh",
        )
        .filter(pl.col("kwh").is_not_null())
    )

    # Step 2: Add service_date (parsed from date column)
    out = out.with_columns([_service_date_expr(date_col).alias("service_date")])

    # Step 3: Extract time components from interval column name
    out = out.with_columns([pl.col("interval_col").str.extract(r"HR(\d{4})", 1).alias("time_str")])

    # Step 4: Parse hour and minute from time_str
    out = out.with_columns([
        pl.col("time_str").str.slice(0, 2).cast(pl.Int16).alias("hour_raw"),
        pl.col("time_str").str.slice(2, 2).cast(pl.Int16).alias("minute"),
    ])

    # Step 5: Handle DST extensions (hours 24, 25 -> days_offset)
    out = out.with_columns([
        (pl.col("hour_raw") // 24).alias("days_offset"),
        (pl.col("hour_raw") % 24).alias("hour"),
    ])

    # Step 6: Build datetime
    out = out.with_columns([
        (
            pl.col("service_date").cast(pl.Datetime)
            + pl.duration(days=pl.col("days_offset"), hours=pl.col("hour"), minutes=pl.col("minute"))
        ).alias("datetime"),
    ])

    # Step 7: Select final columns with proper names
    out = out.select([
        pl.col("ZIP_CODE").alias("zip_code"),
        pl.col("DELIVERY_SERVICE_CLASS").alias("delivery_service_class"),
        pl.col("DELIVERY_SERVICE_NAME").alias("delivery_service_name"),
        pl.col("ACCOUNT_IDENTIFIER").alias("account_identifier"),
        pl.col("datetime"),
        pl.col("kwh").cast(pl.Float64),
    ])

    return out


def add_time_columns(df: pl.DataFrame, day_mode: str = "calendar") -> pl.DataFrame:
    """Add derived time columns and day-attribution flags.

    Args:
        df: Polars DataFrame with a 'datetime' column.
        day_mode: 'calendar' (default) or 'billing'
            - "calendar": 00:00 belongs to the new day.
            - "billing":  00:00 readings assigned to the previous date.

    """
    if day_mode not in {"calendar", "billing"}:
        raise ValueError("day_mode must be 'calendar' or 'billing'")

    # 2023-only flags retained for continuity; DST handling is otherwise implicit in interval columns.
    DST_SPRING_2023 = _date(2023, 3, 12)
    DST_FALL_2023 = _date(2023, 11, 5)

    dt = pl.col("datetime")

    if day_mode == "calendar":
        date_expr = dt.dt.date()
    else:
        date_expr = (
            pl.when((dt.dt.hour() == 0) & (dt.dt.minute() == 0))
            .then((dt - pl.duration(days=1)).dt.date())
            .otherwise(dt.dt.date())
        )

    df = (
        df.with_columns([
            date_expr.alias("date"),
            dt.dt.hour().alias("hour"),
        ])
        .with_columns([
            pl.col("date").dt.weekday().alias("weekday"),
            (pl.col("date").dt.weekday() >= 5).alias("is_weekend"),
        ])
        .with_columns([
            (pl.col("date") == DST_SPRING_2023).alias("is_spring_forward_day_2023"),
            (pl.col("date") == DST_FALL_2023).alias("is_fall_back_day_2023"),
            ((pl.col("date") == DST_SPRING_2023) | (pl.col("date") == DST_FALL_2023)).alias("is_dst_day_2023"),
        ])
    )

    return df
