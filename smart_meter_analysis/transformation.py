# smart_meter_analysis/transformation.py
"""
Data transformation utilities for ComEd smart meter data.

This module supports both eager (local) and lazy (S3) processing paths.
It converts wide-format ComEd CSVs into a normalized long format,
adds timestamps, and applies optional day-attribution modes.

Day attribution modes:
- "calendar": 00:00 readings belong to the new day (default)
- "billing":  00:00 readings belong to the previous day
"""

from __future__ import annotations

from datetime import date as _date

import polars as pl

__all__ = [
    "add_time_columns",
    "transform_wide_to_long",
]

# Interval columns pattern — same as in aws_loader
COMED_INTERVAL_COLUMNS = [f"INTERVAL_HR{m // 60:02d}{m % 60:02d}_ENERGY_QTY" for m in range(30, 24 * 60 + 1, 30)]


def transform_wide_to_long(
    df: pl.DataFrame,
    date_col: str = "INTERVAL_READING_DATE",
) -> pl.DataFrame:
    """
    Convert wide-format ComEd data into a long-format DataFrame.

    Each row represents a 30-minute interval with a timestamp and energy usage.
    """
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

    requested_cols = [c for c in id_cols + COMED_INTERVAL_COLUMNS if c in df.columns]

    df_long = (
        df.select(requested_cols)
        .unpivot(
            index=id_cols,
            on=[c for c in COMED_INTERVAL_COLUMNS if c in df.columns],
            variable_name="interval_col",
            value_name="kwh",
        )
        .filter(pl.col("kwh").is_not_null())
        .with_columns(pl.col("interval_col").str.extract(r"HR(\d{4})", 1).alias("time_str"))
        .with_columns([
            pl.col(date_col).str.strptime(pl.Date, format="%m/%d/%Y", strict=False).alias("service_date"),
            pl.col("time_str").str.slice(0, 2).cast(pl.Int16).alias("hour_raw"),
            pl.col("time_str").str.slice(2, 2).cast(pl.Int16).alias("minute"),
        ])
        .with_columns([
            (pl.col("hour_raw") // 24).alias("days_offset"),
            (pl.col("hour_raw") % 24).alias("hour"),
        ])
        .with_columns([
            (
                pl.col("service_date").cast(pl.Datetime)
                + pl.duration(days=pl.col("days_offset"), hours=pl.col("hour"), minutes=pl.col("minute"))
            ).alias("datetime")
        ])
        .select([
            pl.col("ZIP_CODE").alias("zip_code"),
            pl.col("DELIVERY_SERVICE_CLASS").alias("delivery_service_class"),
            pl.col("DELIVERY_SERVICE_NAME").alias("delivery_service_name"),
            pl.col("ACCOUNT_IDENTIFIER").alias("account_identifier"),
            pl.col("datetime"),
            pl.col("kwh").cast(pl.Float64),
        ])
    )

    return df_long


def add_time_columns(df: pl.DataFrame, day_mode: str = "calendar") -> pl.DataFrame:
    """
    Add derived time columns and day-attribution flags.

    Args:
        df: Polars DataFrame with a 'datetime' column.
        day_mode: 'calendar' (default) or 'billing'
            - "calendar": 00:00 belongs to the new day.
            - "billing":  00:00 readings assigned to the previous day.
    """
    if day_mode not in {"calendar", "billing"}:
        raise ValueError("day_mode must be 'calendar' or 'billing'")

    DST_SPRING_2023 = _date(2023, 3, 12)
    DST_FALL_2023 = _date(2023, 11, 5)

    dt = pl.col("datetime")

    if day_mode == "calendar":
        date_expr = dt.dt.date()
    else:
        # Assign midnight (00:00) readings to previous date
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
            (pl.col("date") == DST_SPRING_2023).alias("is_spring_forward_day"),
            (pl.col("date") == DST_FALL_2023).alias("is_fall_back_day"),
            ((pl.col("date") == DST_SPRING_2023) | (pl.col("date") == DST_FALL_2023)).alias("is_dst_day"),
        ])
    )

    return df
