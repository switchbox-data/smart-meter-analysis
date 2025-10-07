# smart_meter_analysis/step0_transform.py
"""
Step 0: Transform ComEd wide-format CSVs to long format with timestamps.
Handles DST transitions and adds time features.
"""

from __future__ import annotations

import polars as pl

# Column configurations
STEP0_ID_COLS = [
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

INTERVAL_PREFIXES = ("INTERVAL_HR", "HR")

# Error messages
ERR_NO_INTERVAL_COLS = "No interval columns found. Looked for prefixes: {}"


def detect_interval_columns(df: pl.DataFrame) -> list[str]:
    """Return wide interval columns like INTERVAL_HR0000, HR0030, etc."""
    cols: list[str] = []
    for c in df.columns:
        for p in INTERVAL_PREFIXES:
            if c.startswith(p):
                cols.append(c)
                break
    if not cols:
        raise ValueError(ERR_NO_INTERVAL_COLS.format(INTERVAL_PREFIXES))
    return sorted(cols)


def transform_wide_to_long(
    df_raw: pl.DataFrame,
    date_col: str = "INTERVAL_READING_DATE",
    id_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Melt wide ComEd interval file to long format with proper timestamps.
    Handles HR2400/HR2430 (next-day midnight/00:30 roll).

    Returns: DataFrame with columns [zip_code, account_identifier, datetime, kwh]
    """
    if id_cols is None:
        id_cols = STEP0_ID_COLS

    keep_ids = [c for c in id_cols if c in df_raw.columns]
    interval_cols = detect_interval_columns(df_raw)

    long_df = (
        df_raw.select(keep_ids + interval_cols)
        .unpivot(
            index=keep_ids,
            on=interval_cols,
            variable_name="interval_col",
            value_name="kwh",
        )
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
        .sort(["ACCOUNT_IDENTIFIER", "datetime"] if "ACCOUNT_IDENTIFIER" in df_raw.columns else ["datetime"])
        .select([
            pl.col("ZIP_CODE").alias("zip_code") if "ZIP_CODE" in df_raw.columns else pl.lit(None).alias("zip_code"),
            pl.col("DELIVERY_SERVICE_CLASS").alias("delivery_service_class")
            if "DELIVERY_SERVICE_CLASS" in df_raw.columns
            else pl.lit(None).alias("delivery_service_class"),
            pl.col("DELIVERY_SERVICE_NAME").alias("delivery_service_name")
            if "DELIVERY_SERVICE_NAME" in df_raw.columns
            else pl.lit(None).alias("delivery_service_name"),
            pl.col("ACCOUNT_IDENTIFIER").alias("account_identifier")
            if "ACCOUNT_IDENTIFIER" in df_raw.columns
            else pl.lit(None).alias("account_identifier"),
            pl.col("datetime"),
            pl.col("kwh").cast(pl.Float64),
        ])
    )
    return long_df


def add_time_columns(df_long: pl.DataFrame) -> pl.DataFrame:
    """Add date/hour/weekday/is_weekend columns to long-format data."""
    return df_long.with_columns([
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        (pl.col("datetime").dt.weekday() >= 5).alias("is_weekend"),
    ])


def daily_interval_qc(df_long: pl.DataFrame) -> pl.DataFrame:
    """
    QC: Count intervals per account/day and flag DST transitions.

    UPDATED: Only counts non-null kWh values to properly identify normal (48) vs DST days.

    Returns: DataFrame with day_type ('normal', 'spring_forward', 'fall_back', 'odd')
    """
    df = df_long.with_columns(pl.col("datetime").dt.date().alias("date"))
    return (
        df.group_by(["account_identifier", "date"])
        .agg([
            pl.len().alias("total_intervals"),
            pl.col("kwh").filter(pl.col("kwh").is_not_null()).len().alias("n_intervals"),
            pl.col("kwh").null_count().alias("null_intervals"),
            pl.col("kwh").sum().alias("sum_kwh"),
        ])
        .with_columns([
            pl.when(pl.col("n_intervals") == 46)
            .then(pl.lit("spring_forward"))
            .when(pl.col("n_intervals") == 50)
            .then(pl.lit("fall_back"))
            .when(pl.col("n_intervals") == 48)
            .then(pl.lit("normal"))
            .otherwise(pl.lit("odd"))
            .alias("day_type"),
            pl.col("n_intervals").is_in([46, 50]).alias("is_dst_transition"),
            (~pl.col("n_intervals").is_in([46, 48, 50])).alias("is_odd_count"),
        ])
    )


def dst_transition_dates(df_long: pl.DataFrame) -> pl.DataFrame:
    """List unique DST transition dates in the data."""
    qc = daily_interval_qc(df_long)
    return qc.filter(pl.col("is_dst_transition")).select(["date", "day_type"]).unique().sort("date")
