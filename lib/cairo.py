"""CAIRO post-processing: add delivered fuel bills to combined bill data."""

from __future__ import annotations

import polars as pl

_KWH_PER_GAL_HEATING_OIL = 40.6
_KWH_PER_GAL_PROPANE = 26.8
_OIL_CONSUMPTION_COL = "out.fuel_oil.total.energy_consumption"
_PROPANE_CONSUMPTION_COL = "out.propane.total.energy_consumption"
_MONTH_INT_TO_STR: dict[int, str] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def add_delivered_fuel_bills(
    comb_bills: pl.LazyFrame,
    load_curve_monthly: pl.LazyFrame,
    monthly_prices: pl.DataFrame,
) -> pl.LazyFrame:
    """Top up combined bills with oil/propane costs from monthly consumption x EIA prices."""
    fuel_with_prices = (
        load_curve_monthly.select(
            pl.col("bldg_id"),
            pl.col("month"),
            pl.col(_OIL_CONSUMPTION_COL).fill_null(0),
            pl.col(_PROPANE_CONSUMPTION_COL).fill_null(0),
        )
        .join(monthly_prices.lazy(), on="month", how="left")
        .with_columns(
            (pl.col(_OIL_CONSUMPTION_COL) / _KWH_PER_GAL_HEATING_OIL * pl.col("oil_price_per_gallon")).alias(
                "oil_bill"
            ),
            (pl.col(_PROPANE_CONSUMPTION_COL) / _KWH_PER_GAL_PROPANE * pl.col("propane_price_per_gallon")).alias(
                "propane_bill"
            ),
        )
        .with_columns((pl.col("oil_bill") + pl.col("propane_bill")).alias("delivered_fuel_bill"))
    )
    fuel_monthly = fuel_with_prices.select(
        pl.col("bldg_id"),
        pl.col("month").replace_strict(_MONTH_INT_TO_STR, return_dtype=pl.String),
        pl.col("delivered_fuel_bill"),
    )
    fuel_annual = (
        fuel_monthly.group_by("bldg_id")
        .agg(pl.col("delivered_fuel_bill").sum())
        .with_columns(pl.lit("Annual").alias("month"))
        .select("bldg_id", "month", "delivered_fuel_bill")
    )
    all_fuel = pl.concat([fuel_monthly, fuel_annual])
    combined = comb_bills.join(all_fuel, on=["bldg_id", "month"], how="left")
    return combined.with_columns((pl.col("bill_level") + pl.col("delivered_fuel_bill")).alias("bill_level")).drop(
        "delivered_fuel_bill"
    )
