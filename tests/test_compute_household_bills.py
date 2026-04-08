#!/usr/bin/env python3
"""Unit tests for analysis/rtp/compute_household_bills.py.

Tests compute_household_bills() and _join_tariff() directly using
synthetic DataFrames.  No large data files required; runs in milliseconds.

NOTE ON TARIFF FORMAT
---------------------
compute_household_bills() expects tariff DataFrames that have already been
processed by load_tariff_prices(), which renames the price column:
  tariff A → price_A_cents   (column expected by _join_tariff(..., "price_A_cents", "A"))
  tariff B → price_B_cents   (column expected by _join_tariff(..., "price_B_cents", "B"))

The _make_tariff() helper below accepts a label ("A" or "B") to produce the
correctly named column.
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar

import polars as pl
import pytest

from analysis.rtp.compute_household_bills import _join_tariff, compute_household_bills

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_loads(
    accounts: list[str],
    hours: list[dt.datetime],
    kwh_per_hour: float = 1.0,
    zip_code: str = "60601-1234",
) -> pl.DataFrame:
    """Build a minimal hourly loads DataFrame."""
    rows = []
    for acct in accounts:
        for h in hours:
            rows.append({
                "account_identifier": acct,
                "zip_code": zip_code,
                "hour_chicago": h,
                "kwh_hour": kwh_per_hour,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("hour_chicago").cast(pl.Datetime("us")),
        pl.col("kwh_hour").cast(pl.Float64),
    )


def _make_tariff(hours: list[dt.datetime], price: float, label: str) -> pl.DataFrame:
    """Build a flat-price hourly tariff with the pre-renamed price column.

    load_tariff_prices(path, label) renames price_cents_per_kwh → price_{label}_cents.
    compute_household_bills() expects this already-renamed form.
    """
    return pl.DataFrame({
        "datetime_chicago": pl.Series(hours).cast(pl.Datetime("us")),
        f"price_{label}_cents": pl.Series([price] * len(hours), dtype=pl.Float64),
    })


def _hours(start: dt.datetime, n: int) -> list[dt.datetime]:
    return [start + dt.timedelta(hours=i) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Bill arithmetic
# ═══════════════════════════════════════════════════════════════════════════


class TestBillArithmetic:
    """Verify the core dollar math is correct."""

    def test_bill_a_equals_kwh_times_price_a(self) -> None:
        """bill_a_dollars = sum(kwh_hour * price_A_cents) / 100."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=2.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")  # 10 cents/kWh
        tariff_b = _make_tariff(hrs, price=5.0, label="B")  # 5 cents/kWh

        result = compute_household_bills(loads, tariff_a, tariff_b)

        # 24 hours x 2 kWh x 10 c/kWh / 100 = $4.80
        assert result["bill_a_dollars"][0] == pytest.approx(24 * 2.0 * 10.0 / 100.0)

    def test_bill_b_equals_kwh_times_price_b(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=2.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=5.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        assert result["bill_b_dollars"][0] == pytest.approx(24 * 2.0 * 5.0 / 100.0)

    def test_bill_diff_is_a_minus_b(self) -> None:
        """bill_diff_dollars = bill_a - bill_b (positive when B is cheaper)."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=1.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=6.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        bill_a = result["bill_a_dollars"][0]
        bill_b = result["bill_b_dollars"][0]
        expected_diff = bill_a - bill_b
        assert result["bill_diff_dollars"][0] == pytest.approx(expected_diff)
        assert result["bill_diff_dollars"][0] > 0  # B is cheaper

    def test_bill_diff_negative_when_b_more_expensive(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=1.0)
        tariff_a = _make_tariff(hrs, price=5.0, label="A")
        tariff_b = _make_tariff(hrs, price=15.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        assert result["bill_diff_dollars"][0] < 0  # B is more expensive

    def test_total_kwh_correct(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=1.5)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=10.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        assert result["total_kwh"][0] == pytest.approx(24 * 1.5)

    def test_peak_kwh_hour_is_max_hourly(self) -> None:
        """peak_kwh_hour should be the maximum single hourly load."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 3)
        rows = [
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "hour_chicago": hrs[0],
                "kwh_hour": 1.0,
            },
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "hour_chicago": hrs[1],
                "kwh_hour": 5.0,
            },
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "hour_chicago": hrs[2],
                "kwh_hour": 2.0,
            },
        ]
        loads = pl.DataFrame(rows).with_columns(
            pl.col("hour_chicago").cast(pl.Datetime("us")),
            pl.col("kwh_hour").cast(pl.Float64),
        )
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=10.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        assert result["peak_kwh_hour"][0] == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. pct_savings and net_pct_savings
# ═══════════════════════════════════════════════════════════════════════════


class TestPercentSavings:
    """pct_savings = bill_diff / bill_a x 100; null when bill_a <= 0."""

    def test_pct_savings_definition(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=1.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=8.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        bill_a = result["bill_a_dollars"][0]
        bill_diff = result["bill_diff_dollars"][0]
        expected_pct = bill_diff / bill_a * 100
        assert result["pct_savings"][0] == pytest.approx(expected_pct)

    def test_pct_savings_null_when_zero_kwh(self) -> None:
        """Zero usage → bill_a = 0 → pct_savings must be null (no division by zero)."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=0.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=8.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        assert result["pct_savings"][0] is None

    def test_net_pct_savings_with_capacity_charge(self) -> None:
        """net_pct_savings = (bill_diff - capacity - admin) / bill_a x 100."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=1.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=8.0, label="B")

        result = compute_household_bills(
            loads,
            tariff_a,
            tariff_b,
            capacity_rate_dollars_per_kw_month=2.0,
            admin_fee_dollars=1.0,
        )

        bill_a = result["bill_a_dollars"][0]
        bill_diff = result["bill_diff_dollars"][0]
        cap = result["capacity_charge_dollars"][0]
        admin = result["admin_fee_dollars"][0]
        expected_net_pct = (bill_diff - cap - admin) / bill_a * 100
        assert result["net_pct_savings"][0] == pytest.approx(expected_net_pct)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Capacity and admin fee
# ═══════════════════════════════════════════════════════════════════════════


class TestCapacityAndAdminFee:
    """Verify capacity charge and admin fee are applied correctly to tariff B."""

    def test_capacity_charge_equals_peak_times_rate(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 3)
        # Peak hour = 5 kWh
        rows = [
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "hour_chicago": hrs[0],
                "kwh_hour": 5.0,
            },
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "hour_chicago": hrs[1],
                "kwh_hour": 2.0,
            },
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "hour_chicago": hrs[2],
                "kwh_hour": 1.0,
            },
        ]
        loads = pl.DataFrame(rows).with_columns(
            pl.col("hour_chicago").cast(pl.Datetime("us")),
            pl.col("kwh_hour").cast(pl.Float64),
        )
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=10.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b, capacity_rate_dollars_per_kw_month=3.0)

        # capacity_kw = peak_kwh_hour = 5.0; capacity_charge = 5 x $3 = $15
        assert result["capacity_charge_dollars"][0] == pytest.approx(15.0)

    def test_admin_fee_fixed_per_household(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 2)
        loads = _make_loads(["A001", "A002"], hrs, kwh_per_hour=1.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=10.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b, admin_fee_dollars=5.0)

        for val in result["admin_fee_dollars"].to_list():
            assert val == pytest.approx(5.0)

    def test_no_charges_when_defaults(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 2)
        loads = _make_loads(["A001"], hrs)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=10.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        assert result["capacity_charge_dollars"][0] == pytest.approx(0.0)
        assert result["admin_fee_dollars"][0] == pytest.approx(0.0)

    def test_net_bill_diff_reduced_by_charges(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs, kwh_per_hour=1.0)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=8.0, label="B")

        result_no_charges = compute_household_bills(loads, tariff_a, tariff_b)
        result_with_charges = compute_household_bills(
            loads, tariff_a, tariff_b, capacity_rate_dollars_per_kw_month=1.0, admin_fee_dollars=0.5
        )

        assert result_with_charges["net_bill_diff_dollars"][0] < result_no_charges["net_bill_diff_dollars"][0]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Multiple accounts
# ═══════════════════════════════════════════════════════════════════════════


class TestMultipleAccounts:
    """Each account gets its own row; no cross-account contamination."""

    def test_one_row_per_account(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001", "A002", "A003"], hrs)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=8.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        assert result.height == 3
        assert result["account_identifier"].n_unique() == 3

    def test_accounts_dont_share_kwh(self) -> None:
        """A001 uses 1 kWh/hr, A002 uses 3 kWh/hr — bills must differ proportionally."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        rows_a1 = [
            {
                "account_identifier": "A001",
                "zip_code": "60601-0001",
                "hour_chicago": h,
                "kwh_hour": 1.0,
            }
            for h in hrs
        ]
        rows_a2 = [
            {
                "account_identifier": "A002",
                "zip_code": "60601-0001",
                "hour_chicago": h,
                "kwh_hour": 3.0,
            }
            for h in hrs
        ]
        loads = pl.DataFrame(rows_a1 + rows_a2).with_columns(
            pl.col("hour_chicago").cast(pl.Datetime("us")),
            pl.col("kwh_hour").cast(pl.Float64),
        )
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=10.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)

        a1_kwh = result.filter(pl.col("account_identifier") == "A001")["total_kwh"][0]
        a2_kwh = result.filter(pl.col("account_identifier") == "A002")["total_kwh"][0]
        assert a2_kwh == pytest.approx(3 * a1_kwh)


# ═══════════════════════════════════════════════════════════════════════════
# 5. _join_tariff fail-loud conditions
# ═══════════════════════════════════════════════════════════════════════════


class TestJoinTariffFailLoud:
    """_join_tariff must raise on null prices (gap) or row inflation (duplicate hours)."""

    def test_null_price_raises_value_error(self) -> None:
        """A tariff missing one hour produces null prices → ValueError."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 3)
        loads = _make_loads(["A001"], hrs)

        # Tariff only covers 2 of the 3 hours, and has the price col named correctly
        tariff = pl.DataFrame({
            "datetime_chicago": pl.Series(hrs[:2]).cast(pl.Datetime("us")),
            "price_A_cents": [10.0, 10.0],
        })

        with pytest.raises(ValueError, match="no matching price"):
            _join_tariff(loads, tariff, "price_A_cents", "A")

    def test_duplicate_tariff_hour_raises_runtime_error(self) -> None:
        """Duplicate datetime_chicago in tariff inflates row count → RuntimeError."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 2)
        loads = _make_loads(["A001"], hrs)

        # Tariff has a duplicate at hrs[0]
        tariff = pl.DataFrame({
            "datetime_chicago": pl.Series([hrs[0], hrs[0], hrs[1]]).cast(pl.Datetime("us")),
            "price_A_cents": [10.0, 10.0, 10.0],
        })

        with pytest.raises(RuntimeError, match="row count"):
            _join_tariff(loads, tariff, "price_A_cents", "A")

    def test_full_coverage_does_not_raise(self) -> None:
        """A tariff that covers all load hours must not raise."""
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 24)
        loads = _make_loads(["A001"], hrs)
        tariff = pl.DataFrame({
            "datetime_chicago": pl.Series(hrs).cast(pl.Datetime("us")),
            "price_A_cents": [10.0] * 24,
        })
        result = _join_tariff(loads, tariff, "price_A_cents", "A")
        assert result.height == loads.height


# ═══════════════════════════════════════════════════════════════════════════
# 6. Output schema
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputSchema:
    """All expected output columns must be present regardless of optional params."""

    REQUIRED_COLS: ClassVar[set[str]] = {
        "account_identifier",
        "zip_code",
        "total_kwh",
        "bill_a_dollars",
        "bill_b_dollars",
        "bill_diff_dollars",
        "peak_kwh_hour",
        "capacity_kw",
        "capacity_charge_dollars",
        "admin_fee_dollars",
        "net_bill_diff_dollars",
        "pct_savings",
        "net_pct_savings",
    }

    def test_all_columns_present_no_charges(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 2)
        loads = _make_loads(["A001"], hrs)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=8.0, label="B")

        result = compute_household_bills(loads, tariff_a, tariff_b)
        missing = self.REQUIRED_COLS - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_all_columns_present_with_charges(self) -> None:
        hrs = _hours(dt.datetime(2023, 7, 1, 0), 2)
        loads = _make_loads(["A001"], hrs)
        tariff_a = _make_tariff(hrs, price=10.0, label="A")
        tariff_b = _make_tariff(hrs, price=8.0, label="B")

        result = compute_household_bills(
            loads,
            tariff_a,
            tariff_b,
            capacity_rate_dollars_per_kw_month=1.0,
            admin_fee_dollars=0.5,
        )
        missing = self.REQUIRED_COLS - set(result.columns)
        assert not missing, f"Missing columns with charges: {missing}"
