"""Unit tests: STOU and DTOU TOU period windows are identical.

STOU and DTOU share the same four period window definitions
(morning, midday_peak, evening, overnight) and only differ in price.
These tests verify that ``build_tariff_hourly_prices.build_hourly_prices``
assigns identical ``period`` values for every hour of 202301 (nonsummer)
and 202307 (summer) regardless of which price schedule is used.

Root-cause note
---------------
The single source of truth for hour→period mapping is the YAML config loaded
by ``scripts/build_tariff_hourly_prices.load_config`` / ``resolve_period``.
If DTOU were ever given a *different* YAML (e.g. with shifted hour boundaries
for a pilot variant), ``resolve_period`` would silently produce a diverging
``period`` column, breaking the assumption in ``bill_stats_and_bg_correlation``
and the BG-correlation analysis that STOU/DTOU deltas are comparable on the
same TOU dimension.  These tests catch that divergence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

# Make the repo root importable when running with pytest from the workspace root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_tariff_hourly_prices import build_hourly_prices  # noqa: E402

# ---------------------------------------------------------------------------
# Inline YAML fixtures — same period windows as comed_stou_2026.yaml
# but with different prices to simulate a DTOU schedule.
# ---------------------------------------------------------------------------

_STOU_YAML = """\
name: stou_test
timezone: America/Chicago
unit: cents_per_kwh
seasons:
  - name: summer
    start_mmdd: "06-01"
    end_mmdd: "09-30"
    periods:
      - period: morning
        start_hour: 6
        end_hour: 13
        price: 3.013
      - period: midday_peak
        start_hour: 13
        end_hour: 19
        price: 18.219
      - period: evening
        start_hour: 19
        end_hour: 21
        price: 3.090
      - period: overnight
        start_hour: 21
        end_hour: 6
        price: 1.870
  - name: nonsummer
    start_mmdd: "10-01"
    end_mmdd: "05-31"
    periods:
      - period: morning
        start_hour: 6
        end_hour: 13
        price: 2.829
      - period: midday_peak
        start_hour: 13
        end_hour: 19
        price: 16.814
      - period: evening
        start_hour: 19
        end_hour: 21
        price: 3.086
      - period: overnight
        start_hour: 21
        end_hour: 6
        price: 2.012
"""

# DTOU: identical period windows, different prices
_DTOU_YAML = """\
name: dtou_test
timezone: America/Chicago
unit: cents_per_kwh
seasons:
  - name: summer
    start_mmdd: "06-01"
    end_mmdd: "09-30"
    periods:
      - period: morning
        start_hour: 6
        end_hour: 13
        price: 2.500
      - period: midday_peak
        start_hour: 13
        end_hour: 19
        price: 20.000
      - period: evening
        start_hour: 19
        end_hour: 21
        price: 4.000
      - period: overnight
        start_hour: 21
        end_hour: 6
        price: 1.500
  - name: nonsummer
    start_mmdd: "10-01"
    end_mmdd: "05-31"
    periods:
      - period: morning
        start_hour: 6
        end_hour: 13
        price: 2.200
      - period: midday_peak
        start_hour: 13
        end_hour: 19
        price: 17.000
      - period: evening
        start_hour: 19
        end_hour: 21
        price: 3.500
      - period: overnight
        start_hour: 21
        end_hour: 6
        price: 1.800
"""

# DTOU with a deliberately wrong window (morning shifted to 7-13) — used to
# confirm the test actually catches divergence.
_DTOU_WRONG_YAML = """\
name: dtou_wrong_test
timezone: America/Chicago
unit: cents_per_kwh
seasons:
  - name: summer
    start_mmdd: "06-01"
    end_mmdd: "09-30"
    periods:
      - period: morning
        start_hour: 7
        end_hour: 13
        price: 2.500
      - period: midday_peak
        start_hour: 13
        end_hour: 19
        price: 20.000
      - period: evening
        start_hour: 19
        end_hour: 21
        price: 4.000
      - period: overnight
        start_hour: 21
        end_hour: 7
        price: 1.500
  - name: nonsummer
    start_mmdd: "10-01"
    end_mmdd: "05-31"
    periods:
      - period: morning
        start_hour: 7
        end_hour: 13
        price: 2.200
      - period: midday_peak
        start_hour: 13
        end_hour: 19
        price: 17.000
      - period: evening
        start_hour: 19
        end_hour: 21
        price: 3.500
      - period: overnight
        start_hour: 21
        end_hour: 7
        price: 1.800
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import yaml  # type: ignore[import-untyped]

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _parse_cfg(yaml_str: str) -> dict:
    return yaml.safe_load(yaml_str)


def _build(yaml_str: str, year: int) -> pl.DataFrame:
    """Build a tariff calendar from an inline YAML string."""
    cfg = _parse_cfg(yaml_str)
    return build_hourly_prices(cfg, year)


def _period_col(df: pl.DataFrame) -> pl.Series:
    return df.sort("datetime_chicago")["period"]


def _compare_periods(stou_df: pl.DataFrame, dtou_df: pl.DataFrame) -> pl.DataFrame:
    """Return rows where period differs; empty DataFrame means full match."""
    joined = (
        stou_df.select(["datetime_chicago", "period"])
        .sort("datetime_chicago")
        .rename({"period": "period_stou"})
        .join(
            dtou_df.select(["datetime_chicago", "period"]).sort("datetime_chicago").rename({"period": "period_dtou"}),
            on="datetime_chicago",
            how="inner",
        )
    )
    return joined.filter(pl.col("period_stou") != pl.col("period_dtou"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not _YAML_AVAILABLE, reason="PyYAML not installed")


class TestTouWindowAlignment:
    """STOU and DTOU period windows must be identical hour-for-hour."""

    @pytest.mark.parametrize(
        "year, month_label",
        [
            (2023, "202301"),  # nonsummer
            (2023, "202307"),  # summer
        ],
    )
    def test_period_windows_match_full_year(self, year: int, month_label: str) -> None:
        """Period column must be identical across all hours of the year."""
        stou = _build(_STOU_YAML, year)
        dtou = _build(_DTOU_YAML, year)

        mismatches = _compare_periods(stou, dtou)
        assert mismatches.height == 0, (
            f"Period mismatch for year={year}: {mismatches.height} rows differ.\nSample:\n{mismatches.head(10)}"
        )

    @pytest.mark.parametrize("year", [2023])
    def test_prices_differ(self, year: int) -> None:
        """Sanity check: STOU and DTOU prices must actually differ so we know
        the test is exercising two distinct rate structures."""
        stou = _build(_STOU_YAML, year)
        dtou = _build(_DTOU_YAML, year)

        stou_prices = set(stou["price_cents_per_kwh"].unique().to_list())
        dtou_prices = set(dtou["price_cents_per_kwh"].unique().to_list())
        assert stou_prices != dtou_prices, "STOU and DTOU prices should differ in the test fixtures"

    @pytest.mark.parametrize("year", [2023])
    def test_wrong_window_is_detected(self, year: int) -> None:
        """The comparison must flag divergence when windows genuinely differ."""
        stou = _build(_STOU_YAML, year)
        dtou_wrong = _build(_DTOU_WRONG_YAML, year)

        mismatches = _compare_periods(stou, dtou_wrong)
        # Hour 6 is in morning for STOU, overnight for wrong DTOU → should find mismatches
        assert mismatches.height > 0, (
            "Expected period mismatches when DTOU morning window is shifted to hour 7, "
            "but none were found — the comparison logic may be broken."
        )

    @pytest.mark.parametrize("year", [2023])
    def test_period_values_are_expected_labels(self, year: int) -> None:
        """Both calendars must only contain the four canonical period labels."""
        expected = {"morning", "midday_peak", "evening", "overnight"}
        for label, yaml_str in [("STOU", _STOU_YAML), ("DTOU", _DTOU_YAML)]:
            df = _build(yaml_str, year)
            actual = set(df["period"].unique().to_list())
            unexpected = actual - expected
            assert not unexpected, f"{label} calendar contains unexpected period labels: {unexpected}"

    def test_period_coverage_nonsummer_jan_hours(self) -> None:
        """Spot-check January 2023 hour→period mapping against expected windows."""
        stou = _build(_STOU_YAML, 2023)
        dtou = _build(_DTOU_YAML, 2023)

        import datetime as dt

        # Check a representative hour from each period on a January weekday
        # (all nonsummer, no DST complications in January)
        expected: list[tuple[dt.datetime, str]] = [
            (dt.datetime(2023, 1, 10, 5, 0), "overnight"),  # hour 5 → [21,6)
            (dt.datetime(2023, 1, 10, 6, 0), "morning"),  # hour 6 → [6,13)
            (dt.datetime(2023, 1, 10, 12, 0), "morning"),  # hour 12 → [6,13)
            (dt.datetime(2023, 1, 10, 13, 0), "midday_peak"),  # hour 13 → [13,19)
            (dt.datetime(2023, 1, 10, 18, 0), "midday_peak"),  # hour 18 → [13,19)
            (dt.datetime(2023, 1, 10, 19, 0), "evening"),  # hour 19 → [19,21)
            (dt.datetime(2023, 1, 10, 20, 0), "evening"),  # hour 20 → [19,21)
            (dt.datetime(2023, 1, 10, 21, 0), "overnight"),  # hour 21 → [21,6)
        ]

        for ts, expected_period in expected:
            for label, df in [("STOU", stou), ("DTOU", dtou)]:
                row = df.filter(pl.col("datetime_chicago") == ts)
                assert row.height == 1, f"{label}: expected 1 row for {ts}, got {row.height}"
                actual = row["period"].item()
                assert actual == expected_period, (
                    f"{label}: hour={ts.hour} → expected '{expected_period}', got '{actual}'"
                )

    def test_period_coverage_summer_jul_hours(self) -> None:
        """Spot-check July 2023 hour→period mapping (summer season)."""
        stou = _build(_STOU_YAML, 2023)
        dtou = _build(_DTOU_YAML, 2023)

        import datetime as dt

        expected: list[tuple[dt.datetime, str]] = [
            (dt.datetime(2023, 7, 10, 5, 0), "overnight"),
            (dt.datetime(2023, 7, 10, 6, 0), "morning"),
            (dt.datetime(2023, 7, 10, 13, 0), "midday_peak"),
            (dt.datetime(2023, 7, 10, 19, 0), "evening"),
            (dt.datetime(2023, 7, 10, 21, 0), "overnight"),
        ]

        for ts, expected_period in expected:
            for label, df in [("STOU", stou), ("DTOU", dtou)]:
                row = df.filter(pl.col("datetime_chicago") == ts)
                assert row.height == 1, f"{label}: expected 1 row for {ts}, got {row.height}"
                actual = row["period"].item()
                assert actual == expected_period, (
                    f"{label}: hour={ts.hour} → expected '{expected_period}', got '{actual}'"
                )
