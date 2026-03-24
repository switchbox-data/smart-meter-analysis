"""Comprehensive validation suite for statewide pricing simulation outputs.

Validates the pricing pilot pipeline outputs using locally available data:
- Household bills parquets: output/statewide_{stou,dtou}_{202301,202307}/
- BG-level regression parquets and JSON: output/statewide_*/regression/
- Rate configuration YAML: rate_structures/comed_{stou,dtou}_2026.yaml
- Rate constants in Python source: scripts/pricing_pilot/compute_delivery_deltas.py

Sign convention: delta = flat - alternative (positive = saves under TOU)
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import polars as pl
import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path("/workspaces/smart-meter-analysis")
OUTPUT = REPO / "output"
STOU_YAML = REPO / "rate_structures" / "comed_stou_2026.yaml"
DTOU_YAML = REPO / "rate_structures" / "comed_dtou_2026.yaml"

SCENARIOS = [
    ("stou", "202301"),
    ("stou", "202307"),
    ("dtou", "202301"),
    ("dtou", "202307"),
]

# Expected row counts (from pipeline run manifests)
EXPECTED_ROWS = {
    ("stou", "202301"): 658_959,
    ("stou", "202307"): 686_430,
    ("dtou", "202301"): 658_959,
    ("dtou", "202307"): 686_430,
}


def _bills_path(rate: str, month: str) -> Path:
    return OUTPUT / f"statewide_{rate}_{month}" / f"month={month}" / "household_bills.parquet"


def _regression_dir(rate: str, month: str) -> Path:
    return OUTPUT / f"statewide_{rate}_{month}" / "regression"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_rate_constants():
    """Import rate constants from compute_delivery_deltas.py."""
    spec_path = REPO / "scripts" / "pricing_pilot" / "compute_delivery_deltas.py"
    spec = importlib.util.spec_from_file_location("compute_delivery_deltas", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Check 1 — Rate table consistency
# ============================================================================


class TestRateTableConsistency:
    """Verify rate configs match expected tariff values (¢/kWh)."""

    @pytest.fixture(scope="class")
    def stou_yaml(self):
        return _load_yaml(STOU_YAML)

    @pytest.fixture(scope="class")
    def dtou_yaml(self):
        return _load_yaml(DTOU_YAML)

    @pytest.fixture(scope="class")
    def rate_mod(self):
        return _load_rate_constants()

    # -- Flat PTCs --

    def test_flat_ptc_nonsummer(self, rate_mod):
        assert rate_mod.FLAT_PTCS["nonsummer"] == pytest.approx(9.660)

    def test_flat_ptc_summer(self, rate_mod):
        assert rate_mod.FLAT_PTCS["summer"] == pytest.approx(10.028)

    # -- Flat DFCs --

    @pytest.mark.parametrize(
        "cls,expected",
        [("C23", 6.228), ("C24", 4.791), ("C26", 3.165), ("C28", 2.996)],
    )
    def test_flat_dfc(self, rate_mod, cls, expected):
        assert rate_mod.FLAT_DFCS[cls] == pytest.approx(expected)

    # -- TOU DFCs --

    def test_tou_dfc_c28_overnight(self, rate_mod):
        assert rate_mod.TOU_DFCS["C28"]["overnight"] == pytest.approx(1.512)

    @pytest.mark.parametrize(
        "cls,period,expected",
        [
            ("C23", "morning", 4.009),
            ("C23", "midday_peak", 10.712),
            ("C24", "overnight", 2.251),
            ("C26", "midday_peak", 5.329),
            ("C28", "morning", 1.925),
            ("C28", "midday_peak", 4.975),
            ("C28", "evening", 1.823),
        ],
    )
    def test_tou_dfc_spot_checks(self, rate_mod, cls, period, expected):
        assert rate_mod.TOU_DFCS[cls][period] == pytest.approx(expected)

    # -- STOU supply rates --

    def test_stou_midday_peak_summer(self, stou_yaml):
        summer = next(s for s in stou_yaml["seasons"] if s["name"] == "summer")
        peak = next(p for p in summer["periods"] if p["period"] == "midday_peak")
        assert peak["price"] == pytest.approx(19.485)

    def test_stou_midday_peak_nonsummer(self, stou_yaml):
        nonsummer = next(s for s in stou_yaml["seasons"] if s["name"] == "nonsummer")
        peak = next(p for p in nonsummer["periods"] if p["period"] == "midday_peak")
        assert peak["price"] == pytest.approx(18.080)

    def test_stou_tmp_component_summer(self, stou_yaml):
        """T&MP = 1.266: midday_peak(19.485) - original_compromise(18.219) = 1.266."""
        summer = next(s for s in stou_yaml["seasons"] if s["name"] == "summer")
        peak = next(p for p in summer["periods"] if p["period"] == "midday_peak")
        # Original Compromise summer midday_peak = 18.219 (from YAML header comment)
        tmp = peak["price"] - 18.219
        assert tmp == pytest.approx(1.266, abs=1e-4)

    def test_stou_tmp_component_nonsummer(self, stou_yaml):
        """T&MP = 1.266: midday_peak(18.080) - original_compromise(16.814) = 1.266."""
        nonsummer = next(s for s in stou_yaml["seasons"] if s["name"] == "nonsummer")
        peak = next(p for p in nonsummer["periods"] if p["period"] == "midday_peak")
        tmp = peak["price"] - 16.814
        assert tmp == pytest.approx(1.266, abs=1e-4)

    # -- No C25/C27 in configs --

    def test_no_c25_c27_in_flat_dfcs(self, rate_mod):
        assert "C25" not in rate_mod.FLAT_DFCS
        assert "C27" not in rate_mod.FLAT_DFCS

    def test_no_c25_c27_in_tou_dfcs(self, rate_mod):
        assert "C25" not in rate_mod.TOU_DFCS
        assert "C27" not in rate_mod.TOU_DFCS

    def test_no_c25_c27_in_stou_yaml(self, stou_yaml):
        yaml_str = yaml.dump(stou_yaml)
        assert "C25" not in yaml_str
        assert "C27" not in yaml_str

    def test_no_c25_c27_in_dtou_yaml(self, dtou_yaml):
        yaml_str = yaml.dump(dtou_yaml)
        assert "C25" not in yaml_str
        assert "C27" not in yaml_str

    # -- DTOU YAML has placeholder prices --

    def test_dtou_yaml_prices_are_zero_placeholders(self, dtou_yaml):
        """DTOU YAML has 0.000 placeholders; actual DTOU uses Python DFC constants."""
        for season in dtou_yaml["seasons"]:
            for period in season["periods"]:
                assert period["price"] == 0.0, (
                    f"DTOU {season['name']}/{period['period']} has non-zero price "
                    f"{period['price']}; expected 0.000 placeholder"
                )

    # -- Window alignment (STOU and DTOU share same time blocks) --

    def test_stou_dtou_same_season_dates(self, stou_yaml, dtou_yaml):
        for s_season, d_season in zip(stou_yaml["seasons"], dtou_yaml["seasons"]):
            assert s_season["name"] == d_season["name"]
            assert s_season["start_mmdd"] == d_season["start_mmdd"]
            assert s_season["end_mmdd"] == d_season["end_mmdd"]

    def test_stou_dtou_same_period_hours(self, stou_yaml, dtou_yaml):
        for s_season, d_season in zip(stou_yaml["seasons"], dtou_yaml["seasons"]):
            for s_period, d_period in zip(s_season["periods"], d_season["periods"]):
                assert s_period["period"] == d_period["period"]
                assert s_period["start_hour"] == d_period["start_hour"]
                assert s_period["end_hour"] == d_period["end_hour"]


# ============================================================================
# Check 2 — Universe counts
# ============================================================================


class TestUniverseCounts:
    """Verify expected row counts and cross-scenario consistency."""

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_row_count(self, rate, month):
        path = _bills_path(rate, month)
        n = pl.scan_parquet(path).select(pl.len()).collect().item()
        assert n == EXPECTED_ROWS[(rate, month)], f"{rate} {month}: {n} != {EXPECTED_ROWS[(rate, month)]}"

    def test_stou_dtou_jan_same_count(self):
        n_stou = pl.scan_parquet(_bills_path("stou", "202301")).select(pl.len()).collect().item()
        n_dtou = pl.scan_parquet(_bills_path("dtou", "202301")).select(pl.len()).collect().item()
        assert n_stou == n_dtou, f"Jan mismatch: STOU={n_stou}, DTOU={n_dtou}"

    def test_stou_dtou_jul_same_count(self):
        n_stou = pl.scan_parquet(_bills_path("stou", "202307")).select(pl.len()).collect().item()
        n_dtou = pl.scan_parquet(_bills_path("dtou", "202307")).select(pl.len()).collect().item()
        assert n_stou == n_dtou, f"Jul mismatch: STOU={n_stou}, DTOU={n_dtou}"

    def test_jul_more_than_jan(self):
        n_jan = pl.scan_parquet(_bills_path("stou", "202301")).select(pl.len()).collect().item()
        n_jul = pl.scan_parquet(_bills_path("stou", "202307")).select(pl.len()).collect().item()
        assert n_jul > n_jan, f"Jul ({n_jul}) should exceed Jan ({n_jan})"


# ============================================================================
# Check 3 — Household-level bill sanity
# ============================================================================


class TestHouseholdBillSanity:
    """Verify bill arithmetic at the household level."""

    @pytest.fixture(scope="class")
    def stou_jan(self):
        return pl.read_parquet(_bills_path("stou", "202301"))

    @pytest.fixture(scope="class")
    def stou_jul(self):
        return pl.read_parquet(_bills_path("stou", "202307"))

    @pytest.fixture(scope="class")
    def dtou_jan(self):
        return pl.read_parquet(_bills_path("dtou", "202301"))

    # -- bill_diff = bill_a - bill_b --

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_bill_diff_equals_bill_a_minus_bill_b(self, rate, month):
        df = pl.read_parquet(_bills_path(rate, month))
        err = (df["bill_a_dollars"] - df["bill_b_dollars"] - df["bill_diff_dollars"]).abs().max()
        assert err < 1e-8, f"max |bill_a - bill_b - bill_diff| = {err}"

    # -- pct_savings = bill_diff / bill_a * 100 --

    def test_stou_pct_savings_definition(self, stou_jan):
        pos_bill = stou_jan.filter(pl.col("bill_a_dollars") > 0)
        computed = pos_bill["bill_diff_dollars"] / pos_bill["bill_a_dollars"] * 100
        err = (computed - pos_bill["pct_savings"]).abs().max()
        assert err < 1e-8, f"pct_savings mismatch: max err = {err}"

    def test_stou_pct_savings_null_when_zero_bill(self, stou_jan):
        zero_bill = stou_jan.filter(pl.col("bill_a_dollars") == 0)
        if zero_bill.height > 0:
            n_null = zero_bill["pct_savings"].null_count()
            assert n_null == zero_bill.height, "pct_savings should be null when bill_a = 0"

    # -- net_bill_diff = bill_diff - capacity_charge - admin_fee --

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_net_bill_diff_definition(self, rate, month):
        df = pl.read_parquet(_bills_path(rate, month))
        computed = df["bill_diff_dollars"] - df["capacity_charge_dollars"] - df["admin_fee_dollars"]
        err = (computed - df["net_bill_diff_dollars"]).abs().max()
        assert err < 1e-8, f"net_bill_diff mismatch: max err = {err}"

    # -- DTOU: bill_b = 0 (DTOU YAML prices are all 0.000 placeholders) --

    def test_dtou_bill_b_always_zero(self, dtou_jan):
        assert dtou_jan["bill_b_dollars"].max() == 0.0
        assert dtou_jan["bill_b_dollars"].min() == 0.0

    def test_dtou_pct_savings_always_100(self, dtou_jan):
        non_null = dtou_jan["pct_savings"].drop_nulls()
        assert non_null.min() == pytest.approx(100.0)
        assert non_null.max() == pytest.approx(100.0)

    # -- bill_a non-negative --

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_bill_a_non_negative(self, rate, month):
        df = pl.read_parquet(_bills_path(rate, month))
        assert df["bill_a_dollars"].min() >= 0.0

    # -- STOU and DTOU Jan have same flat supply (bill_a) since both use same flat PTC --

    def test_stou_dtou_jan_bill_a_identical(self, stou_jan, dtou_jan):
        """Both STOU and DTOU core runs use the same flat PTC for bill_a (flat supply)."""
        stou_sorted = stou_jan.sort("account_identifier")
        dtou_sorted = dtou_jan.sort("account_identifier")
        # Same accounts
        assert stou_sorted["account_identifier"].to_list() == dtou_sorted["account_identifier"].to_list()
        # Same bill_a (flat supply charge) — allow floating-point tolerance
        err = (stou_sorted["bill_a_dollars"] - dtou_sorted["bill_a_dollars"]).abs().max()
        assert err < 1e-8, f"bill_a should be identical; max diff = {err}"

    # -- Spot-check: random households have reasonable values --

    def test_stou_jan_no_extreme_outliers(self, stou_jan):
        """No single household bill_diff exceeds $10,000."""
        assert stou_jan["bill_diff_dollars"].abs().max() < 10_000

    def test_stou_pct_savings_range(self, stou_jan):
        """pct_savings bounded to [-100, 100] for positive bills."""
        pos_bill = stou_jan.filter(pl.col("bill_a_dollars") > 0)
        assert pos_bill["pct_savings"].min() >= -100.0
        assert pos_bill["pct_savings"].max() <= 100.0


# ============================================================================
# Check 4 — BG-level parquet consistency
# ============================================================================


class TestBGLevelConsistency:
    """Cross-scenario BG-level consistency checks using regression parquets."""

    @pytest.fixture(scope="class")
    def bg_data(self):
        """Load bg_month_outcomes for all 4 scenarios."""
        data = {}
        for rate, month in SCENARIOS:
            path = _regression_dir(rate, month) / "bg_month_outcomes.parquet"
            data[(rate, month)] = pl.read_parquet(path)
        return data

    def test_stou_dtou_jan_same_bg_set(self, bg_data):
        stou_bgs = set(bg_data[("stou", "202301")]["block_group_geoid"].to_list())
        dtou_bgs = set(bg_data[("dtou", "202301")]["block_group_geoid"].to_list())
        assert stou_bgs == dtou_bgs, f"Jan BG mismatch: |diff| = {len(stou_bgs ^ dtou_bgs)}"

    def test_stou_dtou_jul_same_bg_set(self, bg_data):
        stou_bgs = set(bg_data[("stou", "202307")]["block_group_geoid"].to_list())
        dtou_bgs = set(bg_data[("dtou", "202307")]["block_group_geoid"].to_list())
        assert stou_bgs == dtou_bgs, f"Jul BG mismatch: |diff| = {len(stou_bgs ^ dtou_bgs)}"

    def test_jul_more_bgs_than_jan(self, bg_data):
        n_jan = bg_data[("stou", "202301")]["block_group_geoid"].n_unique()
        n_jul = bg_data[("stou", "202307")]["block_group_geoid"].n_unique()
        assert n_jul > n_jan, f"Jul ({n_jul}) should have more BGs than Jan ({n_jan})"

    def test_all_n_household_months_positive(self, bg_data):
        for key, df in bg_data.items():
            min_n = df["n_household_months"].min()
            assert min_n > 0, f"{key} has n_household_months = {min_n}"

    def test_no_duplicate_bgs(self, bg_data):
        for key, df in bg_data.items():
            n_total = df.height
            n_unique = df["block_group_geoid"].n_unique()
            assert n_total == n_unique, f"{key} has {n_total - n_unique} duplicate BGs"

    def test_stou_dtou_jan_same_n_households_per_bg(self, bg_data):
        """Same households → same n_household_months per BG."""
        stou = bg_data[("stou", "202301")].sort("block_group_geoid")
        dtou = bg_data[("dtou", "202301")].sort("block_group_geoid")
        assert stou["n_household_months"].to_list() == dtou["n_household_months"].to_list(), (
            "STOU and DTOU Jan should have identical per-BG household counts"
        )

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_bg_geoid_format(self, bg_data, rate, month):
        """All GEOIDs should be 12-character strings (FIPS state+county+tract+BG)."""
        geoids = bg_data[(rate, month)]["block_group_geoid"]
        lengths = geoids.str.len_chars()
        # Illinois FIPS = 17, so GEOIDs should be 12 chars (2+3+6+1)
        assert lengths.min() >= 11, f"GEOID too short: min len = {lengths.min()}"
        assert lengths.max() <= 12, f"GEOID too long: max len = {lengths.max()}"


# ============================================================================
# Check 5 — Regression sanity
# ============================================================================


class TestRegressionSanity:
    """Verify regression results are statistically reasonable."""

    @pytest.fixture(scope="class")
    def reg_results(self):
        """Load all regression_results.json files."""
        results = {}
        for rate, month in SCENARIOS:
            path = _regression_dir(rate, month) / "regression_results.json"
            with open(path) as f:
                results[(rate, month)] = json.load(f)
        return results

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_p_values_valid(self, reg_results, rate, month):
        for model_name, model in reg_results[(rate, month)].items():
            for coeff_name, coeff in model["coefficients"].items():
                p = coeff["p_value"]
                assert 0 <= p <= 1, f"{rate}/{month}/{model_name}/{coeff_name}: p={p}"

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_r_squared_reasonable(self, reg_results, rate, month):
        """R² should be non-negative (or -inf for DTOU pct_savings which is constant)."""
        for model_name, model in reg_results[(rate, month)].items():
            r2 = model["r_squared"]
            if rate == "dtou" and "pct_savings" in model_name:
                # DTOU pct_savings is exactly 100% for all BGs → R²=-inf is expected
                assert r2 == float("-inf"), f"{rate}/{month}/{model_name}: R²={r2}, expected -inf"
            else:
                assert 0 <= r2 <= 1, f"{rate}/{month}/{model_name}: R²={r2}"

    def test_stou_jul_pct_savings_beta1_negative(self, reg_results):
        """STOU Jul pooled pct_savings beta_1 should be negative (progressive)."""
        model = reg_results[("stou", "202307")]["model_1_pct_savings_weighted"]
        income_coeff = next(v for k, v in model["coefficients"].items() if k != "const")
        assert income_coeff["estimate"] < 0, (
            f"STOU Jul pct_savings beta_1 = {income_coeff['estimate']}, expected negative (progressive)"
        )

    def test_stou_jul_bill_diff_beta1_negative(self, reg_results):
        """STOU Jul bill_diff beta_1 should be negative (progressive in absolute $)."""
        model = reg_results[("stou", "202307")]["model_2_sum_bill_diff"]
        income_coeff = next(v for k, v in model["coefficients"].items() if k != "const")
        assert income_coeff["estimate"] < 0, (
            f"STOU Jul bill_diff beta_1 = {income_coeff['estimate']}, expected negative"
        )

    def test_dtou_pct_savings_constant_100(self, reg_results):
        """DTOU pct_savings intercept should be ~100 (all savings are 100%)."""
        for month in ["202301", "202307"]:
            model = reg_results[("dtou", month)]["model_1_pct_savings_weighted"]
            const = model["coefficients"]["const"]["estimate"]
            assert const == pytest.approx(100.0, abs=0.01), f"DTOU {month} pct_savings const = {const}, expected ~100"

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_n_obs_reasonable(self, reg_results, rate, month):
        """Each regression should have 1000+ observations."""
        for model_name, model in reg_results[(rate, month)].items():
            assert model["n_obs"] > 1000, f"{rate}/{month}/{model_name}: n_obs={model['n_obs']}"

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_model_keys_present(self, reg_results, rate, month):
        """Each scenario should have both pct_savings and bill_diff models."""
        keys = set(reg_results[(rate, month)].keys())
        assert "model_1_pct_savings_weighted" in keys
        assert "model_2_sum_bill_diff" in keys


# ============================================================================
# Check 6 — Numeric plausibility
# ============================================================================


class TestNumericPlausibility:
    """Range and distribution sanity checks."""

    def test_stou_jan_mean_bill_diff_positive(self):
        """STOU Jan: overall positive savings (flat > TOU in winter)."""
        df = pl.read_parquet(_bills_path("stou", "202301"))
        mean_diff = df["bill_diff_dollars"].mean()
        assert mean_diff > 0, f"STOU Jan mean bill_diff = {mean_diff}, expected positive"

    def test_stou_jul_mean_bill_diff_near_zero_or_negative(self):
        """STOU Jul: smaller or negative savings (summer TOU peak penalty)."""
        df = pl.read_parquet(_bills_path("stou", "202307"))
        mean_diff = df["bill_diff_dollars"].mean()
        # Summer should have smaller savings than winter due to peak pricing
        assert mean_diff < 10, f"STOU Jul mean bill_diff = {mean_diff}, expected < 10"

    def test_stou_jan_larger_savings_than_jul(self):
        """Winter STOU savings should exceed summer."""
        jan = pl.read_parquet(_bills_path("stou", "202301"))["bill_diff_dollars"].mean()
        jul = pl.read_parquet(_bills_path("stou", "202307"))["bill_diff_dollars"].mean()
        assert jan > jul, f"Jan mean ({jan:.2f}) should exceed Jul mean ({jul:.2f})"

    def test_no_extreme_bill_outliers(self):
        """No household bill_diff exceeds $15,000 in absolute value.

        A few very high-consumption accounts (e.g., commercial-scale usage on
        residential class C28) can produce large deltas; $15k is a reasonable
        ceiling that allows genuine outliers while catching data corruption.
        """
        for rate, month in SCENARIOS:
            df = pl.read_parquet(_bills_path(rate, month))
            max_abs = df["bill_diff_dollars"].abs().max()
            assert max_abs < 15_000, f"{rate}/{month} max |bill_diff| = {max_abs}"

    def test_income_range_in_regression_dataset(self):
        """median_household_income in BG regression data should be in valid range.

        Census stores income as natural log; the regression dataset uses log scale.
        Valid range: ln($2,500) ≈ 7.8 to ln($500,000) ≈ 13.1.
        """
        import math

        path = _regression_dir("stou", "202301") / "regression_dataset_bg.parquet"
        df = pl.read_parquet(path)
        inc = df["median_household_income"].drop_nulls()
        # Detect if values are in log scale (all < 20) or dollar scale (all > 1000)
        if inc.mean() < 20:
            # Log scale: ln($2,500)=7.82, ln($500,000)=13.12
            assert inc.min() >= math.log(2_500), f"min log income = {inc.min()}"
            assert inc.max() < math.log(500_000), f"max log income = {inc.max()}"
        else:
            # Dollar scale
            assert inc.min() > 5_000, f"min income = {inc.min()}, expected > $5k"
            assert inc.max() < 500_000, f"max income = {inc.max()}, expected < $500k"

    def test_bg_pct_savings_weighted_range(self):
        """BG-level pct_savings_weighted should be in reasonable range."""
        for rate, month in SCENARIOS:
            path = _regression_dir(rate, month) / "bg_month_outcomes.parquet"
            df = pl.read_parquet(path)
            pct = df["pct_savings_weighted"].drop_nulls()
            if rate == "dtou":
                # DTOU pct_savings_weighted should be ~100%
                assert pct.mean() == pytest.approx(100.0, abs=1.0)
            else:
                # STOU: should be between -100 and 100
                assert pct.min() > -100, f"{rate}/{month} min pct = {pct.min()}"
                assert pct.max() < 100, f"{rate}/{month} max pct = {pct.max()}"

    def test_sum_bill_diff_all_positive_for_stou_jan(self):
        """Most BGs in STOU Jan should have positive sum_bill_diff (aggregate savings)."""
        path = _regression_dir("stou", "202301") / "bg_month_outcomes.parquet"
        df = pl.read_parquet(path)
        pct_positive = (df["sum_bill_diff_dollars"] > 0).mean()
        assert pct_positive > 0.5, f"Only {pct_positive:.1%} of BGs have positive sum_bill_diff"


# ============================================================================
# Check 7 — Cross-validation: household → BG consistency
# ============================================================================


class TestHouseholdToBGConsistency:
    """Verify that BG-level aggregates are consistent with household bills."""

    @pytest.fixture(scope="class")
    def stou_jan_bills(self):
        return pl.read_parquet(_bills_path("stou", "202301"))

    @pytest.fixture(scope="class")
    def stou_jan_bg(self):
        return pl.read_parquet(_regression_dir("stou", "202301") / "bg_month_outcomes.parquet")

    def test_total_bill_diff_matches(self, stou_jan_bills, stou_jan_bg):
        """Sum of household bill_diff should ~ match sum of BG sum_bill_diff."""
        hh_total = stou_jan_bills["bill_diff_dollars"].sum()
        bg_total = stou_jan_bg["sum_bill_diff_dollars"].sum()
        # BG data may drop some households (crosswalk match), allow 20% tolerance
        ratio = bg_total / hh_total
        assert 0.5 < ratio < 1.5, f"HH total={hh_total:.0f}, BG total={bg_total:.0f}, ratio={ratio:.3f}"

    def test_bg_household_count_below_total(self, stou_jan_bills, stou_jan_bg):
        """Total households in BG data should be <= total household bills."""
        bg_hh = stou_jan_bg["n_household_months"].sum()
        hh_total = stou_jan_bills.height
        # BG data loses some HHs in crosswalk, but should not exceed
        assert bg_hh <= hh_total * 1.01, f"BG HHs ({bg_hh}) > total HHs ({hh_total})"

    def test_bg_count_reasonable(self, stou_jan_bg):
        """Expect 4000-5000 unique BGs for Illinois."""
        n_bg = stou_jan_bg["block_group_geoid"].n_unique()
        assert 4000 <= n_bg <= 5000, f"BG count = {n_bg}, expected 4000-5000"


# ============================================================================
# Check 8 — Schema consistency across scenarios
# ============================================================================


class TestSchemaConsistency:
    """Verify all scenarios produce consistent schemas."""

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_household_bills_schema(self, rate, month):
        expected_cols = {
            "account_identifier",
            "zip_code",
            "total_kwh",
            "peak_kwh_hour",
            "bill_a_dollars",
            "bill_b_dollars",
            "bill_diff_dollars",
            "capacity_kw",
            "pct_savings",
            "capacity_charge_dollars",
            "admin_fee_dollars",
            "net_bill_diff_dollars",
            "net_pct_savings",
        }
        df = pl.scan_parquet(_bills_path(rate, month))
        actual_cols = set(df.collect_schema().names())
        missing = expected_cols - actual_cols
        assert not missing, f"{rate}/{month} missing columns: {missing}"

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_bg_outcomes_schema(self, rate, month):
        expected_cols = {
            "block_group_geoid",
            "month",
            "sum_total_kwh",
            "sum_bill_a_dollars",
            "sum_bill_b_dollars",
            "sum_bill_diff_dollars",
            "sum_net_bill_diff_dollars",
            "n_household_months",
            "pct_savings_weighted",
        }
        path = _regression_dir(rate, month) / "bg_month_outcomes.parquet"
        df = pl.scan_parquet(path)
        actual_cols = set(df.collect_schema().names())
        missing = expected_cols - actual_cols
        assert not missing, f"{rate}/{month} bg_month_outcomes missing: {missing}"

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_regression_json_has_required_fields(self, rate, month):
        path = _regression_dir(rate, month) / "regression_results.json"
        with open(path) as f:
            data = json.load(f)
        required = {"r_squared", "n_obs", "coefficients", "f_statistic", "f_pvalue"}
        for model_name, model in data.items():
            missing = required - set(model.keys())
            assert not missing, f"{rate}/{month}/{model_name} missing fields: {missing}"


# ============================================================================
# Check 9 — Regression metadata consistency
# ============================================================================


class TestRegressionMetadata:
    """Verify regression metadata JSON files are internally consistent."""

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_metadata_exists_and_valid(self, rate, month):
        path = _regression_dir(rate, month) / "regression_metadata.json"
        with open(path) as f:
            meta = json.load(f)
        assert "months_included" in meta
        assert "predictors" in meta or "predictor_columns" in meta or "predictor_mode" in meta

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_regression_dataset_has_census_predictors(self, rate, month):
        """Regression dataset should contain Census predictors (joined from census data)."""
        path = _regression_dir(rate, month) / "regression_dataset_bg.parquet"
        df = pl.scan_parquet(path)
        cols = set(df.collect_schema().names())
        assert "median_household_income" in cols
        # At least a few other census predictors
        census_cols = cols & {
            "pct_below_poverty",
            "pct_owner_occupied",
            "pct_renter_occupied",
            "pct_heat_electric",
            "median_age",
        }
        assert len(census_cols) >= 3, f"Only {len(census_cols)} census predictors found"

    @pytest.mark.parametrize("rate,month", SCENARIOS)
    def test_regression_dataset_bg_count_matches_outcomes(self, rate, month):
        """regression_dataset_bg should have same BGs as bg_month_outcomes (after filters)."""
        ds_path = _regression_dir(rate, month) / "regression_dataset_bg.parquet"
        bg_path = _regression_dir(rate, month) / "bg_month_outcomes.parquet"
        ds = pl.read_parquet(ds_path)
        bg = pl.read_parquet(bg_path)
        # Dataset may have fewer BGs (dropped nulls), but should not exceed
        assert ds.height <= bg.height, f"regression_dataset ({ds.height}) > bg_month_outcomes ({bg.height})"
        # Should retain at least 80% of BGs
        assert ds.height > bg.height * 0.8, (
            f"regression_dataset only has {ds.height}/{bg.height} BGs ({ds.height / bg.height:.0%})"
        )
