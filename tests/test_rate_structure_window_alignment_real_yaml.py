"""Guard: STOU and DTOU REAL YAML files must have identical TOU window definitions.

If someone edits hour boundaries in comed_stou_2026.yaml but forgets to
mirror the change in comed_dtou_2026.yaml (or vice versa), these tests fail
loudly before any calendar parquet is built.

What is checked:
  - Both YAML files exist in rate_structures/
  - Season names, start_mmdd, end_mmdd are identical
  - Period names (morning, midday_peak, evening, overnight) are identical
  - Period start_hour and end_hour are identical in every season
  - Prices are intentionally NOT compared (they differ by design)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_tariff_hourly_prices import compare_window_definitions, load_config  # noqa: E402

STOU_YAML = REPO_ROOT / "rate_structures" / "comed_stou_2026.yaml"
DTOU_YAML = REPO_ROOT / "rate_structures" / "comed_dtou_2026.yaml"

_YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None

pytestmark = pytest.mark.skipif(not _YAML_AVAILABLE, reason="PyYAML not installed")

_EXPECTED_PERIODS = {"morning", "midday_peak", "evening", "overnight"}
_EXPECTED_SEASONS = {"summer", "nonsummer"}


def test_both_yaml_files_exist() -> None:
    """Guard: DTOU YAML must be committed alongside STOU YAML."""
    assert STOU_YAML.exists(), f"STOU YAML not found: {STOU_YAML}"
    assert DTOU_YAML.exists(), f"DTOU YAML not found: {DTOU_YAML}"


def test_stou_dtou_window_definitions_are_identical() -> None:
    """Core invariant: every season date and period hour boundary must match."""
    stou_cfg = load_config(STOU_YAML)
    dtou_cfg = load_config(DTOU_YAML)

    ok, diff_msg = compare_window_definitions(
        stou_cfg,
        dtou_cfg,
        name_a=STOU_YAML.name,
        name_b=DTOU_YAML.name,
    )
    assert ok, f"\n{diff_msg}"


def test_stou_dtou_have_expected_season_names() -> None:
    """Both files must define summer and nonsummer seasons."""
    for label, yaml_path in [("STOU", STOU_YAML), ("DTOU", DTOU_YAML)]:
        cfg = load_config(yaml_path)
        actual = {s["name"] for s in cfg["seasons"]}
        missing = _EXPECTED_SEASONS - actual
        unexpected = actual - _EXPECTED_SEASONS
        assert not missing, f"{label}: missing seasons {missing}"
        assert not unexpected, f"{label}: unexpected seasons {unexpected}"


def test_stou_dtou_have_expected_period_names() -> None:
    """Both files must define the four canonical period labels in every season."""
    for label, yaml_path in [("STOU", STOU_YAML), ("DTOU", DTOU_YAML)]:
        cfg = load_config(yaml_path)
        for season in cfg["seasons"]:
            actual = {p["period"] for p in season["periods"]}
            missing = _EXPECTED_PERIODS - actual
            unexpected = actual - _EXPECTED_PERIODS
            assert not missing, f"{label} season '{season['name']}': missing periods {missing}"
            assert not unexpected, f"{label} season '{season['name']}': unexpected periods {unexpected}"


def test_stou_dtou_prices_differ() -> None:
    """Sanity: once DTOU placeholder prices are filled in, they must differ from STOU.

    Skipped when DTOU prices are still all-zero placeholders (see TODO in
    rate_structures/comed_dtou_2026.yaml).
    """
    stou_cfg = load_config(STOU_YAML)
    dtou_cfg = load_config(DTOU_YAML)

    dtou_prices = {p["price"] for s in dtou_cfg["seasons"] for p in s["periods"]}
    if dtou_prices == {0.0} or dtou_prices == {0}:
        pytest.skip("DTOU prices are still placeholder zeros — skipping price diff check")

    stou_prices = {p["price"] for s in stou_cfg["seasons"] for p in s["periods"]}
    assert stou_prices != dtou_prices, (
        "STOU and DTOU prices are identical — is comed_dtou_2026.yaml correctly filled in?"
    )
