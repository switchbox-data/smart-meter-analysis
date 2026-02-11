# Pricing Simulation Test Plan

Reviewer-facing checklist for the RTP billing pipeline (`analysis/rtp/`) and
orchestrator (`scripts/run_billing_pipeline.py`).

## Quick Reference: Running Tests

```bash
# All pricing simulation tests (< 3 min on sample data)
pytest tests/test_billing_pipeline_e2e.py tests/test_regression_predictor_modes.py tests/test_fail_loud_conditions.py -v

# Individual modules
pytest tests/test_billing_pipeline_e2e.py -v          # E2E orchestrator (~90s)
pytest tests/test_regression_predictor_modes.py -v     # Predictor modes (~2s)
pytest tests/test_fail_loud_conditions.py -v           # Fail-loud conditions (~15s)

# DST roll-in tests (already merged)
pytest tests/test_dst_rollin.py -v
```

## Test Modules

### 1. `tests/test_billing_pipeline_e2e.py`

End-to-end orchestrator test using real sample data in `data/processed/` and
`data/reference/`.  Runs via subprocess, writes to `tmp_path`.

| Test class | What it checks | Key assertions |
|---|---|---|
| `TestDirectoryLayout` | Expected dir tree: `_tmp/`, `month=YYYYMM/`, manifest, log | Files/dirs exist |
| `TestPerMonthOutputs` | Hourly loads and bills are non-empty, have required columns | Column presence, no nulls in price columns |
| `TestAnnualAggregate` | Annual totals equal monthly sums; household count matches | Dollar-sum equality, pct_savings recomputed |
| `TestJoinCoverage` | No duplicate accounts in bills; no silent drops from loads | Set comparison of account IDs |
| `TestRegressionOutputs` | Regression artifacts exist; JSON has both models, R-sq, predictors | Schema of results + metadata JSON |
| `TestManifest` | git SHA, months, month_summary, parameters, steps_completed | Required keys + value validation |

**Runtime:** ~90 seconds (two subprocess runs: with and without regression).

**Prerequisite data:**
- `data/processed/comed_202308.parquet` (145,824 rows, 98 accounts)
- `data/reference/comed_flat_hourly_prices_2023.parquet`
- `data/reference/comed_stou_hourly_prices_2023.parquet`
- `data/reference/comed_bg_zip4_crosswalk.txt`
- `data/reference/census_17_2023.parquet`

If any file is missing, the module is skipped with a clear message.

### 2. `tests/test_regression_predictor_modes.py`

Unit tests for `detect_predictors()`, `_normalize_zip4_expr()`, and
`_resolve_col()` using synthetic DataFrames (no subprocess, no disk I/O).

| Test class | What it checks |
|---|---|
| `TestAutoMode` | Auto-infer numeric cols, exclude IDs/NAME/all-null, sorted output |
| `TestCoreMode` | Returns income + building_pct; partial match; fails if neither present |
| `TestExplicitMode` | Validates comma-separated list against census; fails on missing cols |
| `TestNormalizeZip4` | Formats: `#####-####`, `#########`, 5-digit, empty, whitespace, mixed |
| `TestResolveCol` | Preferred found, fallback used, neither raises RuntimeError |

**Runtime:** < 2 seconds.

### 3. `tests/test_fail_loud_conditions.py`

Negative tests that verify non-zero exit codes and clear error messages.

| Test class | What it checks |
|---|---|
| `TestMissingBillsColumns` | Missing `account_identifier`; missing both savings columns |
| `TestCrosswalkCoverageThreshold` | 100% drop rate vs 5% threshold; 100% threshold allows proceeding |
| `TestZeroPredictors` | Nonexistent explicit predictor; core mode with no core cols in census |
| `TestMissingInputFiles` | Nonexistent bills file; nonexistent census file |
| `TestOrchestratorFailLoud` | Missing interval data; invalid month format (`2023-08` vs `202308`) |
| `TestOutcomeColumnFallback` | `pct_savings` fallback when `net_pct_savings` absent; `net_bill_diff_dollars` fallback |

**Runtime:** ~15 seconds (small synthetic data, subprocess calls).

## Failure Semantics

All scripts use **fail-loud** semantics: exit non-zero with a clear error
message rather than silently producing wrong results.

| Condition | Script | Expected behavior |
|---|---|---|
| Required column missing from bills | `build_regression_dataset.py` | Exit 1, log error naming the column |
| Neither savings column found | `build_regression_dataset.py` | RuntimeError: "neither 'net_pct_savings' nor 'pct_savings'" |
| Crosswalk drop rate > threshold | `build_regression_dataset.py` | RuntimeError with actual vs threshold pct |
| Zero BGs after filtering | `build_regression_dataset.py` | Exit 1: "No block groups remain" |
| Explicit predictor not in census | `build_regression_dataset.py` | RuntimeError: "not found in census" |
| Too few observations for OLS | `build_regression_dataset.py` | RuntimeError: "only N complete observations for M predictors" |
| statsmodels not installed | `build_regression_dataset.py` | `sys.exit()` with install instructions |
| Interval file missing for month | `run_billing_pipeline.py` | Exit 1: "Interval data not found for YYYYMM" |
| Invalid YYYYMM format | `run_billing_pipeline.py` | ValueError: "Invalid month format" |
| Tariff file missing | `run_billing_pipeline.py` | Exit 1: "tariff-a not found" |
| Monthly bills missing during aggregate | `run_billing_pipeline.py` | FileNotFoundError |

## Pipeline Data Flow

```
interval parquet (per month)
    |
    v
compute_hourly_loads.py
    | -> hourly_loads.parquet (account_identifier, zip_code, hour_chicago, kwh_hour)
    v
compute_household_bills.py  (x2 tariffs)
    | -> household_bills.parquet (13 cols: account_identifier, zip_code, ...)
    v
build_annual_aggregate()     (concat months, group_by account)
    | -> annual_household_aggregate.parquet
    v
build_regression_dataset.py  (ZIP+4 -> BG crosswalk -> census join -> OLS)
    | -> regression_dataset_bg.parquet
    | -> regression_results.json
    | -> regression_summary.txt
    | -> regression_metadata.json
```

## Reviewer Checklist

- [ ] All three test modules pass: `pytest tests/test_billing_pipeline_e2e.py tests/test_regression_predictor_modes.py tests/test_fail_loud_conditions.py -v`
- [ ] DST roll-in tests still pass: `pytest tests/test_dst_rollin.py -v`
- [ ] Regression script produces two OLS models (savings + bill_diff) with non-zero observations
- [ ] Annual aggregate dollar sums match monthly sums (verified by `TestAnnualAggregate`)
- [ ] `pct_savings` is recomputed from annual totals, not averaged from monthly values
- [ ] Manifest JSON includes git SHA, month-by-month row counts, all parameters
- [ ] No writes to tracked data directories (all test output via `tmp_path`)
- [ ] `--predictors core` restricts to `median_household_income` + `old_building_pct` only
- [ ] `--predictors auto` excludes `block_group_geoid`, `GEOID`, `NAME`, and all-null columns
- [ ] Crosswalk join uses `zip4` (9-char `#####-####`) not 5-digit `zip_code`
- [ ] Outcome column fallback: `net_pct_savings` -> `pct_savings` with metadata flag
