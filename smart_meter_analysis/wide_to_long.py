from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import polars as pl

__all__ = ["IntervalColSpec", "transform_wide_to_long", "transform_wide_to_long_lf"]

# -------------------------------------------------------------------------------------------------
# Zip4 wide → long canonicalization (ComEd smart-meter interval data)
#
# Context:
# - Source data arrives as one row per account per day with 48 interval "end time" columns
#   (0030..2400). DST anomalies may appear as extra end-time columns (2430, 2500).
# - Downstream clustering and regression require a canonical "long" representation:
#     one row per account per interval with a true interval START timestamp.
#
# Primary design goals:
# 1) Fail-loud contract enforcement in strict mode (regulatory defensibility / auditability).
# 2) Stable, canonical output schema (order + dtypes) independent of input inference quirks.
# 3) DST policy is: fold extras into their base intervals with null-preserving semantics.
# 4) Determinism is required for partitioned Parquet writing, but global sorts are operationally
#    expensive at month scale. We therefore gate sorting behind sort_output.
#
# Operational note:
# - For month-scale validation in constrained environments (e.g., Docker devcontainer),
#   prefer sort_output=False and validate determinism separately on bounded samples or
#   in a higher-memory runtime.
# -------------------------------------------------------------------------------------------------


# Exact header match only (no IGNORECASE).
_INTERVAL_COL_RE = re.compile(r"^INTERVAL_HR(?P<hhmm>\d{4})_ENERGY_QTY$")

# Standard 48 end-times: 0030...2400 (0000 absent) at 30-min cadence.
# Expressed as minutes since midnight for simple set arithmetic.
_STANDARD_END_MINUTES: set[int] = set(range(30, 1441, 30))

# DST extras appear as end-times 24:30 and 25:00 (minutes 1470, 1500).
_DST_EXTRA_END_MINUTES: set[int] = {1470, 1500}  # 2430, 2500

# DST fold-in map
_DST_FOLD_MAP = {
    "INTERVAL_HR2430_ENERGY_QTY": "INTERVAL_HR2330_ENERGY_QTY",
    "INTERVAL_HR2500_ENERGY_QTY": "INTERVAL_HR2400_ENERGY_QTY",
}

# Used historically for dtype enforcement; kept as an immutable set for defensive checks if needed.
_INTEGER_DTYPES = frozenset({
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
})


@dataclass(frozen=True)
class IntervalColSpec:
    """
    Parsed metadata for one interval column.

    Why keep this structure:
    - We want a durable, inspectable representation of the interval headers rather than
      relying on ad hoc string slicing scattered across the transform.
    - Using end_minutes/start_minutes (not just HHMM) makes validation and datetime math simpler.
    """

    colname: str
    hhmm: int
    end_minutes: int
    start_minutes: int


def _format_list_preview(x: list, max_items: int = 10) -> str:
    """
    Format a bounded preview of a python list for error messages.

    Why:
    - When strict validation fails, we want diagnostic information without dumping huge values.
    - This keeps exceptions readable in CI logs and in interactive debugging sessions.
    """
    if len(x) <= max_items:
        return str(x)
    return str(x[:max_items])[:-1] + f", ...] (n={len(x)})"


def _require_columns_from_names(schema_names: set[str], required: Iterable[str]) -> None:
    """
    Fail-loud if required wide columns are missing.

    Why:
    - Missing required columns almost always indicates upstream schema drift, not something
      we should guess at or attempt to "repair" inside the transform.
    """
    missing = [c for c in required if c not in schema_names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _enforce_total_columns_59_from_names(schema_names_in_order: Sequence[str]) -> None:
    """
    Enforce the "59 columns total" wide schema contract in strict mode.

    Why:
    - The source format is treated as a contract. Silent acceptance of schema changes
      is a common root cause of hard-to-debug downstream failures.
    - This is deliberately an exact count check, not "at least" or "contains".
    """
    if len(schema_names_in_order) != 59:
        cols = list(schema_names_in_order)
        raise ValueError(
            "Contract violation: expected exactly 59 columns in the wide CSV schema.\n"
            f"- observed_n_columns={len(cols)}\n"
            f"- first_20_columns={cols[:20]}\n"
        )


def _parse_interval_specs_from_columns(columns: Sequence[str]) -> list[IntervalColSpec]:
    """
    Parse all interval columns (including possible DST extras) from exact headers.

    Why:
    - Header parsing is the authoritative place to enforce interval-label invariants.
    - We do not attempt to normalize invalid HHMMs; invalid headers are treated as data contract
      violations and should fail loudly.
    """
    specs: list[IntervalColSpec] = []
    for c in columns:
        m = _INTERVAL_COL_RE.match(c)
        if not m:
            continue

        hhmm = int(m.group("hhmm"))
        hh = hhmm // 100
        mm = hhmm % 100

        # Reject invalid minutes and invalid hour range.
        if mm not in (0, 30):
            raise ValueError(f"Contract violation: invalid interval column minutes (expected 00/30).\n- column={c}\n")
        if hh < 0 or hh > 25:
            raise ValueError(f"Contract violation: invalid interval column hour (expected 00..25).\n- column={c}\n")

        # Authoritative 0000 rejection (locked contract).
        # Why: the dataset is defined in terms of end-times 0030..2400; 0000 must not appear.
        if hhmm == 0:
            raise ValueError(
                f"Contract violation: found an interval ending at 0000 (HHMM=0000). Do not guess.\n- column={c}\n"
            )

        end_minutes = 60 * hh + mm
        start_minutes = end_minutes - 30

        # Defensive redundancy. This should be unreachable given the explicit 0000 rejection,
        # but it provides additional protection against malformed headers.
        if start_minutes < 0:
            raise ValueError(
                "Contract violation: interval implies start_minutes < 0 (likely 0000), which must not exist.\n"
                f"- column={c}\n"
            )

        specs.append(
            IntervalColSpec(
                colname=c,
                hhmm=hhmm,
                end_minutes=end_minutes,
                start_minutes=start_minutes,
            )
        )

    specs.sort(key=lambda s: (s.end_minutes, s.colname))
    return specs


def _validate_interval_set(interval_specs: Sequence[IntervalColSpec], *, strict: bool) -> None:
    """
    Validate the observed end-time set against locked contract requirements.

    Why:
    - Most downstream correctness depends on having exactly the expected interval grid.
      Missing or unexpected interval headers cannot be fixed reliably downstream.
    - We validate in terms of end-minutes, which is robust to header ordering and avoids
      string-based comparisons.
    """
    observed_end = {s.end_minutes for s in interval_specs}

    if strict:
        missing_standard = sorted(_STANDARD_END_MINUTES - observed_end)
        if missing_standard:
            raise ValueError(
                "Interval HHMM set missing standard end-minutes (expected 48 columns).\n"
                f"- missing_end_minutes={missing_standard}\n"
            )

        allowed = set(_STANDARD_END_MINUTES) | set(_DST_EXTRA_END_MINUTES)
        unexpected = sorted(observed_end - allowed)
        if unexpected:
            raise ValueError(
                "Interval HHMM set has unexpected end-minutes (allowed extras only 1470/1500).\n"
                f"- unexpected_end_minutes={unexpected}\n"
            )

        observed_standard = observed_end & _STANDARD_END_MINUTES
        if len(observed_standard) != 48:
            raise ValueError(
                "Contract violation: standard interval end-times are not exactly 48 distinct values.\n"
                f"- observed_standard_count={len(observed_standard)}\n"
            )


def _validate_interval_length_1800_lf(lf: pl.LazyFrame) -> None:
    """
    Contract check: INTERVAL_LENGTH must represent constant 1800 seconds (fail-loud).

    Why keep this validation even though INTERVAL_LENGTH is dropped from output:
    - It's assumed in the datetime semantics. If intervals aren't 30 minutes,
     the entire long representation becomes invalid.
    - In practice, S3 scans often infer INTERVAL_LENGTH as String. The contract is the *value*,
      not the storage dtype, so we accept either as long as it parses to 1800 everywhere.
    """
    col = "INTERVAL_LENGTH"
    schema = lf.collect_schema()
    if col not in schema.names():
        raise ValueError("Missing required column: INTERVAL_LENGTH")

    # Accept either integer-typed or string-typed inputs; enforce that the value parses to 1800.
    # We use strict=False cast to avoid blowing up on benign string representations like "1800".
    il_int = pl.col(col).cast(pl.Int32, strict=False)
    invalid = il_int.is_null() | (il_int != pl.lit(1800, dtype=pl.Int32))

    any_invalid = bool(lf.select(invalid.any()).collect().item())
    if not any_invalid:
        return

    # Provide a small raw sample of offending values for debugging (bounded to 30).
    bad_vals = lf.filter(invalid).select(pl.col(col).cast(pl.Utf8).unique().head(30)).collect().to_series().to_list()
    raise ValueError(
        "INTERVAL_LENGTH contract violation: values must be 1800 seconds (string or integer accepted).\n"
        f"- raw_values_sample={_format_list_preview(bad_vals, max_items=30)}\n"
    )


def _validate_reading_date_parses_strict_lf(lf: pl.LazyFrame, *, colname: str) -> None:
    """
    Locked contract: INTERVAL_READING_DATE parses with %m/%d/%Y only (fail-loud).

    Why:
    - Date parsing ambiguity is a classic source of silent data corruption (e.g., DD/MM vs MM/DD).
    - We do not accept "best effort" parsing here; strictness is intentional.
    """
    parsed = pl.col(colname).cast(pl.Utf8).str.strptime(pl.Date, format="%m/%d/%Y", strict=False)
    bad_mask = parsed.is_null() & pl.col(colname).is_not_null()
    any_bad = bool(lf.select(bad_mask.any()).collect().item())
    if not any_bad:
        return

    bad_vals = lf.filter(bad_mask).select(pl.col(colname).unique().head(30)).collect().to_series().to_list()
    raise ValueError(
        f"Failed to parse {colname} into Date for some rows using %m/%d/%Y.\n"
        f"- raw_values_failed_parse_sample={_format_list_preview(bad_vals, max_items=30)}\n"
    )


def _fold_in_preserve_nulls(base: pl.Expr, extra: pl.Expr) -> pl.Expr:
    """
    Policy:
      - HR2330 := HR2330 + HR2430
      - HR2400 := HR2400 + HR2500
      - Drop extras after fold-in.

    Null semantics are important:
      - If both base and extra are null, output must remain null (unknown).
      - Otherwise treat null as 0.0 for summation.

    Why:
    - This reflects how DST extras behave operationally: an extra interval is additive if present,
      but we must not turn a fully-missing pair into a synthetic 0.
    """
    base_f = base.cast(pl.Float64, strict=False)
    extra_f = extra.cast(pl.Float64, strict=False)
    return (
        pl.when(base_f.is_null() & extra_f.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(base_f.fill_null(0.0) + extra_f.fill_null(0.0))
    )


def transform_wide_to_long_lf(
    lf: pl.LazyFrame,
    *,
    strict: bool = True,
    sort_output: bool = True,
) -> pl.LazyFrame:
    """
    Wide CSV -> Long (canonical) LazyFrame transform (transform-only; no writing).

    This function is intentionally "pure transform":
    - It does not read/write files directly.
    - It does not manage batching.
    - It does not choose execution resources.
    Those concerns belong to the driver/orchestrator layer.

    Determinism:
    - sort_output=True enforces deterministic global ordering on
      (zip_code, account_identifier, datetime).
    - Month-scale validation in constrained environments should typically use
      sort_output=False and validate determinism separately on bounded samples.

    Final output schema (exact order + dtypes):
      1) zip_code: Utf8
      2) delivery_service_class: Categorical
      3) delivery_service_name: Categorical
      4) account_identifier: Utf8
      5) datetime: Datetime(us)
      6) energy_kwh: Float64
      7) plc_value: Float64
      8) nspl_value: Float64
      9) year: Int32
      10) month: Int8
    """
    required = [
        "ZIP_CODE",
        "DELIVERY_SERVICE_CLASS",
        "DELIVERY_SERVICE_NAME",
        "ACCOUNT_IDENTIFIER",
        "INTERVAL_READING_DATE",
        "INTERVAL_LENGTH",
        "PLC_VALUE",
        "NSPL_VALUE",
    ]

    # Collecting schema is metadata-only and does not scan data. We use it to make
    # validation decisions without triggering a full execution.
    schema = lf.collect_schema()
    schema_cols_in_order = schema.names()
    schema_names = set(schema_cols_in_order)

    _require_columns_from_names(schema_names, required)

    if strict:
        _enforce_total_columns_59_from_names(schema_cols_in_order)

    interval_specs_all = _parse_interval_specs_from_columns(schema_cols_in_order)
    if not interval_specs_all:
        raise ValueError("Contract violation: no interval columns found matching ^INTERVAL_HR\\d{4}_ENERGY_QTY$.\n")

    _validate_interval_set(interval_specs_all, strict=strict)

    if strict:
        _validate_interval_length_1800_lf(lf)
        _validate_reading_date_parses_strict_lf(lf, colname="INTERVAL_READING_DATE")

    # Derive the canonical "standard" interval columns from the observed schema. We do not
    # hardcode the headers to avoid dependence on input ordering; strict mode ensures the set.
    standard_specs = [s for s in interval_specs_all if s.end_minutes in _STANDARD_END_MINUTES]
    standard_cols = [s.colname for s in standard_specs]

    if strict and len(standard_cols) != 48:
        raise ValueError(
            "Contract violation: expected exactly 48 standard interval columns.\n"
            f"- observed_n_standard_cols={len(standard_cols)}\n"
        )

    # Fail-loud if fold targets missing. We must always have base columns 2330 and 2400
    # since DST fold-in adds into them.
    if "INTERVAL_HR2330_ENERGY_QTY" not in schema_names or "INTERVAL_HR2400_ENERGY_QTY" not in schema_names:
        raise ValueError("Contract violation: missing required standard columns HR2330 or HR2400.\n")

    # Parse date as Date (not Datetime) first; this keeps semantics explicit and avoids
    # timezone ambiguity. We later cast to Datetime(us) for interval math.
    reading_date_expr = (
        pl.col("INTERVAL_READING_DATE")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, format="%m/%d/%Y", strict=True)
        .alias("interval_reading_date")
    )

    # Project early to minimize memory pressure:
    # - keep only identifier columns + PLC/NSPL + reading_date + interval columns
    # - drop filler columns and any other wide fields not needed for the canonical long output
    dst_extra_cols = [extra for extra in _DST_FOLD_MAP if extra in schema_names]
    wide = lf.select([
        pl.col("ZIP_CODE").cast(pl.Utf8).alias("zip_code"),
        pl.col("DELIVERY_SERVICE_CLASS").cast(pl.Categorical).alias("delivery_service_class"),
        pl.col("DELIVERY_SERVICE_NAME").cast(pl.Categorical).alias("delivery_service_name"),
        pl.col("ACCOUNT_IDENTIFIER").cast(pl.Utf8).alias("account_identifier"),
        pl.col("PLC_VALUE").cast(pl.Float64, strict=False).alias("plc_value"),
        pl.col("NSPL_VALUE").cast(pl.Float64, strict=False).alias("nspl_value"),
        reading_date_expr,
        *[pl.col(c).cast(pl.Float64, strict=False) for c in standard_cols],
        *[pl.col(c).cast(pl.Float64, strict=False) for c in dst_extra_cols],
    ])

    # Apply DST Option B fold-in via mapping, then drop the extra columns.
    fold_exprs: list[pl.Expr] = []
    for extra_col, base_col in _DST_FOLD_MAP.items():
        if extra_col in schema_names:
            fold_exprs.append(_fold_in_preserve_nulls(pl.col(base_col), pl.col(extra_col)).alias(base_col))
    if fold_exprs:
        wide = wide.with_columns(fold_exprs)

    if dst_extra_cols:
        wide = wide.drop(dst_extra_cols)

    # id_vars define the "identity" columns that are repeated for each unpivoted interval.
    # interval_reading_date is kept only until we compute datetime; it is not part of final output.
    id_vars = [
        "zip_code",
        "delivery_service_class",
        "delivery_service_name",
        "account_identifier",
        "plc_value",
        "nspl_value",
        "interval_reading_date",
    ]

    # Unpivot produces one row per (id_vars, interval_col). We immediately cast energy_kwh
    # to Float64 to enforce canonical dtype regardless of upstream inference.
    long = wide.unpivot(
        index=id_vars,
        on=standard_cols,
        variable_name="interval_col",
        value_name="energy_kwh",
    ).with_columns(pl.col("energy_kwh").cast(pl.Float64, strict=False))

    # Extract end-time HHMM from the interval column label. This is intentionally strict:
    # interval headers are part of the upstream contract; if they don't match, we should fail.
    long = long.with_columns(
        pl.col("interval_col")
        .str.extract(r"^INTERVAL_HR(\d{4})_ENERGY_QTY$", 1)
        .cast(pl.Int32, strict=True)
        .alias("hhmm")
    ).with_columns((((pl.col("hhmm") // 100) * 60) + (pl.col("hhmm") % 100)).alias("end_minutes"))

    if strict:
        # After unpivot, ensure only standard end-times remain.
        allowed = sorted(_STANDARD_END_MINUTES)
        any_bad_end = bool(long.select((~pl.col("end_minutes").is_in(allowed)).any()).collect().item())
        if any_bad_end:
            bad_cols = (
                long.filter(~pl.col("end_minutes").is_in(allowed))
                .select(pl.col("interval_col").unique().head(30))
                .collect()
                .to_series()
                .to_list()
            )
            raise ValueError(
                "Contract violation: unexpected interval columns appeared after unpivot.\n"
                f"- unexpected_interval_cols_sample={_format_list_preview(bad_cols, max_items=30)}\n"
            )

    # datetime = interval START time:
    # - Input labels are end-times (e.g., HR0030 ends at 00:30).
    # - We subtract 30 minutes to get the interval start.
    # - HR2400 therefore maps to 23:30 same day (no rollover), matching the locked semantics.
    long = long.with_columns(
        (
            pl.col("interval_reading_date").cast(pl.Datetime("us"))
            + pl.duration(minutes=pl.col("end_minutes").cast(pl.Int64) - pl.lit(30))
        ).alias("datetime")
    )

    # Derived partition columns. These must come from datetime (not from INTERVAL_READING_DATE),
    # because datetime semantics are the canonical time representation.
    long = long.with_columns([
        pl.col("datetime").dt.year().cast(pl.Int32).alias("year"),
        pl.col("datetime").dt.month().cast(pl.Int8).alias("month"),
    ])

    # Drop helper columns promptly to reduce downstream memory footprint.
    long = long.drop(["interval_col", "hhmm", "end_minutes"])

    if sort_output:
        # Sorting is intentionally optional:
        # - required for deterministic output in write paths
        # - avoided in month-scale validation in constrained environments
        long = long.sort(["zip_code", "account_identifier", "datetime"])

    # Authoritative final projection:
    # - enforces schema order and dtypes
    # - ensures interval_reading_date is not in the final output
    return long.select([
        pl.col("zip_code").cast(pl.Utf8),
        pl.col("delivery_service_class").cast(pl.Categorical),
        pl.col("delivery_service_name").cast(pl.Categorical),
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("datetime").cast(pl.Datetime("us")),
        pl.col("energy_kwh").cast(pl.Float64, strict=False),
        pl.col("plc_value").cast(pl.Float64, strict=False),
        pl.col("nspl_value").cast(pl.Float64, strict=False),
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int8),
    ])


def transform_wide_to_long(
    df: pl.DataFrame,
    *,
    strict: bool = True,
    sort_output: bool = True,
) -> pl.DataFrame:
    """
    Backward-compatible DataFrame API wrapper.

    Why keep this:
    - Some call sites prefer an eager DataFrame API (e.g., unit tests, small local files).
    - We keep the LazyFrame transform as the source of truth and collect at the boundary.
    """
    return transform_wide_to_long_lf(df.lazy(), strict=strict, sort_output=sort_output).collect()
