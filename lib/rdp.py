"""Shared reporting utilities and rate-design-platform fetch helpers."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


DEFAULT_RESSTOCK_DATA_ROOT_NAMES = ("resstock", "ResStock")
DIST_PARAM_KEYS = (
    "annual_future_distr_costs",
    "distr_peak_hrs",
    "nc_ratio_baseline",
)


def repo_root() -> Path:
    """Return the repository root for the current checkout."""
    return Path(__file__).resolve().parent.parent


def _choose_existing_path(candidates: Iterable[Path], description: str) -> Path:
    checked = []
    for candidate in candidates:
        checked.append(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {description} found. Checked: " + ", ".join(str(path) for path in checked))


def _normalize_upgrade(upgrade: str | int) -> str:
    if isinstance(upgrade, int):
        return f"{upgrade:02d}"
    if upgrade.isdigit():
        return f"{int(upgrade):02d}"
    return upgrade


def resolve_resstock_data_root(
    candidates: Sequence[Path] | None = None,
    root: Path | None = None,
) -> Path:
    """Return the first existing local ResStock data root under the repo."""
    root = root or repo_root()
    candidates = list(candidates or [])
    if not candidates:
        candidates = [root / "data" / name for name in DEFAULT_RESSTOCK_DATA_ROOT_NAMES]
    return _choose_existing_path(candidates, "local ResStock data root")


def resolve_resstock_release_root(
    release: str,
    data_root: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Return the existing directory for a ResStock release."""
    base = data_root or resolve_resstock_data_root(root=root)
    return _choose_existing_path([base / release], f"ResStock release directory for {release!r}")


def resolve_resstock_metadata_dir(
    release: str,
    state: str,
    upgrade: str | int,
    data_root: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Return the existing metadata directory for a release/state/upgrade."""
    release_root = resolve_resstock_release_root(release, data_root=data_root, root=root)
    upgrade_str = _normalize_upgrade(upgrade)
    candidates = [
        release_root / "metadata" / f"state={state}" / f"upgrade={upgrade_str}",
        release_root / "metadata" / state / f"up{upgrade_str}",
        release_root / "metadata" / state / f"upgrade={upgrade_str}",
        release_root / "metadata" / f"state={state}" / f"up{upgrade_str}",
    ]
    return _choose_existing_path(
        candidates,
        f"ResStock metadata directory for release={release!r}, state={state!r}, upgrade={upgrade_str!r}",
    )


def resolve_resstock_metadata_file(base: Path) -> Path:
    """Return the preferred metadata parquet file from a metadata directory."""
    candidates = [
        base / "metadata-sb.parquet",
        base / "metadata-sb-with-utilities.parquet",
        base / "metadata.parquet",
    ]
    return _choose_existing_path(candidates, f"metadata parquet in {base}")


def resolve_resstock_hourly_loads_dir(
    release: str,
    state: str | None = None,
    upgrade: str | int | None = None,
    data_root: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Return the best available hourly-load directory for a ResStock release."""
    release_root = resolve_resstock_release_root(release, data_root=data_root, root=root)
    hourly_root = release_root / "load_curve_hourly"
    candidates = []

    if state is not None and upgrade is not None:
        upgrade_str = _normalize_upgrade(upgrade)
        candidates.extend([
            hourly_root / f"state={state}" / f"upgrade={upgrade_str}",
            hourly_root / state / f"up{upgrade_str}",
            hourly_root / state / f"upgrade={upgrade_str}",
            hourly_root / f"state={state}" / f"up{upgrade_str}",
        ])

    candidates.append(hourly_root)
    return _choose_existing_path(
        candidates,
        f"ResStock hourly-load directory for release={release!r}",
    )


def build_hp_flag_expr(schema_cols: list[str]) -> pl.Expr:
    """Return a Polars expression that identifies heat-pump homes."""
    import polars as pl

    if "postprocess_group.has_hp" in schema_cols:
        return pl.col("postprocess_group.has_hp")

    if "in.hvac_heating_and_fuel_type" in schema_cols:
        return pl.col("in.hvac_heating_and_fuel_type").str.to_lowercase().str.contains("hp").fill_null(False)

    if "in.hvac_heating_type_and_fuel" in schema_cols:
        return pl.col("in.hvac_heating_type_and_fuel").str.to_lowercase().str.contains("hp").fill_null(False)

    raise ValueError(
        "Could not determine HP flag from metadata columns. "
        "Expected `postprocess_group.has_hp`, `in.hvac_heating_and_fuel_type`, "
        "or `in.hvac_heating_type_and_fuel`."
    )


def resolve_heating_type_column(schema_cols: list[str]) -> str:
    """Return the metadata column that stores the home heating type."""
    if "in.hvac_heating_and_fuel_type" in schema_cols:
        return "in.hvac_heating_and_fuel_type"
    if "in.hvac_heating_type_and_fuel" in schema_cols:
        return "in.hvac_heating_type_and_fuel"
    raise ValueError(
        "Could not determine heating-type column from metadata. "
        "Expected `in.hvac_heating_and_fuel_type` or `in.hvac_heating_type_and_fuel`."
    )


def build_hp_group_expr(
    hp_flag_col: str = "hp_flag",
    heating_type_col: str = "heating_type",
) -> pl.Expr:
    """Return a Polars expression that buckets homes into HP/non-HP groups."""
    import polars as pl

    return (
        pl.when(pl.col(hp_flag_col))
        .then(pl.lit("HP"))
        .when(pl.col(heating_type_col).str.to_lowercase().str.contains("electric"))
        .then(pl.lit("Electric (non-HP heating)"))
        .otherwise(pl.lit("Non-HP"))
    )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into ``(bucket, key)``."""
    without_prefix = uri[len("s3://") :]
    bucket, _, key = without_prefix.partition("/")
    return bucket, key


def _read_s3_bytes(s3_uri: str) -> bytes:
    """Read raw bytes from an S3 URI using boto3."""
    import boto3

    bucket, key = _parse_s3_uri(s3_uri)
    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def _to_polars_frame(frame) -> pl.DataFrame:
    """Convert a pandas or Polars DataFrame into a Polars DataFrame."""
    import polars as pl

    if isinstance(frame, pl.DataFrame):
        return frame

    if frame.__class__.__module__.startswith("pandas"):
        return pl.from_pandas(frame)

    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _reset_index_if_needed(frame: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
    """Reset index for pandas-like frames that expose ``reset_index``."""
    reset_index = getattr(frame, "reset_index", None)
    if callable(reset_index):
        return reset_index()
    return frame


def read_s3_csv(s3_uri: str, **kwargs) -> pl.DataFrame:
    """Read a CSV from an S3 URI into Polars."""
    import polars as pl

    return pl.read_csv(io.BytesIO(_read_s3_bytes(s3_uri)), **kwargs)


def read_s3_json(s3_uri: str) -> dict:
    """Read a JSON object from an S3 URI using boto3 (no s3fs required)."""
    import boto3

    bucket, key = _parse_s3_uri(s3_uri)
    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def find_latest_run_dir(run_base: str, run_name: str) -> str:
    """Return the S3 URI of the most recent output directory matching ``run_name``."""
    import boto3

    bucket, prefix = _parse_s3_uri(run_base)
    prefix = prefix.rstrip("/") + "/"

    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    matching: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            dir_prefix = entry["Prefix"]
            dir_name = dir_prefix[len(prefix) :].rstrip("/")
            if dir_name.endswith(f"_{run_name}"):
                matching.append(dir_name)

    if not matching:
        raise FileNotFoundError(
            f"No output directory matching run_name={run_name!r} found under {run_base}. "
            "Re-run the scenario to generate outputs."
        )

    latest = sorted(matching)[-1]
    return f"{run_base.rstrip('/')}/{latest}"


def resolve_dist_params(defaults: dict, candidates: list[Path] | None = None) -> dict:
    """Return distribution-cost parameters from the first existing JSON candidate."""
    candidates = candidates or []
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            loaded = json.load(f)
        return {key: loaded[key] for key in DIST_PARAM_KEYS}
    return defaults


def choose_latest_run(run_root: Path) -> Path:
    """Return the lexicographically latest run directory under ``run_root``."""
    runs = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run directories found in {run_root}")
    return runs[-1]


def load_dist_mc_from_run(run_dir: Path | str) -> pl.DataFrame:
    """Load ``distribution_marginal_costs.csv`` into a timezone-aware Polars DataFrame."""
    import polars as pl

    csv_path = f"{str(run_dir).rstrip('/')}/distribution_marginal_costs.csv"
    if csv_path.startswith("s3://"):
        df = read_s3_csv(csv_path, try_parse_dates=True)
    else:
        df = pl.read_csv(csv_path, try_parse_dates=True)

    value_col = df.columns[1]
    return force_timezone_est_polars(
        df.select(
            pl.col("time"),
            pl.col(value_col).alias("Marginal Distribution Costs ($/kWh)"),
        ),
        timestamp_col="time",
    )


def load_cambium_from_parquet_s3(s3_uri: str, target_year: int) -> pl.DataFrame:
    """Load Cambium marginal costs from an S3 parquet file into Polars."""
    import polars as pl

    frame = cast(
        pl.DataFrame,
        pl.scan_parquet(s3_uri)
        .filter(pl.col("t") == target_year)
        .select(["timestamp_local", "energy_cost_enduse", "capacity_cost_enduse"])
        .rename({
            "timestamp_local": "time",
            "energy_cost_enduse": "Marginal Energy Costs ($/kWh)",
            "capacity_cost_enduse": "Marginal Capacity Costs ($/kWh)",
        })
        .with_columns(
            (pl.col("Marginal Energy Costs ($/kWh)") / 1000.0).alias("Marginal Energy Costs ($/kWh)"),
            (pl.col("Marginal Capacity Costs ($/kWh)") / 1000.0).alias("Marginal Capacity Costs ($/kWh)"),
        )
        .collect(),
    )
    return force_timezone_est_polars(frame, timestamp_col="time")


def force_timezone_est_polars(
    frame: pl.DataFrame,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Ensure a Polars datetime column is timezone-aware and converted to EST."""
    import polars as pl

    if timestamp_col not in frame.columns:
        raise ValueError(f"{timestamp_col} not found in frame columns")

    dtype = frame.schema[timestamp_col]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        expr = pl.col(timestamp_col).dt.convert_time_zone("EST")
    else:
        expr = pl.col(timestamp_col).cast(pl.Datetime, strict=False).dt.replace_time_zone("EST")
    return frame.with_columns(expr.alias(timestamp_col))


def build_bldg_id_to_load_filepath(
    path_resstock_loads: Path,
    building_ids: list[int],
) -> dict[int, Path]:
    """Map requested building IDs to their ResStock load parquet paths."""
    bldg_set = {int(i) for i in building_ids}
    mapping: dict[int, Path] = {}
    for parquet_file in path_resstock_loads.rglob("*.parquet"):
        try:
            bldg_id = int(parquet_file.stem.split("-")[0])
        except ValueError:
            continue
        if bldg_id in bldg_set:
            mapping[bldg_id] = parquet_file
    missing = bldg_set - set(mapping)
    if missing:
        print(f"Warning: missing load files for {len(missing)} building IDs")
    return mapping


_CROSS_SUBSIDY_WEIGHTED_COLS: dict[str, str] = {
    "BAT_vol": "BAT_vol_weighted_avg",
    "BAT_peak": "BAT_peak_weighted_avg",
    "BAT_percustomer": "BAT_percustomer_weighted_avg",
    "customer_level_residual_share_volumetric": "residual_vol_weighted_avg",
    "customer_level_residual_share_peak": "residual_peak_weighted_avg",
    "customer_level_residual_share_percustomer": "residual_percustomer_weighted_avg",
    "Annual": "Annual_bill_weighted_avg",
    "customer_level_economic_burden": "Economic_burden_weighted_avg",
}


def summarize_cross_subsidy(cross: pd.DataFrame | pl.DataFrame, metadata: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    """Compute weighted cross-subsidy metrics for HP and Non-HP groups."""
    import polars as pl

    cross_pl = _to_polars_frame(cross)
    metadata_pl = _to_polars_frame(metadata)

    merged = cross_pl.join(
        metadata_pl.select(["bldg_id", "postprocess_group.has_hp", "weight"]),
        on=["bldg_id", "weight"],
        how="left",
    )

    agg_exprs = [pl.col("weight").sum().alias("customers_weighted")]
    agg_exprs.extend(
        ((pl.col(source_col) * pl.col("weight")).sum() / pl.col("weight").sum()).alias(output_col)
        for source_col, output_col in _CROSS_SUBSIDY_WEIGHTED_COLS.items()
    )

    return (
        merged.group_by("postprocess_group.has_hp")
        .agg(agg_exprs)
        .with_columns(
            pl.when(pl.col("postprocess_group.has_hp")).then(pl.lit("HP")).otherwise(pl.lit("Non-HP")).alias("group")
        )
        .select("postprocess_group.has_hp", "customers_weighted", "group", *_CROSS_SUBSIDY_WEIGHTED_COLS.values())
        .sort("postprocess_group.has_hp", descending=True)
    )


def summarize_cross_subsidy_by_heating_type(
    cross: pd.DataFrame | pl.DataFrame,
    metadata: pd.DataFrame | pl.DataFrame,
) -> pl.DataFrame:
    """Compute weighted cross-subsidy metrics grouped by heating type."""
    import polars as pl

    cross_pl = _to_polars_frame(cross)
    metadata_pl = _to_polars_frame(metadata)

    agg_exprs = [pl.col("weight").sum().alias("customers_weighted")]
    agg_exprs.extend(
        ((pl.col(source_col) * pl.col("weight")).sum() / pl.col("weight").sum()).alias(output_col)
        for source_col, output_col in _CROSS_SUBSIDY_WEIGHTED_COLS.items()
    )

    return (
        cross_pl.join(
            metadata_pl.select(["bldg_id", "postprocess_group.heating_type", "weight"]),
            on=["bldg_id", "weight"],
            how="left",
        )
        .group_by("postprocess_group.heating_type")
        .agg(agg_exprs)
        .with_columns(pl.col("postprocess_group.heating_type").cast(pl.String).alias("group"))
        .select("postprocess_group.heating_type", "customers_weighted", "group", *_CROSS_SUBSIDY_WEIGHTED_COLS.values())
        .sort("customers_weighted", descending=True)
    )


def build_hourly_group_loads(
    raw_load_elec: pd.DataFrame | pl.DataFrame,
    metadata: pd.DataFrame | pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate weighted hourly electricity load by HP flag and total."""
    import polars as pl

    raw_prepared = _reset_index_if_needed(raw_load_elec)
    raw_pl = _to_polars_frame(raw_prepared).select("time", "bldg_id", "electricity_net")
    metadata_pl = _to_polars_frame(metadata).select("bldg_id", "postprocess_group.has_hp", "weight")

    return (
        raw_pl.join(metadata_pl, on="bldg_id", how="left")
        .with_columns((pl.col("electricity_net") * pl.col("weight")).alias("weighted_load_kwh"))
        .group_by("time")
        .agg(
            pl.when(pl.col("postprocess_group.has_hp"))
            .then(pl.col("weighted_load_kwh"))
            .otherwise(0.0)
            .sum()
            .alias("hp_load_kwh"),
            pl.when(~pl.col("postprocess_group.has_hp").fill_null(False))
            .then(pl.col("weighted_load_kwh"))
            .otherwise(0.0)
            .sum()
            .alias("non_hp_load_kwh"),
        )
        .sort("time")
        .with_columns((pl.col("non_hp_load_kwh") + pl.col("hp_load_kwh")).alias("total_load_kwh"))
    )


def build_hourly_heating_type_loads(
    raw_load_elec: pd.DataFrame | pl.DataFrame,
    metadata: pd.DataFrame | pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate weighted hourly electricity load by heating type."""
    import polars as pl

    raw_prepared = _reset_index_if_needed(raw_load_elec)
    raw_pl = _to_polars_frame(raw_prepared).select("time", "bldg_id", "electricity_net")
    metadata_pl = _to_polars_frame(metadata).select("bldg_id", "postprocess_group.heating_type", "weight")

    return (
        raw_pl.join(metadata_pl, on="bldg_id", how="left")
        .with_columns((pl.col("electricity_net") * pl.col("weight")).alias("weighted_load_kwh"))
        .group_by(["time", "postprocess_group.heating_type"])
        .agg(pl.col("weighted_load_kwh").sum().alias("load_kwh"))
        .rename({"postprocess_group.heating_type": "heating_type"})
        .sort(["time", "heating_type"])
    )


def build_cross_components(cross_summary: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    """Build benchmark component contributions for charting cross-subsidy impacts."""
    import polars as pl

    cross_summary_pl = _to_polars_frame(cross_summary)
    component_labels = {
        "BAT_vol_weighted_avg": "Volumetric benchmark",
        "BAT_peak_weighted_avg": "Peak benchmark",
        "BAT_percustomer_weighted_avg": "Per-customer benchmark",
    }

    return pl.concat(
        [
            cross_summary_pl.select(
                "group",
                "customers_weighted",
                pl.lit(component).alias("component"),
                pl.col(component).alias("weighted_avg_bat_usd_per_customer_year"),
            ).with_columns(
                pl.lit(label).alias("component_label"),
                (pl.col("weighted_avg_bat_usd_per_customer_year") * pl.col("customers_weighted") / 1e6).alias(
                    "component_transfer_total_musd_per_year"
                ),
            )
            for component, label in component_labels.items()
        ],
        how="vertical",
    )


def summarize_positive_distribution_hours(
    hourly: pd.DataFrame | pl.DataFrame,
    customer_count_map: dict[str, float],
) -> pl.DataFrame:
    """Summarize per-customer load behavior in positive marginal distribution-cost hours."""
    import polars as pl

    hourly_pl = _to_polars_frame(hourly)
    rows = []
    for col, label in [("hp_load_kwh", "HP"), ("non_hp_load_kwh", "Non-HP")]:
        customer_count = float(customer_count_map[label])
        stats = hourly_pl.select(
            pl.col(col).sum().alias("annual"),
            pl.when(pl.col("mdc_positive")).then(pl.col(col)).otherwise(0.0).sum().alias("positive"),
            pl.when(pl.col("mdc_positive")).then(pl.col(col)).otherwise(None).mean().alias("positive_mean"),
            pl.when(~pl.col("mdc_positive")).then(pl.col(col)).otherwise(None).mean().alias("zero_mean"),
        ).row(0, named=True)
        annual = float(stats["annual"])
        positive = float(stats["positive"])
        rows.append({
            "group": label,
            "weighted_customers": customer_count,
            "annual_load_mwh_per_customer": (annual / customer_count) / 1000,
            "positive_dist_cost_hours_load_mwh_per_customer": (positive / customer_count) / 1000,
            "share_of_annual_load_in_positive_dist_cost_hours": positive / annual,
            "avg_hourly_load_kwh_during_positive_dist_hours": (float(stats["positive_mean"]) / customer_count),
            "avg_hourly_load_kwh_during_zero_dist_hours": (float(stats["zero_mean"]) / customer_count),
        })
    return pl.DataFrame(rows)


def build_tariff_components(
    hourly: pd.DataFrame | pl.DataFrame,
    cross_summary: pd.DataFrame | pl.DataFrame,
    fixed_monthly: float,
    vol_rate: float,
) -> pl.DataFrame:
    """Compute annual fixed and volumetric charges collected by customer group."""
    import polars as pl

    hourly_pl = _to_polars_frame(hourly)
    cross_summary_pl = _to_polars_frame(cross_summary)

    group_load = pl.DataFrame({
        "group": ["Non-HP", "HP"],
        "annual_load_kwh": [
            hourly_pl.select(pl.col("non_hp_load_kwh").sum()).item(),
            hourly_pl.select(pl.col("hp_load_kwh").sum()).item(),
        ],
    })

    return group_load.join(
        cross_summary_pl.select(
            "group",
            pl.col("customers_weighted").alias("weighted_customers"),
        ),
        on="group",
        how="left",
    ).with_columns(
        (pl.lit(fixed_monthly * 12) * pl.col("weighted_customers")).alias("annual_fixed_charge_collected_usd"),
        (pl.lit(vol_rate) * pl.col("annual_load_kwh")).alias("annual_volumetric_charge_collected_usd"),
    )


def fetch_rdp_file(path: str, ref: str) -> str:
    """Fetch a file from rate-design-platform on GitHub; return contents as string.

    Uses the GitHub API with ``GITHUB_TOKEN`` when available (required for
    private repos), otherwise falls back to the public raw URL.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        url = f"https://api.github.com/repos/switchbox-data/rate-design-platform/contents/{path}?ref={ref}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        return base64.b64decode(data["content"]).decode()
    url = f"https://raw.githubusercontent.com/switchbox-data/rate-design-platform/{ref}/{path}"
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode()


def parse_urdb_json(content: str | bytes) -> dict:
    """Parse URDB tariff JSON (string or bytes) into a dict."""
    if isinstance(content, bytes):
        content = content.decode()
    return json.loads(content)
