"""
Build Block-Group load-shape features by routing anonymized usage files
through a ZIP+4 -> BG crosswalk (mock now, real later).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from .aws_loader import list_s3_files, process_single_csv
from .zip4_from_s3key import zip4_from_key

logger = logging.getLogger(__name__)

PEAK_START = 16  # 4 PM
PEAK_END = 21  # 9 PM
MIN_DAYS_PER_BG = 20


@dataclass
class ProcessingStats:
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    failed_file_keys: list[str] = field(default_factory=list)

    total_rows_raw: int = 0
    rows_with_zip4: int = 0
    rows_mapped_to_bg: int = 0
    rows_dropped_null_kwh: int = 0

    unique_zip4s_seen: int = 0
    unique_zip4s_mapped: int = 0
    unique_zip4s_unmapped: int = 0
    unique_bgs: int = 0

    unmapped_zip4s: set[str] = field(default_factory=set)

    bgs_below_min_days: int = 0
    bgs_above_min_days: int = 0

    def compute_rates(self) -> dict[str, float]:
        return {
            "file_success_rate": self.successful_files / max(self.total_files, 1),
            "zip4_mapping_rate": self.rows_mapped_to_bg / max(self.rows_with_zip4, 1),
            "row_retention_rate": self.rows_mapped_to_bg / max(self.total_rows_raw, 1),
            "bg_retention_rate": self.bgs_above_min_days / max(self.unique_bgs, 1),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unmapped_zip4s"] = list(self.unmapped_zip4s)[:100]
        d["rates"] = self.compute_rates()
        return d


def _add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.month().alias("month"),
        ((pl.col("datetime").dt.hour() >= PEAK_START) & (pl.col("datetime").dt.hour() < PEAK_END)).alias("is_peak"),
    ])


def _build_summary_features(bg_hourly: pl.DataFrame) -> pl.DataFrame:
    daily = bg_hourly.group_by(["bg_geoid", "date"]).agg([
        pl.sum("kwh").alias("kwh_day"),
        (pl.when(pl.col("is_peak")).then(pl.col("kwh")).otherwise(0.0)).sum().alias("kwh_peak"),
    ])

    feat = (
        daily.group_by("bg_geoid")
        .agg([
            pl.count().alias("n_days"),
            pl.mean("kwh_day").alias("mean_day"),
            pl.std("kwh_day").alias("std_day"),
            pl.mean("kwh_peak").alias("mean_peak"),
        ])
        .with_columns([
            pl.when(pl.col("mean_day") > 0).then(pl.col("std_day") / pl.col("mean_day")).otherwise(0.0).alias("cv_day"),
            pl.when(pl.col("mean_day") > 0)
            .then(pl.col("mean_peak") / pl.col("mean_day"))
            .otherwise(0.0)
            .alias("peak_share"),
        ])
        .drop(["std_day", "mean_peak"])
    )
    return feat


def _build_diurnal_features(bg_hourly: pl.DataFrame) -> pl.DataFrame:
    diurnal = (
        bg_hourly.group_by(["bg_geoid", "hour"])
        .agg(pl.mean("kwh").alias("kwh_hr"))
        .pivot(values="kwh_hr", index="bg_geoid", on="hour")
    )
    for h in range(24):
        src = str(h)
        dst = f"hr_{h:02d}"
        diurnal = diurnal.rename({src: dst}) if src in diurnal.columns else diurnal.with_columns(pl.lit(0.0).alias(dst))
    return diurnal


def _build_seasonal_features(bg_hourly: pl.DataFrame) -> pl.DataFrame:
    season = (
        bg_hourly.group_by(["bg_geoid", "month"])
        .agg(pl.mean("kwh").alias("kwh_mo"))
        .pivot(values="kwh_mo", index="bg_geoid", on="month")
    )
    for m in range(1, 13):
        src = str(m)
        dst = f"mo_{m:02d}"
        season = season.rename({src: dst}) if src in season.columns else season.with_columns(pl.lit(0.0).alias(dst))
    return season


def _build_zip4_lookup(
    crosswalk: pl.DataFrame,
) -> tuple[dict[str, str], pl.DataFrame | None]:
    """
    Return:
      - single_map: {(zip5,plus4) -> bg_geoid} for ZIP+4 that map to exactly one BG
      - multi_slice: DataFrame [zip5, plus4, bg_geoid, weight] for splits (or None)
    """
    need = {"zip5", "plus4", "bg_geoid"}
    if not need.issubset(set(crosswalk.columns)):
        raise ValueError  # TRY003

    if "weight" not in crosswalk.columns:
        crosswalk = crosswalk.with_columns(pl.lit(1.0).alias("weight"))

    multiplicity = (
        crosswalk.group_by(["zip5", "plus4"])
        .agg(pl.count().alias("_n"))
        .with_columns((pl.col("_n") > 1).alias("_is_multi"))
    )

    cw = crosswalk.join(multiplicity, on=["zip5", "plus4"], how="left")
    single = cw.filter(pl.col("_is_multi") == False)  # noqa: E712
    multi = cw.filter(pl.col("_is_multi") == True).select(  # noqa: E712
        ["zip5", "plus4", "bg_geoid", "weight"]
    )

    single_map = {f"{r['zip5']}-{r['plus4']}": r["bg_geoid"] for r in single.iter_rows(named=True)}
    multi_slice = None if multi.is_empty() else multi
    return single_map, multi_slice


def _bg_hourly_from_key(
    s3_key: str,
    bucket: str,
    single_map: dict[str, str],
    multi_slice: pl.DataFrame | None,
    stats: ProcessingStats,
) -> pl.DataFrame:
    """
    Process one S3 CSV into BG x timestamp kWh.
    """
    df = process_single_csv(s3_key, bucket=bucket)
    stats.total_rows_raw += df.height

    zip5, plus4 = zip4_from_key(s3_key)
    df = df.with_columns([pl.lit(zip5).alias("zip5"), pl.lit(plus4).alias("plus4")])
    stats.rows_with_zip4 += df.height

    key = f"{zip5}-{plus4}"
    if key in single_map:
        df = df.with_columns(pl.lit(single_map[key]).alias("bg_geoid"))
    else:
        if multi_slice is None:
            stats.unmapped_zip4s.add(key)
            return pl.DataFrame({"bg_geoid": [], "datetime": [], "kwh": []})
        df = df.join(multi_slice, on=["zip5", "plus4"], how="inner").with_columns(
            (pl.col("kwh") * pl.col("weight")).alias("kwh")
        )

    df_mapped = df.drop_nulls(subset=["bg_geoid"])
    stats.rows_mapped_to_bg += df_mapped.height

    df_valid = df_mapped.filter(pl.col("kwh").is_not_null() & pl.col("kwh").is_finite() & (pl.col("kwh") >= 0))
    stats.rows_dropped_null_kwh += df_mapped.height - df_valid.height

    return df_valid.group_by(["bg_geoid", "datetime"]).agg(pl.sum("kwh").alias("kwh"))


def build_bg_features_for_month(
    year_month: str,
    zip4_to_bg: pl.DataFrame,
    bucket: str,
    prefix: str,
    out_features_parquet: str,
    out_quality_json: str | None = None,
    max_files: int | None = None,
    min_days: int = MIN_DAYS_PER_BG,
    strict_month: bool = False,
) -> ProcessingStats:
    """
    Build BG-level features for a single month (YYYYMM).

    strict_month=True keeps only rows whose datetime.month == requested month.
    """
    keys = list_s3_files(year_month, bucket=bucket, prefix=prefix, max_files=max_files)
    if not keys:
        raise ValueError  # TRY003

    stats = ProcessingStats(total_files=len(keys))
    single_map, multi_slice = _build_zip4_lookup(zip4_to_bg)

    bg_hourly: pl.DataFrame | None = None

    for key in keys:
        try:
            df_chunk = _bg_hourly_from_key(key, bucket, single_map, multi_slice, stats)
            if strict_month and not df_chunk.is_empty():
                ym = int(year_month[-2:])
                df_chunk = df_chunk.filter(pl.col("datetime").dt.month() == ym)

            bg_hourly = (
                df_chunk
                if bg_hourly is None
                else (
                    pl.concat([bg_hourly, df_chunk], how="vertical")
                    .group_by(["bg_geoid", "datetime"])
                    .agg(pl.sum("kwh").alias("kwh"))
                )
            )
            stats.successful_files += 1
        except Exception:
            logger.exception("Failed on %s", key)
            stats.failed_files += 1
            stats.failed_file_keys.append(key)

    if bg_hourly is None or bg_hourly.is_empty():
        raise RuntimeError  # TRY003

    stats.unique_bgs = bg_hourly.select("bg_geoid").n_unique()
    bg_hourly = _add_time_features(bg_hourly)

    feat = _build_summary_features(bg_hourly)
    diurnal = _build_diurnal_features(bg_hourly)
    season = _build_seasonal_features(bg_hourly)

    feat_filt = feat.filter(pl.col("n_days") >= min_days)
    stats.bgs_below_min_days = feat.height - feat_filt.height
    stats.bgs_above_min_days = feat_filt.height

    features = feat_filt.join(diurnal, on="bg_geoid", how="inner").join(season, on="bg_geoid", how="inner")

    Path(out_features_parquet).parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(out_features_parquet)
    logger.info("Wrote %d BGs x %d features -> %s", *features.shape, out_features_parquet)
    quality_path = (
        Path(out_quality_json) if out_quality_json else Path(out_features_parquet).with_suffix(".quality.json")
    )
    quality_path.write_text(json.dumps(stats.to_dict(), indent=2))
    logger.info("Wrote quality -> %s", quality_path)
    logger.info("Wrote quality -> %s", out_quality_json)

    return stats


def build_bg_features_for_year(
    year: int,
    zip4_to_bg: pl.DataFrame,
    bucket: str,
    prefix: str,
    out_features_parquet: str,
    out_quality_json: str | None = None,
    max_files_per_month: int | None = None,
    min_days: int = MIN_DAYS_PER_BG,
) -> ProcessingStats:
    """
    Build BG features for a full year (12 months).
    """
    stats = ProcessingStats()
    single_map, multi_slice = _build_zip4_lookup(zip4_to_bg)
    bg_hourly: pl.DataFrame | None = None

    for m in range(1, 13):
        ym = f"{year}{m:02d}"
        keys = list_s3_files(ym, bucket=bucket, prefix=prefix, max_files=max_files_per_month)
        for key in keys:
            try:
                df_chunk = _bg_hourly_from_key(key, bucket, single_map, multi_slice, stats)
                bg_hourly = (
                    df_chunk
                    if bg_hourly is None
                    else (
                        pl.concat([bg_hourly, df_chunk], how="vertical")
                        .group_by(["bg_geoid", "datetime"])
                        .agg(pl.sum("kwh").alias("kwh"))
                    )
                )
                stats.successful_files += 1
            except Exception:
                logger.exception("Failed on %s", key)
                stats.failed_files += 1
                stats.failed_file_keys.append(key)

    if bg_hourly is None or bg_hourly.is_empty():
        raise RuntimeError  # TRY003

    stats.unique_bgs = bg_hourly.select("bg_geoid").n_unique()
    bg_hourly = _add_time_features(bg_hourly)

    feat = _build_summary_features(bg_hourly)
    diurnal = _build_diurnal_features(bg_hourly)
    season = _build_seasonal_features(bg_hourly)

    feat_filt = feat.filter(pl.col("n_days") >= min_days)
    stats.bgs_below_min_days = feat.height - feat_filt.height
    stats.bgs_above_min_days = feat_filt.height

    features = feat_filt.join(diurnal, on="bg_geoid", how="inner").join(season, on="bg_geoid", how="inner")

    Path(out_features_parquet).parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(out_features_parquet)
    logger.info("Wrote %d BGs x %d features -> %s", *features.shape, out_features_parquet)

    quality_path = (
        Path(out_quality_json) if out_quality_json else Path(out_features_parquet).with_suffix(".quality.json")
    )
    quality_path.write_text(json.dumps(stats.to_dict(), indent=2))
    logger.info("Wrote quality -> %s", quality_path)

    return stats


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="Build BG-level load-shape features")
    ap.add_argument("--mode", choices=["month", "year"], required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, help="1-12 (for --mode month)")
    ap.add_argument("--zip4_to_bg", required=True)
    ap.add_argument("--bucket", default="smart-meter-data-sb")
    ap.add_argument("--prefix", default="sharepoint-files/Zip4/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality_out", default=None)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--min_days", type=int, default=MIN_DAYS_PER_BG)
    ap.add_argument("--strict_month", action="store_true")

    args = ap.parse_args()
    z2b = pl.read_parquet(args.zip4_to_bg)

    if args.mode == "month":
        if not args.month:
            ap.error("--month required for month mode")
        build_bg_features_for_month(
            year_month=f"{args.year}{args.month:02d}",
            zip4_to_bg=z2b,
            bucket=args.bucket,
            prefix=args.prefix,
            out_features_parquet=args.out,
            out_quality_json=args.quality_out,
            max_files=args.max_files,
            min_days=args.min_days,
            strict_month=args.strict_month,
        )
    else:
        build_bg_features_for_year(
            year=args.year,
            zip4_to_bg=z2b,
            bucket=args.bucket,
            prefix=args.prefix,
            out_features_parquet=args.out,
            out_quality_json=args.quality_out,
            max_files_per_month=args.max_files,
            min_days=args.min_days,
        )
