#!/usr/bin/env python
"""
Sample up to 1,000 customers in ZIP (default 60622) for each month from START_YYYYMM
down to END_YYYYMM (descending), combine to one Parquet, then:

1) CLIP: keep only rows whose calendar month matches sample_month (safety net for uploads that
   include the next month’s first day).
2) CM90 (optional): keep customer-months with >=90% "OK days" (rows/day in {46,48,50}).

Env (with defaults):
  BUCKET=smart-meter-data-sb
  PREFIX_BASE=sharepoint-files/Zip4
  ZIP5=60622
  TARGET_CUSTOMERS=1000
  START_YYYYMM=202509
  END_YYYYMM=202409
  OUTPUT_DIR=analysis/sample_zip60622_recent
  RANDOM_SEED=42
  MAX_RETRIES=5
  RETRY_DELAY=1.5
  MAX_WORKERS=6
"""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import boto3
import polars as pl
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

# ---------------- Config ----------------
BUCKET = os.getenv("BUCKET", "smart-meter-data-sb")
PREFIX_BASE = os.getenv("PREFIX_BASE", "sharepoint-files/Zip4")
ZIP5 = os.getenv("ZIP5", "60622")
TARGET_CUSTOMERS = int(os.getenv("TARGET_CUSTOMERS", "1000"))

START_YYYYMM = os.getenv("START_YYYYMM", "202509")
END_YYYYMM = os.getenv("END_YYYYMM", "202409")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", f"analysis/sample_zip{ZIP5}_recent"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUTPUT_DIR / "checkpoints"
SHARDS_DIR = CHECKPOINT / "shards"
FINAL_DIR = OUTPUT_DIR / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))
BATCH_LOG_EVERY = int(os.getenv("BATCH_LOG_EVERY", "400"))

random.seed(RANDOM_SEED)

# output file names
RANGE_TAG = f"{END_YYYYMM}_{START_YYYYMM}"
FINAL_RAW = FINAL_DIR / f"sample_{ZIP5}_{RANGE_TAG}.parquet"
FINAL_CLIPPED = FINAL_DIR / f"sample_{ZIP5}_{RANGE_TAG}_CLIPPED.parquet"
FINAL_CLIPPED_CM90 = FINAL_DIR / f"sample_{ZIP5}_{RANGE_TAG}_CLIPPED_CM90.parquet"

# -------------- Logging ---------------
LOG_FILE = OUTPUT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("zip-monthly-sampler")

logger.info("=" * 86)
logger.info(f"ZIP {ZIP5} – sample up to {TARGET_CUSTOMERS:,}/month from {START_YYYYMM} → {END_YYYYMM} (descending)")
logger.info("=" * 86)
logger.info(f"S3 bucket/base:   s3://{BUCKET}/{PREFIX_BASE}/")
logger.info(f"Output dir:       {OUTPUT_DIR}")
logger.info(f"Final RAW:        {FINAL_RAW}")
logger.info(f"Final CLIPPED:    {FINAL_CLIPPED}")
logger.info(f"Final CM90:       {FINAL_CLIPPED_CM90}")
logger.info("=" * 86)

# -------------- Project transforms --------------
# Make repo root importable if this script lives in analysis/...
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smart_meter_analysis.transformation import add_time_columns, transform_wide_to_long  # noqa: E402

# -------------- S3 client --------------
s3 = boto3.client(
    "s3",
    config=BotoConfig(
        retries={"max_attempts": MAX_RETRIES, "mode": "standard"},
        max_pool_connections=max(8, MAX_WORKERS),
    ),
)


# -------------- Helpers --------------
def months_desc(start_yyyymm: str, end_yyyymm: str) -> List[str]:
    """Inclusive descending list from start → end (e.g., 202509..202409)."""
    sy, sm = int(start_yyyymm[:4]), int(start_yyyymm[4:])
    ey, em = int(end_yyyymm[:4]), int(end_yyyymm[4:])
    seq = []
    y, m = sy, sm
    while (y > ey) or (y == ey and m >= em):
        seq.append(f"{y:04d}{m:02d}")
        # step back one month
        m -= 1
        if m == 0:
            y -= 1
            m = 12
    return seq


def _norm(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def detect_id_col(columns: List[str]) -> Optional[str]:
    candidates = {"accountidentifier", "accountid", "acctid", "acctidentifier"}
    m = {_norm(c): c for c in columns}
    for key in candidates:
        if key in m:
            return m[key]
    if "ACCOUNT_IDENTIFIER" in columns:
        return "ACCOUNT_IDENTIFIER"
    return None


def month_keys(yyyymm: str, zip5: str) -> List[str]:
    prefix = f"{PREFIX_BASE}/{yyyymm}/"
    out: List[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            # FIXED: Validate filename month matches folder month
            if k.endswith(".csv") and f"_{zip5}-" in k and f"_{yyyymm}_" in k:
                out.append(k)
    return out


def download_tmp(key: str) -> Path:
    fd, path = tempfile.mkstemp(prefix="s3_", suffix=".csv")
    os.close(fd)
    p = Path(path)
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            with open(p, "wb") as f:
                for chunk in resp["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return p
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            msg = e.response.get("Error", {}).get("Message", str(e))
            if code == "RequestTimeTooSkewed":
                logger.error("Clock skew detected; sync system time and re-run. AWS: %s", msg)
                raise
            if attempt >= MAX_RETRIES:
                logger.warning("get_object failed for %s: %s %s — giving up.", key, code, msg)
                raise
            time.sleep(RETRY_DELAY * attempt)


def ids_in_file(key: str) -> List[str]:
    """Read ONLY the ID column from one CSV."""
    tmp = download_tmp(key)
    try:
        with open(tmp, newline="") as f:
            header = next(csv.reader(f))
        id_col = detect_id_col(header)
        if not id_col:
            logger.warning("Skipping %s: no account id column. header=%s", key, header)
            return []
        ids = (
            pl.read_csv(tmp, columns=[id_col])
            .select(pl.col(id_col).cast(pl.Utf8).str.strip_chars().alias("account_identifier"))
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        return ids
    except Exception as e:
        logger.warning("Skipping %s: %s", key, e)
        return []
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def sample_ids_for_month(yyyymm: str, zip5: str, k: int, rng: random.Random) -> Dict[str, Set[str]]:
    """Scan files in random order and collect up to k unique IDs. Returns {key -> set(ids)}."""
    keys = month_keys(yyyymm, zip5)
    rng.shuffle(keys)
    selected: Set[str] = set()
    picked_by_key: Dict[str, Set[str]] = {}
    logger.info("%s: %d candidate CSVs in ZIP%s", yyyymm, len(keys), zip5)

    for i, key in enumerate(keys, 1):
        if len(selected) >= k:
            break
        cand = ids_in_file(key)
        if not cand:
            continue
        new = [cid for cid in cand if cid not in selected]
        if not new:
            continue
        need = k - len(selected)
        take = new if len(new) <= need else rng.sample(new, need)
        selected.update(take)
        picked_by_key.setdefault(key, set()).update(take)

        if i % BATCH_LOG_EVERY == 0:
            logger.info("%s: scanned %d/%d files — have %d/%d IDs", yyyymm, i, len(keys), len(selected), k)

    logger.info("%s: sampled %d unique IDs (target %d)", yyyymm, len(selected), k)
    return picked_by_key


def transform_and_filter(local_csv: Path, keep_ids: Set[str]) -> Optional[pl.DataFrame]:
    """Read CSV → early ID filter (wide) → project transforms → normalize ID → final filter."""
    try:
        df_wide = pl.read_csv(local_csv)
    except Exception as e:
        logger.warning("Failed to read %s: %s", local_csv, e)
        return None

    id_col = detect_id_col(df_wide.columns)
    if id_col:
        df_wide = df_wide.filter(pl.col(id_col).cast(pl.Utf8).str.strip_chars().is_in(keep_ids))
        if df_wide.height == 0:
            return None

    df_long = transform_wide_to_long(df_wide)
    df_long = add_time_columns(df_long)

    out_id = detect_id_col(df_long.columns) or "account_identifier"
    if out_id != "account_identifier":
        df_long = df_long.rename({out_id: "account_identifier"})

    df_long = df_long.with_columns(pl.col("account_identifier").cast(pl.Utf8).str.strip_chars()).filter(
        pl.col("account_identifier").is_in(keep_ids)
    )
    return df_long if df_long.height > 0 else None


# -------------- Pipeline --------------
def run() -> int:
    rng = random.Random(RANDOM_SEED)
    months = months_desc(START_YYYYMM, END_YYYYMM)
    logger.info("Months to process (desc): %s", ", ".join(months))

    FINAL_RAW.unlink(missing_ok=True)
    total_rows = 0

    for yyyymm in months:
        picked_by_key = sample_ids_for_month(yyyymm, ZIP5, TARGET_CUSTOMERS, rng)
        if not picked_by_key:
            logger.warning("%s: no IDs sampled; skipping month.", yyyymm)
            continue

        shard_dir = SHARDS_DIR / yyyymm
        shard_dir.mkdir(parents=True, exist_ok=True)
        wrote = 0

        for idx, (key, ids) in enumerate(picked_by_key.items(), 1):
            shard_path = shard_dir / (Path(key).name.replace(".csv", ".parquet"))
            if shard_path.exists():
                wrote += 1
                continue
            try:
                tmp = download_tmp(key)
                try:
                    df = transform_and_filter(tmp, ids)
                    if df is None:
                        continue
                    df = df.with_columns(
                        pl.lit(yyyymm).alias("sample_month"),
                        pl.lit(key).alias("source_key"),
                    )
                    df.write_parquet(shard_path)
                    total_rows += df.height
                    wrote += 1
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("%s: failed %s: %s", yyyymm, key, e)

            if idx % BATCH_LOG_EVERY == 0:
                logger.info("%s: processed %d files; shards: %d", yyyymm, idx, wrote)

        logger.info("%s: wrote %d shard(s) for sampled IDs", yyyymm, wrote)

    # Final sink (RAW)
    logger.info("Sinking all shards to RAW Parquet...")
    pl.scan_parquet(str(SHARDS_DIR / "*" / "*.parquet")).sink_parquet(FINAL_RAW)

    # CLIP to matching calendar month
    logger.info("Clipping to matching calendar month per sample_month...")
    lf = (
        pl.scan_parquet(FINAL_RAW)
        .with_columns(pl.col("date").dt.strftime("%Y%m").alias("calendar_month"))
        .filter(pl.col("calendar_month") == pl.col("sample_month"))
        .drop("calendar_month")
    )
    lf.sink_parquet(FINAL_CLIPPED)

    # CM90 (>=90% OK days, OK = 46/48/50 intervals)
    logger.info("Building CM90 (>=90%% OK days; OK rows/day in {46,48,50})...")
    lf_c = pl.scan_parquet(FINAL_CLIPPED)
    daily = lf_c.group_by(["sample_month", "account_identifier", "date"]).agg(pl.len().alias("rows"))
    ok_daily = (
        daily.filter(pl.col("rows").is_in([46, 48, 50]))
        .group_by(["sample_month", "account_identifier"])
        .agg(pl.len().alias("ok_days"))
    )
    days_in_month = (
        lf_c.select(["sample_month", "date"]).unique().group_by("sample_month").agg(pl.len().alias("days_in_month"))
    )
    cm_keep = (
        ok_daily.join(days_in_month, on="sample_month", how="left")
        .with_columns((pl.col("ok_days") / pl.col("days_in_month")).alias("pct_ok"))
        .filter(pl.col("pct_ok") >= 0.90)
        .select(["sample_month", "account_identifier"])
    )
    lf_c.join(cm_keep, on=["sample_month", "account_identifier"], how="inner").sink_parquet(FINAL_CLIPPED_CM90)

    # Quick summaries
    def summary(path: Path, tag: str):
        tbl = (
            pl.scan_parquet(path)
            .group_by("sample_month")
            .agg([
                pl.col("account_identifier").n_unique().alias("unique_ids"),
                pl.len().alias("rows"),
                pl.min("date").alias("min_date"),
                pl.max("date").alias("max_date"),
            ])
            .collect()
            .sort("sample_month")
        )
        logger.info("== %s ==", tag)
        logger.info("\n%s", tbl)

    summary(FINAL_RAW, "RAW")
    summary(FINAL_CLIPPED, "CLIPPED")
    summary(FINAL_CLIPPED_CM90, "CLIPPED_CM90")

    logger.info("✅ Done.")
    logger.info("RAW:        %s", FINAL_RAW)
    logger.info("CLIPPED:    %s", FINAL_CLIPPED)
    logger.info("CLIPPED_CM90:%s", FINAL_CLIPPED_CM90)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Exiting.")
        raise
