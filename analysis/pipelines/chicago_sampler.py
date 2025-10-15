#!/usr/bin/env python
"""
CLI wrapper for Chicago smart meter sampling.
Handles multiple ZIP codes and writes output for each.
"""

import argparse
import csv
import logging
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
import polars as pl
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import contextlib

from smart_meter_analysis.transformation import add_time_columns, transform_wide_to_long


class ZipSampler:
    """Sampler for one ZIP code."""

    def __init__(self, config: dict):
        self.bucket = config["bucket"]
        self.prefix_base = config["prefix_base"].rstrip("/")
        self.zip5 = config["zip5"]
        self.target_customers = config["target_customers"]
        self.start_yyyymm = config["start_yyyymm"]
        self.end_yyyymm = config["end_yyyymm"]
        self.output_dir = Path(config["output_dir"])
        self.max_workers = config["max_workers"]
        self.max_retries = config["max_retries"]
        self.retry_delay = config["retry_delay"]
        self.random_seed = config.get("random_seed", 42)
        self.batch_log_every = config.get("batch_log_every", 400)

        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.output_dir / "checkpoints"
        self.shards_dir = self.checkpoint / "shards"
        self.final_dir = self.output_dir / "final"
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.logger = logging.getLogger(f"sampler-{self.zip5}")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add file and console handlers
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # S3 client
        self.s3 = boto3.client(
            "s3",
            config=BotoConfig(
                retries={"max_attempts": self.max_retries, "mode": "standard"},
                max_pool_connections=max(8, self.max_workers),
            ),
        )

        # Output files
        self.final_raw = self.final_dir / "RAW.parquet"
        self.final_clipped = self.final_dir / "CLIPPED.parquet"
        self.final_clipped_cm90 = self.final_dir / "CLIPPED_CM90.parquet"

        random.seed(self.random_seed)
        self.rng = random.Random(self.random_seed)  # noqa: S311 - for data sampling, not cryptography

    def months_desc(self) -> list[str]:
        """Inclusive descending list from start → end."""
        sy, sm = int(self.start_yyyymm[:4]), int(self.start_yyyymm[4:])
        ey, em = int(self.end_yyyymm[:4]), int(self.end_yyyymm[4:])
        seq = []
        y, m = sy, sm
        while (y > ey) or (y == ey and m >= em):
            seq.append(f"{y:04d}{m:02d}")
            m -= 1
            if m == 0:
                y -= 1
                m = 12
        return seq

    @staticmethod
    def _norm(name: str) -> str:
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def detect_id_col(self, columns: list[str]) -> Optional[str]:
        candidates = {"accountidentifier", "accountid", "acctid", "acctidentifier"}
        m = {self._norm(c): c for c in columns}
        for key in candidates:
            if key in m:
                return m[key]
        if "ACCOUNT_IDENTIFIER" in columns:
            return "ACCOUNT_IDENTIFIER"
        return None

    def month_keys(self, yyyymm: str) -> list[str]:
        prefix = f"{self.prefix_base}/{yyyymm}/"
        out: list[str] = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(".csv") and f"_{self.zip5}-" in k and f"_{yyyymm}_" in k:
                    out.append(k)
        return out

    def download_tmp(self, key: str) -> Path:
        fd, path = tempfile.mkstemp(prefix="s3_", suffix=".csv")
        os.close(fd)
        p = Path(path)
        attempt = 0

        while True:
            attempt += 1
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                with open(p, "wb") as f:
                    for chunk in resp["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                msg = e.response.get("Error", {}).get("Message", str(e))
                if code == "RequestTimeTooSkewed":
                    self.logger.exception("Clock skew detected; sync system time and re-run. AWS: %s", msg)
                    raise
                if attempt >= self.max_retries:
                    self.logger.warning("get_object failed for %s: %s %s — giving up.", key, code, msg)
                    raise
                time.sleep(self.retry_delay * attempt)
            else:
                return p

    def ids_in_file(self, key: str) -> list[str]:
        """Read ONLY the ID column from one CSV."""
        tmp = self.download_tmp(key)
        try:
            with open(tmp, newline="") as f:
                header = next(csv.reader(f))
            id_col = self.detect_id_col(header)
            if not id_col:
                self.logger.warning("Skipping %s: no account id column. header=%s", key, header)
                return []
            ids = (
                pl.read_csv(tmp, columns=[id_col])
                .select(pl.col(id_col).cast(pl.Utf8).str.strip_chars().alias("account_identifier"))
                .drop_nulls()
                .unique()
                .to_series()
                .to_list()
            )
        except Exception as e:
            self.logger.warning("Skipping %s: %s", key, e)
            return []
        else:
            return ids
        finally:
            with contextlib.suppress(Exception):
                tmp.unlink(missing_ok=True)

    def sample_ids_for_month(self, yyyymm: str, k: int) -> dict[str, set[str]]:
        """Scan files in random order and collect up to k unique IDs."""
        keys = self.month_keys(yyyymm)
        self.rng.shuffle(keys)
        selected: set[str] = set()
        picked_by_key: dict[str, set[str]] = {}
        self.logger.info("%s: %d candidate CSVs in ZIP%s", yyyymm, len(keys), self.zip5)

        for i, key in enumerate(keys, 1):
            if len(selected) >= k:
                break
            cand = self.ids_in_file(key)
            if not cand:
                continue
            new = [cid for cid in cand if cid not in selected]
            if not new:
                continue
            need = k - len(selected)
            take = new if len(new) <= need else self.rng.sample(new, need)
            selected.update(take)
            picked_by_key.setdefault(key, set()).update(take)

            if i % self.batch_log_every == 0:
                self.logger.info("%s: scanned %d/%d files — have %d/%d IDs", yyyymm, i, len(keys), len(selected), k)

        self.logger.info("%s: sampled %d unique IDs (target %d)", yyyymm, len(selected), k)
        return picked_by_key

    def transform_and_filter(self, local_csv: Path, keep_ids: set[str]) -> Optional[pl.DataFrame]:
        """Read CSV → filter → transform → normalize."""
        try:
            df_wide = pl.read_csv(local_csv)
        except Exception as e:
            self.logger.warning("Failed to read %s: %s", local_csv, e)
            return None

        id_col = self.detect_id_col(df_wide.columns)
        if id_col:
            df_wide = df_wide.filter(pl.col(id_col).cast(pl.Utf8).str.strip_chars().is_in(keep_ids))
            if df_wide.height == 0:
                return None

        df_long = transform_wide_to_long(df_wide)
        df_long = add_time_columns(df_long)

        out_id = self.detect_id_col(df_long.columns) or "account_identifier"
        if out_id != "account_identifier":
            df_long = df_long.rename({out_id: "account_identifier"})

        df_long = df_long.with_columns(pl.col("account_identifier").cast(pl.Utf8).str.strip_chars()).filter(
            pl.col("account_identifier").is_in(keep_ids)
        )

        return df_long if df_long.height > 0 else None

    def run(self) -> int:
        """Main sampling pipeline."""
        self.logger.info("=" * 86)
        self.logger.info(
            f"ZIP {self.zip5} - sample up to {self.target_customers:,}/month from {self.start_yyyymm} → {self.end_yyyymm}"
        )
        self.logger.info("=" * 86)
        self.logger.info(f"S3 bucket/base: s3://{self.bucket}/{self.prefix_base}/")
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info("=" * 86)

        months = self.months_desc()
        self.logger.info("Months to process (desc): %s", ", ".join(months))

        self.final_raw.unlink(missing_ok=True)
        total_rows = 0

        for yyyymm in months:
            picked_by_key = self.sample_ids_for_month(yyyymm, self.target_customers)
            if not picked_by_key:
                self.logger.warning("%s: no IDs sampled; skipping month.", yyyymm)
                continue

            shard_dir = self.shards_dir / yyyymm
            shard_dir.mkdir(parents=True, exist_ok=True)
            wrote = 0

            for idx, (key, ids) in enumerate(picked_by_key.items(), 1):
                shard_path = shard_dir / (Path(key).name.replace(".csv", ".parquet"))
                if shard_path.exists():
                    wrote += 1
                    continue
                try:
                    tmp = self.download_tmp(key)
                    try:
                        df = self.transform_and_filter(tmp, ids)
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
                        with contextlib.suppress(Exception):
                            tmp.unlink(missing_ok=True)
                except Exception as e:
                    self.logger.warning("%s: failed %s: %s", yyyymm, key, e)

                if idx % self.batch_log_every == 0:
                    self.logger.info("%s: processed %d files; shards: %d", yyyymm, idx, wrote)

            self.logger.info("%s: wrote %d shard(s) for sampled IDs", yyyymm, wrote)

        # Final sink (RAW)
        self.logger.info("Sinking all shards to RAW Parquet...")
        pl.scan_parquet(str(self.shards_dir / "*" / "*.parquet")).sink_parquet(self.final_raw)

        # CLIP to matching calendar month
        self.logger.info("Clipping to matching calendar month...")
        lf = (
            pl.scan_parquet(self.final_raw)
            .with_columns(pl.col("date").dt.strftime("%Y%m").alias("calendar_month"))
            .filter(pl.col("calendar_month") == pl.col("sample_month"))
            .drop("calendar_month")
        )
        lf.sink_parquet(self.final_clipped)

        # CM90
        self.logger.info("Building CM90 (>=90%% OK days)...")
        lf_c = pl.scan_parquet(self.final_clipped)
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
        lf_c.join(cm_keep, on=["sample_month", "account_identifier"], how="inner").sink_parquet(self.final_clipped_cm90)

        # Summaries
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
            self.logger.info("== %s ==", tag)
            self.logger.info("\n%s", tbl)

        summary(self.final_raw, "RAW")
        summary(self.final_clipped, "CLIPPED")
        summary(self.final_clipped_cm90, "CLIPPED_CM90")

        self.logger.info("✅ Done.")
        self.logger.info("RAW: %s", self.final_raw)
        self.logger.info("CLIPPED: %s", self.final_clipped)
        self.logger.info("CLIPPED_CM90: %s", self.final_clipped_cm90)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Sample Chicago smart meter data")
    parser.add_argument("--zips", help="Comma-separated ZIP codes")
    parser.add_argument("--zips-file", help="File with ZIP codes (one per line)")
    parser.add_argument("--start", required=True, help="Start month YYYYMM")
    parser.add_argument("--end", required=True, help="End month YYYYMM")
    parser.add_argument("--target-per-zip", type=int, default=1000)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix-base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cm90", type=float, default=None, help="CM90 threshold (unused for now)")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.5)

    args = parser.parse_args()

    # Get ZIP codes
    if args.zips_file:
        with open(args.zips_file) as f:
            zips = [line.strip() for line in f if line.strip()]
    elif args.zips:
        zips = [z.strip() for z in args.zips.split(",") if z.strip()]
    else:
        print("Error: Must provide --zips or --zips-file")
        return 1

    # Process each ZIP
    for zip_code in zips:
        print(f"\n{'=' * 80}")
        print(f"Processing ZIP: {zip_code}")
        print(f"{'=' * 80}")

        config = {
            "bucket": args.bucket,
            "prefix_base": args.prefix_base,
            "zip5": zip_code,
            "target_customers": args.target_per_zip,
            "start_yyyymm": args.start,
            "end_yyyymm": args.end,
            "output_dir": f"{args.out}/zip{zip_code}",
            "max_workers": args.max_workers,
            "max_retries": args.max_retries,
            "retry_delay": args.retry_delay,
        }

        sampler = ZipSampler(config)
        try:
            result = sampler.run()
            if result != 0:
                print(f"⚠️  ZIP {zip_code} failed with code {result}")
        except Exception as e:
            print(f"❌ ZIP {zip_code} failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("✅ All ZIPs processed")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
