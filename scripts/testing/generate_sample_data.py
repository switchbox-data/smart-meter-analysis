#!/usr/bin/env python3
"""
Generate synthetic ComEd-like interval CSV files for local testing.

Output matches production structure and data types:
- Identifier columns
- 30-minute interval columns (end-aligned) HR0030 … HR2400
- HR2430 and HR2500 columns present and empty
- PLC_VALUE and NSPL_VALUE
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import polars as pl

# End-aligned 30-minute intervals: 00:30 through 24:00 inclusive
BASE_INTERVALS: List[str] = [f"INTERVAL_HR{m // 60:02d}{m % 60:02d}_ENERGY_QTY" for m in range(30, 24 * 60 + 1, 30)]

# Fall-back placeholders present in production files (kept empty)
DST_FALL_PLACEHOLDERS: List[str] = [
    "INTERVAL_HR2430_ENERGY_QTY",
    "INTERVAL_HR2500_ENERGY_QTY",
]

ALL_INTERVALS: List[str] = BASE_INTERVALS + DST_FALL_PLACEHOLDERS


@dataclass(frozen=True)
class GenerationConfig:
    num_accounts: int
    num_days: int
    start_date: date
    out_dir: Path
    seed: Optional[int]


def _iter_days(start: date, num_days: int) -> Iterable[date]:
    for i in range(num_days):
        yield start + timedelta(days=i)


def _synthetic_kwh(hour_24: int, rng: random.Random) -> float:
    baseline = 0.15
    morning = 0.25 if 6 <= hour_24 < 9 else 0.0
    evening = 0.50 if 17 <= hour_24 < 21 else 0.0
    noise = rng.uniform(-0.03, 0.03)
    return round(max(0.02, baseline + morning + evening + noise), 3)


def _generate_base_intervals(rng: random.Random) -> list[float]:
    vals: list[float] = []
    for m in range(30, 24 * 60 + 1, 30):
        vals.append(_synthetic_kwh((m // 60) % 24, rng))
    return vals


def _make_day_frame(
    service_day: date,
    accounts: list[int],
    rng: random.Random,
) -> pl.DataFrame:
    zip_choices = ("60614", "60622", "60647", "60657", "60618", "60616")
    svc_class = ("R", "C")
    svc_name = {"R": "Residential", "C": "Commercial"}

    rows = []
    for acc in accounts:
        cls = rng.choice(svc_class)

        row = {
            "ZIP_CODE": rng.choice(zip_choices),
            "DELIVERY_SERVICE_CLASS": cls,
            "DELIVERY_SERVICE_NAME": svc_name[cls],
            "ACCOUNT_IDENTIFIER": acc,
            "INTERVAL_READING_DATE": service_day.strftime("%m/%d/%Y"),
            "INTERVAL_LENGTH": 30,
        }

        intervals = _generate_base_intervals(rng)
        for col, val in zip(BASE_INTERVALS, intervals, strict=True):
            row[col] = val

        # HR2430 and HR2500 exist and are empty
        row["INTERVAL_HR2430_ENERGY_QTY"] = None
        row["INTERVAL_HR2500_ENERGY_QTY"] = None

        total = float(sum(intervals))
        row["TOTAL_REGISTERED_ENERGY"] = round(total, 3)

        # Trailing values present and empty
        row["PLC_VALUE"] = None
        row["NSPL_VALUE"] = None

        rows.append(row)

    df = pl.DataFrame(rows)

    ordered = [
        "ZIP_CODE",
        "DELIVERY_SERVICE_CLASS",
        "DELIVERY_SERVICE_NAME",
        "ACCOUNT_IDENTIFIER",
        "INTERVAL_READING_DATE",
        "INTERVAL_LENGTH",
        "TOTAL_REGISTERED_ENERGY",
        *ALL_INTERVALS,
        "PLC_VALUE",
        "NSPL_VALUE",
    ]
    for col in ordered:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    df = df.select(ordered)

    df = df.with_columns([
        pl.col("ZIP_CODE").cast(pl.Utf8),
        pl.col("DELIVERY_SERVICE_CLASS").cast(pl.Utf8),
        pl.col("DELIVERY_SERVICE_NAME").cast(pl.Utf8),
        pl.col("ACCOUNT_IDENTIFIER").cast(pl.Int64),
        pl.col("INTERVAL_READING_DATE").cast(pl.Utf8),
        pl.col("INTERVAL_LENGTH").cast(pl.Int64),
        pl.col("TOTAL_REGISTERED_ENERGY").cast(pl.Float64),
        *[pl.col(c).cast(pl.Float64) for c in ALL_INTERVALS],
        pl.col("PLC_VALUE").cast(pl.Float64),
        pl.col("NSPL_VALUE").cast(pl.Float64),
    ])
    return df


def _parse_args() -> GenerationConfig:
    ap = argparse.ArgumentParser(description="Generate synthetic ComEd-like interval CSV files.")
    ap.add_argument("--num-accounts", type=int, default=3)
    ap.add_argument("--num-days", type=int, default=5)
    ap.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD (default: today - num_days + 1)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/samples"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    today = date.today()
    if args.start_date:
        y, m, d = map(int, args.start_date.split("-"))
        start = date(y, m, d)
    else:
        start = today - timedelta(days=args.num_days - 1)

    return GenerationConfig(
        num_accounts=args.num_accounts,
        num_days=args.num_days,
        start_date=start,
        out_dir=Path(args.out_dir),
        seed=args.seed,
    )


def main() -> int:
    cfg = _parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    accounts = list(range(10_000_000, 10_000_000 + cfg.num_accounts))

    for day in _iter_days(cfg.start_date, cfg.num_days):
        df = _make_day_frame(day, accounts, rng)
        out_path = cfg.out_dir / f"comed_sample_{day.strftime('%Y%m%d')}.csv"
        df.write_csv(out_path)
        print(f"Wrote {out_path} ({df.height} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
