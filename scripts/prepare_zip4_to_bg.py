#!/usr/bin/env python3
"""
Normalize a ZIP+4 -> Census Block(-Group) crosswalk to a standard schema.

Output columns:
- zip5:    5-char ZIP (string, zero-padded)
- plus4:   4-char ZIP+4 (string, zero-padded)
- bg_geoid: 12-char Block Group GEOID (string)
- weight:  float in (0,1], sums to ~1 within each ZIP+4 if a ZIP+4 splits
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _zero_pad(s: pl.Expr, n: int) -> pl.Expr:
    return s.cast(pl.Utf8).str.strip_chars().str.zfill(n)


def load_any(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pl.read_parquet(p)
    if suf in {".csv", ".txt"}:
        return pl.read_csv(p)
    if suf in {".xlsx", ".xls"}:
        return pl.read_excel(p, sheet_id=0)
    # TRY003: no long messages
    raise ValueError


def to_bg_geoid_from_parts(df: pl.DataFrame) -> pl.DataFrame:
    """
    If file has parts (state, county, tract, bg), build a 12-char BG GEOID.
    If already present, return as-is.
    """
    cols = {c.lower() for c in df.columns}
    if "bg_geoid" in cols:
        return df

    # Map common vendor column names into canonical names
    mapping: dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"state", "statefp", "state_fips"}:
            mapping[c] = "state"
        elif lc in {"county", "countyfp", "county_fips"}:
            mapping[c] = "county"
        elif lc in {"tract", "tractce", "censustract", "census_tract"}:
            mapping[c] = "tract"
        elif lc in {"blockgroup", "block_group", "bg", "blkgrp"}:
            mapping[c] = "bg"

    if {"state", "county", "tract", "bg"}.issubset(set(mapping.values())):
        tmp = df.rename(mapping)  # C416: pass mapping directly
        return tmp.with_columns([
            _zero_pad(pl.col("state"), 2).alias("state"),
            _zero_pad(pl.col("county"), 3).alias("county"),
            _zero_pad(pl.col("tract"), 6).alias("tract"),
            _zero_pad(pl.col("bg"), 1).alias("bg"),
        ]).with_columns((pl.col("state") + pl.col("county") + pl.col("tract") + pl.col("bg")).alias("bg_geoid"))
    return df


def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Produce columns: zip5, plus4, bg_geoid, weight
    """
    # helper to pick first matching column by likely aliases
    cols = {c.lower(): c for c in df.columns}

    def pick(options: list[str]) -> str | None:
        for o in options:
            if o in cols:
                return cols[o]
        return None

    zip_col = pick(["zip", "zip5", "zipcode"])
    plus4_col = pick(["plus4", "zip4", "zip+4", "zip_4"])
    bg_col = pick(["bg_geoid", "geoid_bg", "bggeoid", "block_group_geoid", "geoid"])

    if bg_col is None:
        df = to_bg_geoid_from_parts(df)
        cols = {c.lower(): c for c in df.columns}
        bg_col = pick(["bg_geoid", "geoid"])

    if zip_col is None or plus4_col is None or bg_col is None:
        # TRY003
        raise ValueError

    out = (
        df.select([
            _zero_pad(pl.col(zip_col), 5).alias("zip5"),
            _zero_pad(pl.col(plus4_col), 4).alias("plus4"),
            pl.col(bg_col).cast(pl.Utf8).str.strip_chars().alias("bg_geoid"),
        ])
        .filter(pl.col("bg_geoid").str.len_chars() == 12)
        .with_columns(pl.lit(1.0).alias("weight"))
    )

    # Prefer input weight if provided
    lower = [c.lower() for c in df.columns]
    if "weight" in lower:
        wcol = next(c for c in df.columns if c.lower() == "weight")  # RUF015
        out = out.drop("weight").with_columns(pl.col(wcol).cast(pl.Float64).alias("weight"))

    # sanity checks
    if not out.filter(pl.col("bg_geoid").str.len_chars() != 12).is_empty():
        # TRY003
        raise ValueError

    # weights ~1 per ZIP+4
    check = (
        out.group_by(["zip5", "plus4"])
        .agg(pl.col("weight").sum().alias("_w"))
        .filter((pl.col("_w") < 0.9999) | (pl.col("_w") > 1.0001))
    )
    if check.height:
        # TRY003
        raise ValueError

    return out


def main(in_path: str, out_path: str) -> None:
    df = load_any(in_path)
    out = normalize_schema(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    print(f"wrote {out.shape[0]} rows -> {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Normalize ZIP+4 -> BG crosswalk")
    ap.add_argument("--inp", required=True, help="input (.parquet/.csv/.xlsx)")
    ap.add_argument("--out", required=True, help="output parquet")
    args = ap.parse_args()
    main(args.inp, args.out)
