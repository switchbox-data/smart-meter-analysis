from __future__ import annotations

import hashlib
from collections.abc import Iterable

import polars as pl


def _stable_hash(s: str, seed: int = 0) -> int:
    """Deterministic hash; seed enables reproducible reshuffling."""
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{seed}|{s}".encode())
    return int.from_bytes(h.digest(), "big")


def _pick_county(key: str, counties: list[str], seed: int) -> str:
    if not counties:
        # TRY003
        raise ValueError
    idx = _stable_hash(key, seed) % len(counties)
    return counties[idx]


def mock_zip4_to_bg(
    zip4_list: pl.DataFrame,
    *,
    state_fips: str = "17",
    counties: Iterable[str] | None = None,
    county_pool: Iterable[str] | None = None,
    n_counties: int | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """
    Build a MOCK crosswalk for development/testing when the real one
    is not available. Produces: zip5, plus4, bg_geoid, weight.
    """
    if counties is None:
        pool = list(county_pool or ["031", "043", "089", "097", "111", "197"])
        if n_counties is None:
            counties = pool
        else:
            if n_counties <= 0:
                raise ValueError  # TRY003
            if n_counties > len(pool):
                raise ValueError  # TRY003
            counties = pool[:n_counties]
    else:
        counties = list(counties)

    if not {"zip5", "plus4"}.issubset(set(zip4_list.columns)):
        raise ValueError  # TRY003

    z = zip4_list.select(["zip5", "plus4"]).unique(maintain_order=True)

    def to_bg(zip5: str, plus4: str) -> str:
        key = f"{zip5}-{plus4}"
        c = _pick_county(key, counties, seed)  # 3-digit county FIPS
        tract = "000100"  # fake tract
        bg = "1"  # fake bg digit
        return f"{state_fips}{c}{tract}{bg}"

    return z.with_columns(
        pl.struct(["zip5", "plus4"]).map_elements(lambda s: to_bg(s["zip5"], s["plus4"])).alias("bg_geoid")
    ).with_columns(pl.lit(1.0).alias("weight"))
