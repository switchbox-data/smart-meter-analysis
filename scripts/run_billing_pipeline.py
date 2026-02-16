#!/usr/bin/env python3
"""Multi-month RTP billing pipeline orchestrator.

Chains the full billing analysis pipeline for one or more months:

1. Per month: compute_hourly_loads -> compute_household_bills
2. Concatenate monthly bills into all_months_household_bills.parquet (no aggregation)
3. (Optional) build_regression_dataset on the concatenated bills

Directory layout::

    data/bills/<run-id>/
      _tmp/
        month=YYYYMM/hourly_loads.parquet
      month=YYYYMM/household_bills.parquet
      all_months_household_bills.parquet
      regression/
        bg_month_outcomes.parquet
        bg_annual_outcomes.parquet
        bg_season_outcomes.parquet
        regression_dataset_bg.parquet
        regression_results.json
        regression_summary.txt
        regression_metadata.json
      run_manifest.json
      pipeline.log

Typical usage::

    python scripts/run_billing_pipeline.py \\
        --months 202301,202302,202303 \\
        --interval-pattern "data/processed/comed_{yyyymm}.parquet" \\
        --tariff-a data/reference/comed_flat_hourly_prices_2023.parquet \\
        --tariff-b data/reference/comed_stou_hourly_prices_2023.parquet

    python scripts/run_billing_pipeline.py \\
        --months-file months.txt \\
        --interval-pattern "data/processed/comed_{yyyymm}.parquet" \\
        --cluster-assignments-pattern "data/clustering/{yyyymm}/assignments.parquet" \\
        --tariff-a data/reference/comed_flat_hourly_prices_2023.parquet \\
        --tariff-b data/reference/comed_stou_hourly_prices_2023.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (ported from scripts/run_comed_pipeline.py)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def _get_git_sha(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        sha = (r.stdout or "").strip()
        return sha if r.returncode == 0 and sha else None
    except Exception:
        return None


def _run_subprocess(cmd: list[str], *, label: str) -> None:
    # Each pipeline step runs as a subprocess so that (a) memory is fully
    # released between steps (Polars/Arrow can be hungry), and (b) a step
    # crash produces a clean exit code without taking down the orchestrator.
    logger.info("[%s] Running: %s", label, " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"[{label}] Command failed (exit {r.returncode}): {' '.join(cmd)}")


def generate_run_id(months: list[str]) -> str:
    # Hash includes both the month list and a timestamp so that (a) re-runs
    # of the same months get distinct directories, and (b) the ID is short
    # enough for friendly directory names while still collision-resistant.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hash_input = ",".join(sorted(months)) + "|" + ts
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    return f"{ts}_{short_hash}"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_compute_hourly_loads(
    *,
    input_path: Path,
    cluster_assignments_path: Path | None,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "analysis/rtp/compute_hourly_loads.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if cluster_assignments_path is not None:
        cmd.extend(["--cluster-assignments", str(cluster_assignments_path)])
    _run_subprocess(cmd, label="hourly_loads")


def step_compute_bills(
    *,
    hourly_loads_path: Path,
    tariff_a_path: Path,
    tariff_b_path: Path,
    output_path: Path,
    capacity_rate: float,
    admin_fee: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "analysis/rtp/compute_household_bills.py",
        "--hourly-loads",
        str(hourly_loads_path),
        "--tariff-prices-a",
        str(tariff_a_path),
        "--tariff-prices-b",
        str(tariff_b_path),
        "--output",
        str(output_path),
        "--capacity-rate-dollars-per-kw-month",
        str(capacity_rate),
        "--admin-fee-dollars",
        str(admin_fee),
    ]
    _run_subprocess(cmd, label="bills")


def step_build_regression(
    *,
    bills_path: Path,
    crosswalk_path: Path,
    census_path: Path,
    output_dir: Path,
    predictors: str,
    max_crosswalk_drop_pct: float,
    min_obs_per_bg: int,
    regression_level: str = "annual",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "analysis/rtp/build_regression_dataset.py",
        "--bills",
        str(bills_path),
        "--crosswalk",
        str(crosswalk_path),
        "--census",
        str(census_path),
        "--output-dir",
        str(output_dir),
        "--predictors",
        predictors,
        "--max-crosswalk-drop-pct",
        str(max_crosswalk_drop_pct),
        "--min-obs-per-bg",
        str(min_obs_per_bg),
        "--regression-level",
        regression_level,
    ]
    _run_subprocess(cmd, label="regression")


# ---------------------------------------------------------------------------
# Concatenate monthly bills (no cross-month aggregation by account_identifier)
# ---------------------------------------------------------------------------


def build_all_months_bills(
    run_dir: Path,
    months: list[str],
    out_path: Path,
) -> int:
    """Concatenate per-month household bills with a ``month`` column.

    Household IDs do NOT persist across months, so we must NOT aggregate
    by account_identifier across months.  This function simply stacks the
    monthly bill files and tags each row with its YYYYMM month string.

    Returns:
        Number of rows in the concatenated output.
    """
    lfs: list[pl.LazyFrame] = []
    for ym in months:
        path = run_dir / f"month={ym}" / "household_bills.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Monthly bills not found: {path}")
        lf = pl.scan_parquet(path).with_columns(pl.lit(ym).alias("month"))
        lfs.append(lf)

    combined = pl.concat(lfs, how="vertical")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sink_parquet(out_path)
    n_rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    logger.info(
        "Concatenated %d monthly bill files: %d total rows -> %s",
        len(lfs),
        n_rows,
        out_path,
    )
    return n_rows


# ---------------------------------------------------------------------------
# Month resolution
# ---------------------------------------------------------------------------


def resolve_months(args: argparse.Namespace) -> list[str]:
    """Parse and validate month list from CLI args."""
    if args.months:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
    elif args.months_file:
        p = Path(args.months_file)
        if not p.exists():
            raise FileNotFoundError(f"Months file not found: {p}")
        months = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    else:
        raise ValueError("Provide either --months or --months-file.")

    # Validate YYYYMM format
    for m in months:
        if len(m) != 6 or not m.isdigit():
            raise ValueError(f"Invalid month format '{m}'; expected YYYYMM.")

    return sorted(set(months))


def resolve_path(pattern: str, yyyymm: str) -> Path:
    """Replace {yyyymm} placeholder in a path pattern."""
    return Path(pattern.replace("{yyyymm}", yyyymm))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-month RTP billing pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Month selection (mutually exclusive group)
    mg = p.add_mutually_exclusive_group(required=True)
    mg.add_argument("--months", type=str, help="Comma-separated YYYYMM list.")
    mg.add_argument("--months-file", type=str, help="Path to file with one YYYYMM per line.")

    # Input patterns
    p.add_argument(
        "--interval-pattern",
        type=str,
        required=True,
        help="Path pattern with {yyyymm} for interval parquet files.",
    )
    p.add_argument(
        "--cluster-assignments-pattern",
        type=str,
        default=None,
        help="Optional path pattern with {yyyymm} for cluster assignments.",
    )

    # Tariff files
    p.add_argument("--tariff-a", type=Path, required=True, help="Baseline tariff prices parquet.")
    p.add_argument("--tariff-b", type=Path, required=True, help="Alternative tariff prices parquet.")

    # Reference data
    p.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("data/reference/comed_bg_zip4_crosswalk.txt"),
        help="ZIP+4 -> BG crosswalk TSV.",
    )
    p.add_argument(
        "--census",
        type=Path,
        default=Path("data/reference/census_17_2023.parquet"),
        help="Census demographics parquet.",
    )

    # Billing parameters
    p.add_argument("--capacity-rate", type=float, default=0.0, help="$/kW-month for tariff B.")
    p.add_argument("--admin-fee", type=float, default=0.0, help="$/month admin fee for tariff B.")

    # Regression parameters
    p.add_argument("--predictors", type=str, default="auto", help="'auto' | 'core' | 'col1,col2,...'")
    p.add_argument("--max-crosswalk-drop-pct", type=float, default=5.0)
    p.add_argument("--min-obs-per-bg", type=int, default=3)

    # Pipeline control
    p.add_argument("--run-name", type=str, default=None, help="Override run ID.")
    p.add_argument("--output-dir", type=Path, default=Path("data/bills"), help="Base output dir.")
    p.add_argument("--skip-regression", action="store_true", help="Skip regression step.")
    p.add_argument(
        "--regression-level",
        type=str,
        choices=["annual", "bg_month"],
        default="annual",
        help="Regression granularity: 'annual' or 'bg_month' (BG x month + month FE).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ── Resolve months ───────────────────────────────────────────────────
    months = resolve_months(args)

    # ── Run ID + directories ─────────────────────────────────────────────
    run_id = args.run_name or generate_run_id(months)
    run_dir = args.output_dir / run_id
    # Hourly loads are large intermediate files only needed during the run;
    # _tmp/ signals they can be deleted after the pipeline completes.
    tmp_dir = run_dir / "_tmp"
    regression_dir = run_dir / "regression"

    run_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging BEFORE first logger.info() call
    _configure_logging(run_dir / "pipeline.log")
    logger.info("Months to process: %s", months)
    logger.info("Run ID: %s", run_id)
    logger.info("Output directory: %s", run_dir)

    # ── Validate tariff files exist ──────────────────────────────────────
    for path, label in [(args.tariff_a, "tariff-a"), (args.tariff_b, "tariff-b")]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            return 1

    # ── Per-month processing ─────────────────────────────────────────────
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": _utc_now_iso(),
        "git_sha": _get_git_sha(Path(".")),
        "months": months,
        "month_summary": {},
        "inputs": {
            "interval_pattern": args.interval_pattern,
            "cluster_assignments_pattern": args.cluster_assignments_pattern,
            "tariff_a": str(args.tariff_a),
            "tariff_b": str(args.tariff_b),
            "crosswalk": str(args.crosswalk),
            "census": str(args.census),
        },
        "parameters": {
            "capacity_rate": args.capacity_rate,
            "admin_fee": args.admin_fee,
            "predictors": args.predictors,
            "max_crosswalk_drop_pct": args.max_crosswalk_drop_pct,
            "min_obs_per_bg": args.min_obs_per_bg,
            "skip_regression": args.skip_regression,
            "regression_level": args.regression_level,
        },
        "steps_completed": [],
    }

    for ym in months:
        logger.info("════ Processing month %s ════", ym)

        # Resolve input path
        input_path = resolve_path(args.interval_pattern, ym)
        if not input_path.exists():
            logger.error("Interval data not found for %s: %s", ym, input_path)
            return 1

        # Resolve optional cluster assignments
        cluster_path = None
        if args.cluster_assignments_pattern:
            cluster_path = resolve_path(args.cluster_assignments_pattern, ym)
            if not cluster_path.exists():
                logger.warning(
                    "Cluster assignments not found for %s: %s (proceeding without).",
                    ym,
                    cluster_path,
                )
                cluster_path = None

        # Step 1: hourly loads
        loads_path = tmp_dir / f"month={ym}" / "hourly_loads.parquet"
        step_compute_hourly_loads(
            input_path=input_path,
            cluster_assignments_path=cluster_path,
            output_path=loads_path,
        )

        # Step 2: household bills
        bills_path = run_dir / f"month={ym}" / "household_bills.parquet"
        step_compute_bills(
            hourly_loads_path=loads_path,
            tariff_a_path=args.tariff_a,
            tariff_b_path=args.tariff_b,
            output_path=bills_path,
            capacity_rate=args.capacity_rate,
            admin_fee=args.admin_fee,
        )

        # Record row counts
        loads_rows = pl.scan_parquet(loads_path).select(pl.len()).collect().item()
        bills_rows = pl.scan_parquet(bills_path).select(pl.len()).collect().item()
        manifest["month_summary"][ym] = {
            "rows_hourly_loads": int(loads_rows),
            "rows_bills": int(bills_rows),
        }
        manifest["steps_completed"].append(ym)
        logger.info("Month %s complete: %d loads rows, %d bills rows.", ym, loads_rows, bills_rows)

    # ── Concatenate all months bills ────────────────────────────────────
    logger.info("════ Building all-months household bills ════")
    all_months_bills_path = run_dir / "all_months_household_bills.parquet"
    n_all = build_all_months_bills(run_dir, months, all_months_bills_path)
    manifest["all_months_bills_rows"] = n_all
    manifest["steps_completed"].append("all_months_bills")

    # ── Regression ───────────────────────────────────────────────────────
    if not args.skip_regression:
        logger.info("════ Running regression ════")
        step_build_regression(
            bills_path=all_months_bills_path,
            crosswalk_path=args.crosswalk,
            census_path=args.census,
            output_dir=regression_dir,
            predictors=args.predictors,
            max_crosswalk_drop_pct=args.max_crosswalk_drop_pct,
            min_obs_per_bg=args.min_obs_per_bg,
            regression_level=args.regression_level,
        )
        manifest["steps_completed"].append("regression")
    else:
        logger.info("Skipping regression (--skip-regression).")

    # ── Write manifest ───────────────────────────────────────────────────
    manifest["completed_utc"] = _utc_now_iso()
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)

    logger.info("Pipeline complete. Output: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
