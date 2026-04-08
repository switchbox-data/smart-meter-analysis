#!/usr/bin/env python3
"""Wrapper that runs compaction on all uncompacted production months.

Scans /ebs/home/griffin_switch_box/runs/ for out_*_production directories,
detects which months need compaction (have batch_* files), and runs
compact_month_output.run_compaction() on each.

Usage
-----
  # Dry-run (default): plan only, no writes
  python compact_all_months.py

  # Actually compact everything
  python compact_all_months.py --execute

  # Compact specific months only
  python compact_all_months.py --execute --months 202301,202302

Expected runtime: ~3 minutes per month (~2.5 hours for 47 months).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import compact_month_output from sibling directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "csv_to_parquet"))
from compact_month_output import CompactionConfig, run_compaction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUNS_ROOT = Path("/ebs/home/griffin_switch_box/runs")
ORCHESTRATOR_LOGS = RUNS_ROOT / "_orchestrator_logs"
TARGET_SIZE_BYTES = 1_073_741_824  # 1 GiB


# ---------------------------------------------------------------------------
# Minimal logger that writes JSON lines to stderr
# ---------------------------------------------------------------------------
class StderrLogger:
    """Minimal logger with a .log(dict) method that prints JSON to stderr."""

    def log(self, record: dict) -> None:
        print(json.dumps(record), file=sys.stderr)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_production_dirs() -> list[tuple[str, Path]]:
    """Return (YYYYMM, out_root) pairs for all out_*_production dirs, sorted."""
    results: list[tuple[str, Path]] = []
    for d in RUNS_ROOT.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("out_") or not name.endswith("_production"):
            continue
        # Extract YYYYMM: out_{YYYYMM}_production
        middle = name[len("out_") : -len("_production")]
        if len(middle) == 6 and middle.isdigit():
            results.append((middle, d))
    results.sort(key=lambda x: x[0])
    return results


def needs_compaction(year_month: str, out_root: Path) -> bool:
    """Return True if the month has batch_* files (needs compaction).

    Skip if part-* files exist AND no batch_* files exist (already compacted).
    """
    yyyy = f"{int(year_month[:4]):04d}"
    mm = f"{int(year_month[4:]):02d}"
    month_dir = out_root / yyyy / mm
    if not month_dir.exists():
        return False
    files = list(month_dir.iterdir())
    has_batch = any(f.name.startswith("batch_") for f in files)
    return has_batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compact all uncompacted production months.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Plan only, no writes (default).",
    )
    mode.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Actually perform compaction.",
    )
    parser.add_argument(
        "--months",
        type=str,
        default=None,
        help="Comma-separated list of YYYYMM months to compact (e.g. 202301,202302).",
    )
    args = parser.parse_args()

    dry_run: bool = args.dry_run
    month_filter: set[str] | None = set(args.months.split(",")) if args.months else None

    # Discover
    all_months = discover_production_dirs()
    if month_filter is not None:
        all_months = [(ym, d) for ym, d in all_months if ym in month_filter]

    # Filter to uncompacted
    to_compact = [(ym, d) for ym, d in all_months if needs_compaction(ym, d)]

    total = len(to_compact)
    mode_label = "DRY RUN" if dry_run else "EXECUTE"
    print(
        f"[compact_all_months] {mode_label} — {total} month(s) need compaction",
        file=sys.stderr,
    )

    if total == 0:
        print("[compact_all_months] Nothing to do.", file=sys.stderr)
        return

    logger = StderrLogger()
    failures: list[tuple[str, str]] = []
    overall_t0 = time.monotonic()

    for idx, (year_month, out_root) in enumerate(to_compact, start=1):
        run_id = f"manual_compact_{year_month}"
        run_dir = ORCHESTRATOR_LOGS / run_id

        print(
            f"[compact_all_months] Compacting month {idx}/{total}: {year_month} ...",
            file=sys.stderr,
        )

        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = CompactionConfig(
            year_month=year_month,
            run_id=run_id,
            out_root=out_root,
            run_dir=run_dir,
            target_size_bytes=TARGET_SIZE_BYTES,
            max_files=None,
            overwrite=False,
            dry_run=dry_run,
            no_swap=False,
        )

        t0 = time.monotonic()
        try:
            result = run_compaction(cfg, logger)
            elapsed = time.monotonic() - t0
            print(
                f"[compact_all_months]   Done {year_month} in {elapsed:.1f}s — result: {json.dumps(result)}",
                file=sys.stderr,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            failures.append((year_month, str(exc)))
            print(
                f"[compact_all_months]   ERROR {year_month} after {elapsed:.1f}s: {exc}",
                file=sys.stderr,
            )

    overall_elapsed = time.monotonic() - overall_t0

    print(
        f"\n[compact_all_months] Finished {total} month(s) in {overall_elapsed:.1f}s ({mode_label}).",
        file=sys.stderr,
    )
    if failures:
        print(
            f"[compact_all_months] FAILURES ({len(failures)}/{total}):",
            file=sys.stderr,
        )
        for ym, err in failures:
            print(f"  {ym}: {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("[compact_all_months] All months succeeded.", file=sys.stderr)


if __name__ == "__main__":
    main()
