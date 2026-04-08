#!/usr/bin/env python
"""
Task runner for smart meter analysis workflows.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def cmd_sample(args):
    """
    City-wide sampling (IDs re-randomized per month).
    Writes RAW, CLIPPED, and (optionally) CLIPPED_CM90 parquet files
    under --out/final/.
    """
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build the sampler command

    cmd = [
        sys.executable,
        str(REPO_ROOT / "analysis/pipelines/chicago_sampler.py"),
        "--start",
        args.start,
        "--end",
        args.end,
        "--target-per-zip",
        str(args.target_per_zip),
        "--bucket",
        args.bucket,
        "--prefix-base",
        args.prefix_base,
        "--out",
        str(outdir),
        "--max-workers",
        str(args.max_workers),
        "--max-retries",
        str(args.max_retries),
        "--retry-delay",
        str(args.retry_delay),
    ]

    # Only add zips or zips-file if provided
    if args.zips:
        cmd.extend(["--zips", args.zips])
    elif args.zips_file:
        cmd.extend(["--zips-file", args.zips_file])

    if args.cm90 is not None:
        cmd.extend(["--cm90", str(args.cm90)])

    # Run it
    subprocess.run(cmd, check=True)  # noqa: S603 - calling internal script with validated args
    print("\n✅ Sampling complete.")
    print(f"Look in: {outdir / 'final'}")


def build_parser():
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(description="Task runner for smart meter analysis workflows")

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Sample subcommand
    ps = subparsers.add_parser("sample", help="Run city-wide sampler and write final parquet(s)")
    ps.add_argument("--zips", help="Comma-separated ZIPs, e.g. 60622,60614")
    ps.add_argument("--zips-file", help="Text file with one ZIP per line")
    ps.add_argument("--start", required=True, help="Start YYYYMM (inclusive)")
    ps.add_argument("--end", required=True, help="End YYYYMM (inclusive)")
    ps.add_argument("--target-per-zip", type=int, default=1000)
    ps.add_argument("--bucket", required=True)
    ps.add_argument("--prefix-base", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--cm90", type=float, default=None)
    ps.add_argument("--max-workers", type=int, default=6)
    ps.add_argument("--max-retries", type=int, default=5)
    ps.add_argument("--retry-delay", type=float, default=1.5)
    ps.set_defaults(func=cmd_sample)

    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Call the appropriate command function
    args.func(args)


if __name__ == "__main__":
    main()
