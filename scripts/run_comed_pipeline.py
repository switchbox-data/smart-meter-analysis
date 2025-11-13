# scripts/run_comed_pipeline.py
"""
CLI wrapper to run the ComEd pipeline locally or from S3.
Adds `--day-mode` to control whether 00:00 readings belong
to the same calendar day or the previous billing day.
"""

import argparse
import logging
from pathlib import Path

from smart_meter_analysis import aws_loader, transformation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ComEd pipeline")
    parser.add_argument(
        "source",
        choices=["local", "s3"],
        help="Where to load data from (local CSVs or S3 bucket)",
    )
    parser.add_argument(
        "year_month",
        help="Year-month string, e.g. 202308",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--day-mode",
        choices=["calendar", "billing"],
        default="calendar",
        help="Date attribution mode (default: calendar)",
    )
    parser.add_argument(
        "--sort-output",
        action="store_true",
        help="Sort by datetime before writing output (slower)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_path = Path(f"data/processed/comed_{args.year_month}.parquet")

    if args.source == "s3":
        aws_loader.process_month_batch(
            year_month=args.year_month,
            output_path=output_path,
            max_files=args.max_files,
            sort_output=args.sort_output,
            day_mode=args.day_mode,
        )
    else:
        # Local path: assumes CSVs are under data/samples/

        import polars as pl

        csvs = sorted(Path("data/samples").glob(f"*{args.year_month}*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No local CSVs found for {args.year_month}")

        frames = []
        for path in csvs:
            df = pl.read_csv(path)
            df_long = transformation.transform_wide_to_long(df)
            df_time = transformation.add_time_columns(df_long, day_mode=args.day_mode)
            frames.append(df_time)

        df_out = pl.concat(frames)
        if args.sort_output:
            df_out = df_out.sort("datetime")
        df_out.write_parquet(output_path)
        logging.info(f"Wrote local processed file to {output_path}")


if __name__ == "__main__":
    main()
