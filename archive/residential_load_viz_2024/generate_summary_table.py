#!/usr/bin/env python
"""Generate formatted summary table for paper."""

from pathlib import Path

import polars as pl

DATA_FILE = "analysis/chicago_2024_full_year/combined/chicago_2024_with_april_boost_CM90.parquet"
OUTPUT_DIR = Path("analysis/chicago_2024_full_year/final_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lf = pl.scan_parquet(DATA_FILE)

monthly = (
    lf.group_by("sample_month")
    .agg([
        pl.col("account_identifier").n_unique().alias("customers"),
        pl.col("zipcode").n_unique().alias("zip_codes"),
        pl.col("kwh").mean().alias("mean_kwh"),
        pl.col("kwh").std().alias("std_kwh"),
        pl.len().alias("observations"),
    ])
    .sort("sample_month")
    .collect(engine="streaming")
)

print("=" * 80)
print("TABLE 1: Monthly Sample Summary")
print("=" * 80)
print()
print(f"{'Month':<12} {'Customers':<12} {'ZIP Codes':<12} {'Mean kWh':<12} {'Std Dev':<12} {'Observations':<15}")
print("-" * 80)

for row in monthly.iter_rows():
    month_label = f"{row[0][:4]}-{row[0][4:]}"
    asterisk = "*" if row[0] == "202404" else " "
    print(f"{month_label:<11}{asterisk} {row[1]:<12,} {row[2]:<12} {row[3]:<12.4f} {row[4]:<12.4f} {row[5]:<15,}")

total_customers = lf.select(pl.col("account_identifier").n_unique()).collect(engine="streaming")[0, 0]
total_obs = lf.select(pl.len()).collect(engine="streaming")[0, 0]

print("-" * 80)
print(f"{'TOTAL':<12} {total_customers:<12,} {'-':<12} {'-':<12} {'-':<12} {total_obs:<15,}")
print()
print("*April 2024: Limited data availability due to utility data quality issues.")
print("  Sample size reduced to 229 customers (24% of typical) despite city-wide")
print("  sampling effort across 47 ZIP codes.")
print("=" * 80)

# Save to file
output_file = OUTPUT_DIR / "summary_table.txt"
with open(output_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("TABLE 1: Monthly Sample Summary\n")
    f.write("=" * 80 + "\n\n")
    f.write(
        f"{'Month':<12} {'Customers':<12} {'ZIP Codes':<12} {'Mean kWh':<12} {'Std Dev':<12} {'Observations':<15}\n"
    )
    f.write("-" * 80 + "\n")

    for row in monthly.iter_rows():
        month_label = f"{row[0][:4]}-{row[0][4:]}"
        asterisk = "*" if row[0] == "202404" else " "
        f.write(
            f"{month_label:<11}{asterisk} {row[1]:<12,} {row[2]:<12} {row[3]:<12.4f} {row[4]:<12.4f} {row[5]:<15,}\n"
        )

    f.write("-" * 80 + "\n")
    f.write(f"{'TOTAL':<12} {total_customers:<12,} {'-':<12} {'-':<12} {'-':<12} {total_obs:<15,}\n")
    f.write("\n*April 2024: Limited data availability due to utility data quality issues.\n")
    f.write("  Sample size reduced to 229 customers (24% of typical) despite city-wide\n")
    f.write("  sampling effort across 47 ZIP codes.\n")

print(f"\n✅ Table saved to: {output_file}")

# Also save CSV
csv_file = OUTPUT_DIR / "summary_table.csv"
monthly.write_csv(csv_file)
print(f"✅ CSV saved to: {csv_file}")
