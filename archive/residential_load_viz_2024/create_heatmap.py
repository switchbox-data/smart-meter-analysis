#!/usr/bin/env python
"""
Create load heatmap: Hour × Day of Year
Aggregated across all customers (200 residential accounts).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def create_load_heatmap(usage_df: pl.DataFrame, output_path: str = "load_heatmap.png"):
    """Create beautiful seaborn heatmap of usage patterns."""
    print("📊 Creating heatmap...")

    # Add day_of_year and hour
    df = usage_df.with_columns([
        pl.col("datetime").dt.ordinal_day().alias("day_of_year"),
        pl.col("datetime").dt.hour().alias("hour"),
    ])

    # Aggregate: total kWh for each (day_of_year, hour) combination
    heatmap_data = (
        df.group_by(["day_of_year", "hour"]).agg(pl.col("kwh").sum().alias("total_kwh")).sort(["day_of_year", "hour"])
    )

    print(f"  Data points: {heatmap_data.height:,} hour×day combinations")

    # Pivot to matrix: rows=hours (0-23), cols=days (1-366)
    matrix = heatmap_data.pivot(index="hour", columns="day_of_year", values="total_kwh").fill_null(0)

    # Convert to numpy
    hour_labels = matrix.select("hour").to_series().to_list()
    day_cols = [c for c in matrix.columns if c != "hour"]
    data_matrix = matrix.select(day_cols).to_numpy()

    print(f"  Matrix: {data_matrix.shape[0]} hours × {data_matrix.shape[1]} days")

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 10))

    # Seaborn heatmap with YlOrRd colormap
    sns.heatmap(
        data_matrix,
        cmap="YlOrRd",  # Yellow-Orange-Red
        cbar_kws={"label": "Total kWh (200 homes)", "shrink": 0.8},
        xticklabels=30,  # Show every 30th day
        yticklabels=hour_labels,
        linewidths=0,
        ax=ax,
        robust=True,  # Use robust quantiles for color limits
    )

    # Labels and title
    ax.set_xlabel("Day of Year 2024", fontsize=16, fontweight="bold")
    ax.set_ylabel("Hour of Day", fontsize=16, fontweight="bold")
    ax.set_title(
        "Illinois Residential Electricity Load Patterns - 2024\n200 Random Households",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Invert y-axis so midnight is at top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_hourly_profile(usage_df: pl.DataFrame, output_path: str = "hourly_profile.png"):
    """Create average hourly load profile."""
    print("\n📊 Creating hourly profile...")

    # Average by hour across all days
    hourly = (
        usage_df.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        .group_by("hour")
        .agg([pl.col("kwh").mean().alias("avg_kwh"), pl.col("kwh").std().alias("std_kwh")])
        .sort("hour")
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = hourly["hour"].to_list()
    avg = hourly["avg_kwh"].to_list()
    std = hourly["std_kwh"].to_list()

    ax.plot(hours, avg, linewidth=3, color="#d62728", marker="o")
    ax.fill_between(
        hours, [a - s for a, s in zip(avg, std)], [a + s for a, s in zip(avg, std)], alpha=0.3, color="#d62728"
    )

    ax.set_xlabel("Hour of Day", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average kWh", fontsize=14, fontweight="bold")
    ax.set_title("Average Hourly Electricity Usage - 2024\n200 Illinois Homes", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("RESIDENTIAL LOAD VISUALIZATION")
    print("=" * 60)

    # Load data
    data_file = Path("sampled_customers_2024.parquet")
    print(f"\n📂 Loading: {data_file}")
    df = pl.read_parquet(data_file)

    print(f"  Rows: {df.height:,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ZIP+4 codes: {df['zip_code'].n_unique()}")

    # Create visualizations
    create_load_heatmap(df, "residential_load_heatmap_2024.png")
    create_hourly_profile(df, "residential_hourly_profile_2024.png")

    print("\n" + "=" * 60)
    print("✅ VISUALIZATIONS COMPLETE!")
    print("=" * 60)
