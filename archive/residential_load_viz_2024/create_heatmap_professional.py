#!/usr/bin/env python
"""
Professional-quality load visualizations for presentation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Set Garamond font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond", "Georgia", "Times New Roman"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 18


def create_load_heatmap(usage_df: pl.DataFrame, output_path: str = "load_heatmap_pro.png"):
    """Create professional heatmap with month labels."""
    print("📊 Creating professional heatmap...")

    # Prepare data
    df = usage_df.with_columns([
        pl.col("datetime").dt.ordinal_day().alias("day_of_year"),
        pl.col("datetime").dt.hour().alias("hour"),
    ])

    # Aggregate
    heatmap_data = (
        df.group_by(["day_of_year", "hour"]).agg(pl.col("kwh").sum().alias("total_kwh")).sort(["day_of_year", "hour"])
    )

    # Pivot to matrix
    matrix = heatmap_data.pivot(index="hour", columns="day_of_year", values="total_kwh").fill_null(0)

    hour_labels = matrix.select("hour").to_series().to_list()
    day_cols = [c for c in matrix.columns if c != "hour"]
    data_matrix = matrix.select(day_cols).to_numpy()

    print(f"  Data range: {data_matrix.min():.0f} to {data_matrix.max():.0f} kWh")
    print(f"  Matrix shape: {data_matrix.shape[0]} hours × {data_matrix.shape[1]} days")

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))

    # Heatmap
    sns.heatmap(
        data_matrix,
        cmap="rocket_r",
        cbar_kws={"label": "Total kWh (8,131 homes)", "shrink": 0.8, "pad": 0.02},
        xticklabels=False,  # We'll add custom labels
        yticklabels=hour_labels,
        linewidths=0,
        ax=ax,
        robust=True,
    )

    # Add month labels on x-axis
    # 2024 is a leap year
    month_starts = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_midpoints = [(month_starts[i] + (month_starts[i + 1] if i < 11 else 367)) / 2 - 1 for i in range(12)]

    # Set month names as x-tick labels
    ax.set_xticks(month_midpoints)
    ax.set_xticklabels(month_names, fontsize=11)

    # Add subtle month divider lines
    for day in month_starts[1:]:
        ax.axvline(x=day - 1, color="white", linewidth=0.8, alpha=0.4)

    # Labels
    ax.set_xlabel("Month (2024)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Hour of Day", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        "Illinois Residential Electricity Load Patterns - 2024\n8,131 Random Households",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )

    # Invert y-axis so midnight is at top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_hourly_profile(usage_df: pl.DataFrame, output_path: str = "hourly_profile_pro.png"):
    """Create hourly profile with IQR."""
    print("\n📊 Creating professional hourly profile...")

    # Calculate stats
    hourly = (
        usage_df.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        .group_by("hour")
        .agg([
            pl.col("kwh").mean().alias("mean_kwh"),
            pl.col("kwh").quantile(0.25).alias("p25_kwh"),
            pl.col("kwh").quantile(0.75).alias("p75_kwh"),
        ])
        .sort("hour")
    )

    # Reality check output
    print("\n  📊 REALITY CHECK:")
    print(f"     Lowest avg: {hourly['mean_kwh'].min():.3f} kWh per 30-min")
    print(f"     Highest avg: {hourly['mean_kwh'].max():.3f} kWh per 30-min")
    print(f"     Daily total: ~{hourly['mean_kwh'].sum():.1f} kWh per home per day")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    hours = hourly["hour"].to_list()
    mean = hourly["mean_kwh"].to_list()
    p25 = hourly["p25_kwh"].to_list()
    p75 = hourly["p75_kwh"].to_list()

    # Interquartile range (tighter than std dev)
    ax.fill_between(hours, p25, p75, alpha=0.25, color="#d62728", label="Interquartile range (25th-75th %ile)")
    ax.plot(hours, mean, linewidth=3, color="#d62728", marker="o", markersize=6, label="Average usage", zorder=5)

    # Styling
    ax.set_xlabel("Hour of Day", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Average kWh per 30-min interval", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        "Average Hourly Electricity Usage - 2024\n8,131 Illinois Homes", fontsize=17, fontweight="bold", pad=20
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)

    # Time-of-day shading
    ax.axvspan(0, 6, alpha=0.08, color="blue", label="Overnight (12-6 AM)")
    ax.axvspan(17, 21, alpha=0.08, color="red", label="Evening peak (5-9 PM)")

    # Annotations
    peak_hour = mean.index(max(mean))
    peak_value = max(mean)
    ax.annotate(
        f"Peak: {peak_value:.3f} kWh\nat {peak_hour}:00",
        xy=(peak_hour, peak_value),
        xytext=(peak_hour - 3, peak_value + 0.03),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8, edgecolor="black", linewidth=1.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )

    min_value = min(mean)
    min_hour = mean.index(min_value)
    ax.annotate(
        f"Baseload: {min_value:.3f} kWh\nat {min_hour}:00",
        xy=(min_hour, min_value),
        xytext=(min_hour + 2, min_value + 0.05),
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8, edgecolor="black", linewidth=1.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="black", fancybox=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PROFESSIONAL RESIDENTIAL LOAD VISUALIZATION")
    print("=" * 60)

    data_file = Path("sampled_customers_2024.parquet")
    print(f"\n📂 Loading: {data_file}")
    df = pl.read_parquet(data_file)

    print(f"  Rows: {df.height:,}")
    print(f"  Customers: {df['account_identifier'].n_unique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    create_load_heatmap(df, "residential_load_heatmap_professional.png")
    create_hourly_profile(df, "residential_hourly_profile_professional.png")

    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
