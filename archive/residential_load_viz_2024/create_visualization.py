#!/usr/bin/env python
"""
Professional-quality load visualizations - FIXED for June 2024-June 2025 data.
Memory-safe for Codespaces.
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


def create_monthly_heatmap(data_path: str, output_path: str = "load_heatmap_monthly.png"):
    """
    Create professional heatmap using MONTHLY aggregation (memory-safe).
    Shows hour-of-day vs month (not day-of-year).
    """
    print("📊 Creating monthly heatmap (memory-safe)...")

    # LAZY LOAD - aggregate in-database
    lf = pl.scan_parquet(data_path)

    # Get stats
    stats = lf.select([
        pl.col("account_identifier").n_unique().alias("n_customers"),
        pl.min("date").alias("min_date"),
        pl.max("date").alias("max_date"),
    ]).collect(engine="streaming")

    n_customers = stats["n_customers"][0]
    date_range = f"{stats['min_date'][0]} to {stats['max_date'][0]}"
    print(f"  Customers: {n_customers:,}")
    print(f"  Date range: {date_range}")

    # Aggregate by month and hour (streaming)
    monthly_hourly = (
        lf.group_by(["sample_month", "hour"])
        .agg(pl.col("kwh").sum().alias("total_kwh"))
        .sort(["sample_month", "hour"])
        .collect(engine="streaming")
    )

    print(f"  Aggregated to {len(monthly_hourly):,} rows")

    # Pivot to matrix
    matrix = monthly_hourly.pivot(index="hour", columns="sample_month", values="total_kwh").fill_null(0)

    hour_labels = matrix.select("hour").to_series().to_list()
    month_cols = sorted([c for c in matrix.columns if c != "hour"])
    data_matrix = matrix.select(month_cols).to_numpy()

    print(f"  Matrix: {data_matrix.shape[0]} hours × {data_matrix.shape[1]} months")
    print(f"  Range: {data_matrix.min():,.0f} to {data_matrix.max():,.0f} kWh")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Heatmap
    sns.heatmap(
        data_matrix,
        cmap="rocket_r",
        cbar_kws={"label": f"Total kWh (~{n_customers:,} homes)", "shrink": 0.8, "pad": 0.02},
        xticklabels=[f"{m[:4]}-{m[4:]}" for m in month_cols],  # Format as YYYY-MM
        yticklabels=hour_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        robust=True,
    )

    # Labels
    ax.set_xlabel("Month", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Hour of Day", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Illinois Residential Electricity Load Patterns\n{date_range} • {n_customers:,} Random Households",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )

    # Rotate x-labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Invert y-axis
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_hourly_profile(data_path: str, output_path: str = "hourly_profile_pro.png"):
    """Create hourly profile with IQR (memory-safe)."""
    print("\n📊 Creating hourly profile (memory-safe)...")

    # LAZY LOAD
    lf = pl.scan_parquet(data_path)

    # Get customer count
    n_customers = lf.select(pl.col("account_identifier").n_unique()).collect(engine="streaming")[0, 0]

    # Calculate stats (streaming aggregation)
    hourly = (
        lf.group_by("hour")
        .agg([
            pl.col("kwh").mean().alias("mean_kwh"),
            pl.col("kwh").quantile(0.25).alias("p25_kwh"),
            pl.col("kwh").quantile(0.75).alias("p75_kwh"),
        ])
        .sort("hour")
        .collect(engine="streaming")
    )

    print("\n  📊 Statistics:")
    print(f"     Customers: {n_customers:,}")
    print(f"     Lowest avg: {hourly['mean_kwh'].min():.3f} kWh per 30-min")
    print(f"     Highest avg: {hourly['mean_kwh'].max():.3f} kWh per 30-min")
    print(f"     Daily total: ~{hourly['mean_kwh'].sum():.1f} kWh per home per day")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    hours = hourly["hour"].to_list()
    mean = hourly["mean_kwh"].to_list()
    p25 = hourly["p25_kwh"].to_list()
    p75 = hourly["p75_kwh"].to_list()

    # Interquartile range
    ax.fill_between(hours, p25, p75, alpha=0.25, color="#d62728", label="Interquartile range (25th-75th %ile)")
    ax.plot(hours, mean, linewidth=3, color="#d62728", marker="o", markersize=6, label="Average usage", zorder=5)

    # Styling
    ax.set_xlabel("Hour of Day", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Average kWh per 30-min interval", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Average Hourly Electricity Usage\n{n_customers:,} Illinois Homes • June 2024 - June 2025",
        fontsize=17,
        fontweight="bold",
        pad=20,
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


def create_weekend_comparison(data_path: str, output_path: str = "weekend_comparison.png"):
    """Compare weekday vs weekend load profiles."""
    print("\n📊 Creating weekend comparison...")

    lf = pl.scan_parquet(data_path)

    # Aggregate by hour and weekend status
    comparison = (
        lf.group_by(["hour", "is_weekend"])
        .agg(pl.col("kwh").mean().alias("mean_kwh"))
        .sort(["is_weekend", "hour"])
        .collect(engine="streaming")
    )

    weekday = comparison.filter(pl.col("is_weekend") == False)
    weekend = comparison.filter(pl.col("is_weekend") == True)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        weekday["hour"], weekday["mean_kwh"], linewidth=3, color="#1f77b4", marker="o", label="Weekday", markersize=6
    )
    ax.plot(
        weekend["hour"], weekend["mean_kwh"], linewidth=3, color="#ff7f0e", marker="s", label="Weekend", markersize=6
    )

    ax.set_xlabel("Hour of Day", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average kWh per 30-min", fontsize=14, fontweight="bold")
    ax.set_title("Weekday vs Weekend Load Profiles", fontsize=17, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=13, framealpha=0.95)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PROFESSIONAL RESIDENTIAL LOAD VISUALIZATION")
    print("=" * 60)

    # YOUR ACTUAL FILE
    data_file = "analysis/zip60622_2024/final/sample_60622_202406_202506_CLIPPED.parquet"

    print(f"\n📂 Using: {data_file}")

    # Check file exists
    if not Path(data_file).exists():
        print(f"❌ File not found: {data_file}")
        print("Please update the path!")
        exit(1)

    # Create visualizations (all memory-safe)
    create_monthly_heatmap(data_file, "residential_load_heatmap_professional.png")
    create_hourly_profile(data_file, "residential_hourly_profile_professional.png")
    create_weekend_comparison(data_file, "residential_weekend_comparison.png")

    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
