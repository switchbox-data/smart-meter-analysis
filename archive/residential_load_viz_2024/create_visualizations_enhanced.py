#!/usr/bin/env python
"""
Professional-quality load visualizations with summary statistics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Professional typography
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond", "Georgia", "Times New Roman"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def generate_summary_statistics(data_path: str):
    """Generate comprehensive summary statistics table."""
    print("\n📊 Generating Summary Statistics...")

    lf = pl.scan_parquet(data_path)

    # Overall statistics
    overall = lf.select([
        pl.col("account_identifier").n_unique().alias("unique_customers"),
        pl.len().alias("total_observations"),
        pl.min("date").alias("start_date"),
        pl.max("date").alias("end_date"),
        pl.col("kwh").mean().alias("mean_kwh"),
        pl.col("kwh").median().alias("median_kwh"),
        pl.col("kwh").std().alias("std_kwh"),
        pl.col("kwh").min().alias("min_kwh"),
        pl.col("kwh").max().alias("max_kwh"),
        pl.col("kwh").quantile(0.25).alias("q25_kwh"),
        pl.col("kwh").quantile(0.75).alias("q75_kwh"),
    ]).collect(engine="streaming")

    # Monthly breakdown
    monthly = (
        lf.group_by("sample_month")
        .agg([
            pl.col("account_identifier").n_unique().alias("customers"),
            pl.len().alias("observations"),
            pl.col("kwh").mean().alias("mean_kwh"),
            pl.col("kwh").sum().alias("total_kwh"),
        ])
        .sort("sample_month")
        .collect(engine="streaming")
    )

    # Weekday vs Weekend
    weekend_stats = (
        lf.group_by("is_weekend")
        .agg([
            pl.col("kwh").mean().alias("mean_kwh"),
            pl.col("kwh").median().alias("median_kwh"),
            pl.col("kwh").std().alias("std_kwh"),
        ])
        .collect(engine="streaming")
    )

    # Hourly statistics
    hourly_stats = (
        lf.group_by("hour")
        .agg([
            pl.col("kwh").mean().alias("mean_kwh"),
            pl.col("kwh").std().alias("std_kwh"),
        ])
        .sort("hour")
        .collect(engine="streaming")
    )

    # Print comprehensive report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("=" * 80)

    print("\n📋 OVERALL STATISTICS")
    print("-" * 80)
    print(f"  Total Unique Customers:        {overall['unique_customers'][0]:>12,}")
    print(f"  Total Observations:            {overall['total_observations'][0]:>12,}")
    print(f"  Date Range:                    {overall['start_date'][0]} to {overall['end_date'][0]}")
    print(f"  Average kWh per 30-min:        {overall['mean_kwh'][0]:>12.4f}")
    print(f"  Median kWh per 30-min:         {overall['median_kwh'][0]:>12.4f}")
    print(f"  Std Dev kWh:                   {overall['std_kwh'][0]:>12.4f}")
    print(f"  Min kWh:                       {overall['min_kwh'][0]:>12.4f}")
    print(f"  Max kWh:                       {overall['max_kwh'][0]:>12.4f}")
    print(f"  25th Percentile:               {overall['q25_kwh'][0]:>12.4f}")
    print(f"  75th Percentile:               {overall['q75_kwh'][0]:>12.4f}")

    # Estimated daily/annual usage
    daily_kwh = overall["mean_kwh"][0] * 48  # 48 half-hour intervals per day
    annual_kwh = daily_kwh * 365
    print(f"\n  Est. Daily Usage per Home:     {daily_kwh:>12.1f} kWh/day")
    print(f"  Est. Annual Usage per Home:    {annual_kwh:>12,.0f} kWh/year")

    print("\n📅 MONTHLY BREAKDOWN")
    print("-" * 80)
    print(f"{'Month':<12} {'Customers':>10} {'Observations':>15} {'Avg kWh':>12} {'Total kWh':>15}")
    print("-" * 80)
    for row in monthly.iter_rows():
        month_label = f"{row[0][:4]}-{row[0][4:]}"
        print(f"{month_label:<12} {row[1]:>10,} {row[2]:>15,} {row[3]:>12.4f} {row[4]:>15,.0f}")

    print("\n📊 WEEKDAY vs WEEKEND COMPARISON")
    print("-" * 80)
    print(f"{'Period':<15} {'Mean kWh':>12} {'Median kWh':>12} {'Std Dev':>12}")
    print("-" * 80)
    for row in weekend_stats.iter_rows():
        period = "Weekend" if row[0] else "Weekday"
        print(f"{period:<15} {row[1]:>12.4f} {row[2]:>12.4f} {row[3]:>12.4f}")

    print("\n⏰ HOURLY STATISTICS (Peak Hours)")
    print("-" * 80)
    # Show top 5 and bottom 5 hours
    top5 = hourly_stats.sort("mean_kwh", descending=True).head(5)
    bottom5 = hourly_stats.sort("mean_kwh").head(5)

    print("  Highest Usage Hours:")
    for row in top5.iter_rows():
        print(f"    Hour {row[0]:2d}:00  -  {row[1]:.4f} kWh (±{row[2]:.4f})")

    print("\n  Lowest Usage Hours:")
    for row in bottom5.iter_rows():
        print(f"    Hour {row[0]:2d}:00  -  {row[1]:.4f} kWh (±{row[2]:.4f})")

    print("\n" + "=" * 80)

    # Save to file
    with open("summary_statistics.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Customers: {overall['unique_customers'][0]:,}\n")
        f.write(f"Date Range: {overall['start_date'][0]} to {overall['end_date'][0]}\n")
        f.write(f"Mean kWh per 30-min: {overall['mean_kwh'][0]:.4f}\n")
        f.write(f"Estimated Daily Usage: {daily_kwh:.1f} kWh/day\n")
        f.write(f"Estimated Annual Usage: {annual_kwh:,.0f} kWh/year\n")
        f.write("\nMonthly Breakdown:\n")
        f.write(monthly.write_csv())

    print("\n✅ Summary saved to: summary_statistics.txt")

    return overall, monthly, weekend_stats, hourly_stats


def create_monthly_heatmap(data_path: str, output_path: str = "load_heatmap_seaborn.png"):
    """Create professional seaborn heatmap."""
    print("\n📊 Creating professional seaborn heatmap...")

    lf = pl.scan_parquet(data_path)

    stats = lf.select([
        pl.col("account_identifier").n_unique().alias("n_customers"),
        pl.min("date").alias("min_date"),
        pl.max("date").alias("max_date"),
    ]).collect(engine="streaming")

    n_customers = stats["n_customers"][0]
    date_range = f"{stats['min_date'][0]} to {stats['max_date'][0]}"

    monthly_hourly = (
        lf.group_by(["sample_month", "hour"])
        .agg(pl.col("kwh").sum().alias("total_kwh"))
        .sort(["sample_month", "hour"])
        .collect(engine="streaming")
    )

    matrix = monthly_hourly.pivot(index="hour", columns="sample_month", values="total_kwh").fill_null(0)

    hour_labels = matrix.select("hour").to_series().to_list()
    month_cols = sorted([c for c in matrix.columns if c != "hour"])
    data_matrix = matrix.select(month_cols).to_numpy()

    # Create figure with seaborn
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.heatmap(
        data_matrix,
        cmap="RdYlBu_r",  # Professional diverging colormap
        cbar_kws={
            "label": f"Total Energy Consumption (kWh)\n~{n_customers:,} Households",
            "shrink": 0.75,
            "pad": 0.02,
            "aspect": 30,
        },
        xticklabels=[f"{m[:4]}-{m[4:]}" for m in month_cols],
        yticklabels=hour_labels,
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        robust=True,
        square=False,
        annot=False,
    )

    # Professional styling
    ax.set_xlabel("Month", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        f"Residential Electricity Load Patterns: Temporal Heat Map\n"
        f"Chicago ZIP 60622 • {date_range} • {n_customers:,} Random Households",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    # Invert y-axis
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_hourly_profile_seaborn(data_path: str, output_path: str = "hourly_profile_seaborn.png"):
    """Create professional seaborn hourly profile."""
    print("\n📊 Creating professional seaborn hourly profile...")

    lf = pl.scan_parquet(data_path)
    n_customers = lf.select(pl.col("account_identifier").n_unique()).collect(engine="streaming")[0, 0]

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

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    hours = hourly["hour"].to_list()
    mean = hourly["mean_kwh"].to_list()
    p25 = hourly["p25_kwh"].to_list()
    p75 = hourly["p75_kwh"].to_list()

    # Use seaborn lineplot style
    ax.fill_between(hours, p25, p75, alpha=0.3, color="#e74c3c", label="Interquartile Range (25th-75th percentile)")

    sns.lineplot(x=hours, y=mean, linewidth=3.5, color="#c0392b", marker="o", markersize=8, ax=ax, label="Mean Usage")

    # Styling
    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Energy Consumption (kWh per 30-min interval)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        f"Average Hourly Electricity Usage Profile\n{n_customers:,} Illinois Households • June 2024 - June 2025",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)

    # Time-of-day shading
    ax.axvspan(0, 6, alpha=0.1, color="#3498db", label="Overnight (12-6 AM)", zorder=0)
    ax.axvspan(17, 21, alpha=0.1, color="#e74c3c", label="Evening Peak (5-9 PM)", zorder=0)

    # Annotations with improved styling
    peak_hour = mean.index(max(mean))
    peak_value = max(mean)
    ax.annotate(
        f"Peak Demand\n{peak_value:.3f} kWh\nat {peak_hour}:00",
        xy=(peak_hour, peak_value),
        xytext=(peak_hour - 4, peak_value + 0.04),
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f39c12", alpha=0.9, edgecolor="#000", linewidth=2),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#000", connectionstyle="arc3,rad=0.3"),
    )

    min_value = min(mean)
    min_hour = mean.index(min_value)
    ax.annotate(
        f"Baseload\n{min_value:.3f} kWh\nat {min_hour}:00",
        xy=(min_hour, min_value),
        xytext=(min_hour + 3, min_value + 0.06),
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#3498db", alpha=0.9, edgecolor="#000", linewidth=2),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#000", connectionstyle="arc3,rad=-0.3"),
    )

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#000", fancybox=True, shadow=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_weekend_comparison_seaborn(data_path: str, output_path: str = "weekend_comparison_seaborn.png"):
    """Compare weekday vs weekend with peak/baseload annotations."""
    print("\n📊 Creating professional weekend comparison...")

    lf = pl.scan_parquet(data_path)

    comparison = (
        lf.group_by(["hour", "is_weekend"])
        .agg(pl.col("kwh").mean().alias("mean_kwh"))
        .sort(["is_weekend", "hour"])
        .collect(engine="streaming")
    )

    weekday = comparison.filter(pl.col("is_weekend") == False)
    weekend = comparison.filter(pl.col("is_weekend") == True)

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    weekday_hours = weekday["hour"].to_list()
    weekday_kwh = weekday["mean_kwh"].to_list()
    weekend_hours = weekend["hour"].to_list()
    weekend_kwh = weekend["mean_kwh"].to_list()

    # Seaborn style lines
    sns.lineplot(
        x=weekday_hours, y=weekday_kwh, linewidth=3.5, color="#2980b9", marker="o", markersize=8, ax=ax, label="Weekday"
    )
    sns.lineplot(
        x=weekend_hours, y=weekend_kwh, linewidth=3.5, color="#e67e22", marker="s", markersize=8, ax=ax, label="Weekend"
    )

    # Styling
    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Average Energy Consumption (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        "Weekday vs Weekend Load Profiles\nTemporal Consumption Patterns", fontsize=18, fontweight="bold", pad=25
    )
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)

    # Peak annotations for BOTH weekday and weekend
    # Weekday peak
    weekday_peak_hour = weekday_kwh.index(max(weekday_kwh))
    weekday_peak = max(weekday_kwh)
    ax.annotate(
        f"Weekday Peak\n{weekday_peak:.3f} kWh\nat {weekday_peak_hour}:00",
        xy=(weekday_peak_hour, weekday_peak),
        xytext=(weekday_peak_hour - 3, weekday_peak + 0.04),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#3498db", alpha=0.9, edgecolor="#000", linewidth=2),
        arrowprops=dict(arrowstyle="->", lw=2, color="#000"),
    )

    # Weekend peak
    weekend_peak_hour = weekend_kwh.index(max(weekend_kwh))
    weekend_peak = max(weekend_kwh)
    ax.annotate(
        f"Weekend Peak\n{weekend_peak:.3f} kWh\nat {weekend_peak_hour}:00",
        xy=(weekend_peak_hour, weekend_peak),
        xytext=(weekend_peak_hour + 2, weekend_peak + 0.04),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#e67e22", alpha=0.9, edgecolor="#000", linewidth=2),
        arrowprops=dict(arrowstyle="->", lw=2, color="#000"),
    )

    # Baseload annotations
    # Weekday baseload
    weekday_min = min(weekday_kwh)
    weekday_min_hour = weekday_kwh.index(weekday_min)
    ax.annotate(
        f"Weekday Baseload\n{weekday_min:.3f} kWh",
        xy=(weekday_min_hour, weekday_min),
        xytext=(weekday_min_hour + 2, weekday_min + 0.05),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#bdc3c7", alpha=0.9, edgecolor="#000", linewidth=1.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#000"),
    )

    # Weekend baseload
    weekend_min = min(weekend_kwh)
    weekend_min_hour = weekend_kwh.index(weekend_min)
    ax.annotate(
        f"Weekend Baseload\n{weekend_min:.3f} kWh",
        xy=(weekend_min_hour, weekend_min),
        xytext=(weekend_min_hour - 4, weekend_min + 0.05),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.9, edgecolor="#000", linewidth=1.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#000"),
    )

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#000", fancybox=True, shadow=True, fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("PROFESSIONAL RESIDENTIAL LOAD ANALYSIS & VISUALIZATION")
    print("=" * 80)

    data_file = "../zip60622_2024/final/sample_60622_202406_202506_CLIPPED.parquet"

    if not Path(data_file).exists():
        print(f"❌ File not found: {data_file}")
        exit(1)

    # 1. Generate comprehensive summary statistics
    generate_summary_statistics(data_file)

    # 2. Create professional visualizations
    create_monthly_heatmap(data_file, "residential_load_heatmap_professional.png")
    create_hourly_profile_seaborn(data_file, "residential_hourly_profile_professional.png")
    create_weekend_comparison_seaborn(data_file, "residential_weekend_comparison_professional.png")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 summary_statistics.txt")
    print("  📈 residential_load_heatmap_professional.png")
    print("  📈 residential_hourly_profile_professional.png")
    print("  📈 residential_weekend_comparison_professional.png")
    print("=" * 80)
