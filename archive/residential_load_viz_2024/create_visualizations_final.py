#!/usr/bin/env python
"""
Final professional visualizations with all corrections.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def create_monthly_heatmap(data_path: str, output_path: str = "heatmap_final.png"):
    """Create professional seaborn heatmap."""
    print("\n📊 Creating heatmap...")

    lf = pl.scan_parquet(data_path)

    # Filter to Aug 2024 - Aug 2025 only
    desired_months = [f"2024{m:02d}" for m in range(8, 13)] + [f"2025{m:02d}" for m in range(1, 9)]
    lf = lf.filter(pl.col("sample_month").is_in(desired_months))

    stats = lf.select([
        pl.col("account_identifier").n_unique().alias("n_customers"),
        pl.min("date").alias("min_date"),
        pl.max("date").alias("max_date"),
    ]).collect(engine="streaming")

    n_customers = stats["n_customers"][0]
    date_range = f"{stats['min_date'][0]} to {stats['max_date'][0]}"

    print(f"  Unique households: {n_customers:,}")

    monthly_hourly = (
        lf.group_by(["sample_month", "hour"])
        .agg(pl.col("kwh").sum().alias("total_kwh"))
        .sort(["sample_month", "hour"])
        .collect(engine="streaming")
    )

    # Check what months we actually have
    available_months = sorted(monthly_hourly["sample_month"].unique().to_list())
    print(f"  Available months: {', '.join(available_months)}")
    if "202507" not in available_months:
        print("  ⚠️  Note: July 2025 (202507) missing from source data")

    matrix = monthly_hourly.pivot(index="hour", columns="sample_month", values="total_kwh").fill_null(0)

    hour_labels = matrix.select("hour").to_series().to_list()
    month_cols = sorted([c for c in matrix.columns if c != "hour"])
    data_matrix = matrix.select(month_cols).to_numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.heatmap(
        data_matrix,
        cmap="RdYlBu_r",
        cbar_kws={
            "label": f"Total Energy Consumption (kWh)\n{n_customers:,} Households",
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
    )

    ax.set_xlabel("Month", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        f"Residential Electricity Load Patterns: Temporal Heat Map\n"
        f"Chicago ZIP 60622 • {date_range} • {n_customers:,} Households",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_hourly_profile(data_path: str, output_path: str = "hourly_profile_final.png"):
    """Create hourly profile with seaborn."""
    print("\n📊 Creating hourly profile...")

    lf = pl.scan_parquet(data_path)

    # Filter to Aug 2024 - Aug 2025
    desired_months = [f"2024{m:02d}" for m in range(8, 13)] + [f"2025{m:02d}" for m in range(1, 9)]
    lf = lf.filter(pl.col("sample_month").is_in(desired_months))

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

    fig, ax = plt.subplots(figsize=(15, 8))

    hours = hourly["hour"].to_list()
    mean = hourly["mean_kwh"].to_list()
    p25 = hourly["p25_kwh"].to_list()
    p75 = hourly["p75_kwh"].to_list()

    # Seaborn fill and line
    ax.fill_between(hours, p25, p75, alpha=0.3, color="#e74c3c", label="Interquartile Range")

    sns.lineplot(x=hours, y=mean, linewidth=3.5, color="#c0392b", marker="o", markersize=8, ax=ax, label="Mean Usage")

    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Energy Consumption (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        f"Average Hourly Electricity Usage\n{n_customers:,} Chicago Households • August 2024 - August 2025",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)

    ax.axvspan(0, 6, alpha=0.1, color="#3498db", label="Overnight", zorder=0)
    ax.axvspan(17, 21, alpha=0.1, color="#e74c3c", label="Evening Peak", zorder=0)

    # Annotations
    peak_hour = mean.index(max(mean))
    peak_value = max(mean)
    ax.annotate(
        f"Peak\n{peak_value:.3f} kWh\nat {peak_hour}:00",
        xy=(peak_hour, peak_value),
        xytext=(peak_hour - 4, peak_value + 0.04),
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f39c12", alpha=0.9, edgecolor="#000", linewidth=2),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#000"),
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
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#000"),
    )

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#000", fancybox=True, shadow=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_weekend_comparison(data_path: str, output_path: str = "weekend_comparison_final.png"):
    """Compare weekday vs weekend with seaborn."""
    print("\n📊 Creating weekend comparison...")

    lf = pl.scan_parquet(data_path)

    # Filter to Aug 2024 - Aug 2025
    desired_months = [f"2024{m:02d}" for m in range(8, 13)] + [f"2025{m:02d}" for m in range(1, 9)]
    lf = lf.filter(pl.col("sample_month").is_in(desired_months))

    comparison = (
        lf.group_by(["hour", "is_weekend"])
        .agg(pl.col("kwh").mean().alias("mean_kwh"))
        .sort(["is_weekend", "hour"])
        .collect(engine="streaming")
    )

    weekday = comparison.filter(pl.col("is_weekend") == False)
    weekend = comparison.filter(pl.col("is_weekend") == True)

    fig, ax = plt.subplots(figsize=(15, 9))  # Taller to avoid overlap

    weekday_hours = weekday["hour"].to_list()
    weekday_kwh = weekday["mean_kwh"].to_list()
    weekend_hours = weekend["hour"].to_list()
    weekend_kwh = weekend["mean_kwh"].to_list()

    # Seaborn lines
    sns.lineplot(
        x=weekday_hours, y=weekday_kwh, linewidth=3.5, color="#2980b9", marker="o", markersize=8, ax=ax, label="Weekday"
    )
    sns.lineplot(
        x=weekend_hours, y=weekend_kwh, linewidth=3.5, color="#e67e22", marker="s", markersize=8, ax=ax, label="Weekend"
    )

    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Average Energy (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        "Weekday vs Weekend Load Profiles\nTemporal Consumption Patterns",
        fontsize=18,
        fontweight="bold",
        pad=30,  # More padding
    )
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)

    # Set y-limit to give more headroom for annotations
    max_val = max(max(weekday_kwh), max(weekend_kwh))
    ax.set_ylim(0, max_val * 1.25)  # 25% headroom

    # Annotations - adjusted positions to avoid overlap
    weekday_peak_hour = weekday_kwh.index(max(weekday_kwh))
    weekday_peak = max(weekday_kwh)
    ax.annotate(
        f"Weekday Peak\n{weekday_peak:.3f} kWh\nat {weekday_peak_hour}:00",
        xy=(weekday_peak_hour, weekday_peak),
        xytext=(weekday_peak_hour - 4, weekday_peak + 0.03),  # Closer to point
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#3498db", alpha=0.9, edgecolor="#000", linewidth=2),
        arrowprops=dict(arrowstyle="->", lw=2, color="#000"),
    )

    weekend_peak_hour = weekend_kwh.index(max(weekend_kwh))
    weekend_peak = max(weekend_kwh)
    ax.annotate(
        f"Weekend Peak\n{weekend_peak:.3f} kWh\nat {weekend_peak_hour}:00",
        xy=(weekend_peak_hour, weekend_peak),
        xytext=(weekend_peak_hour + 3, weekend_peak + 0.03),  # Closer to point
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#e67e22", alpha=0.9, edgecolor="#000", linewidth=2),
        arrowprops=dict(arrowstyle="->", lw=2, color="#000"),
    )

    # Baseload annotations
    weekday_min = min(weekday_kwh)
    weekday_min_hour = weekday_kwh.index(weekday_min)
    ax.annotate(
        f"Weekday Low\n{weekday_min:.3f} kWh",
        xy=(weekday_min_hour, weekday_min),
        xytext=(weekday_min_hour + 2, weekday_min + 0.04),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#bdc3c7", alpha=0.9, edgecolor="#000", linewidth=1.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#000"),
    )

    weekend_min = min(weekend_kwh)
    weekend_min_hour = weekend_kwh.index(weekend_min)
    ax.annotate(
        f"Weekend Low\n{weekend_min:.3f} kWh",
        xy=(weekend_min_hour, weekend_min),
        xytext=(weekend_min_hour - 4, weekend_min + 0.04),
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
    print("FINAL PROFESSIONAL VISUALIZATIONS")
    print("=" * 80)

    data_file = "analysis/zip60622_2024/final/sample_60622_202406_202506_CLIPPED_CM90.parquet"

    if not Path(data_file).exists():
        print(f"❌ File not found: {data_file}")
        exit(1)

    print("\n✓ Using seaborn for all visualizations")
    print("✓ Filtering to August 2024 - August 2025 only")

    create_monthly_heatmap(data_file, "residential_heatmap_final.png")
    create_hourly_profile(data_file, "residential_hourly_final.png")
    create_weekend_comparison(data_file, "residential_weekend_final.png")

    print("\n" + "=" * 80)
    print("✅ VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📈 residential_heatmap_final.png")
    print("  📈 residential_hourly_final.png")
    print("  📈 residential_weekend_final.png")
    print("=" * 80)
