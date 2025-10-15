#!/usr/bin/env python
"""
Final visualizations for Chicago smart meter data (CM90 dataset):
- Heatmap shows MEAN kWh per 30-min per customer.
- Monthly bar chart annotates each bar with the month's mean kWh.
- Hourly profile with peak/baseload annotations.
- Weekend vs weekday comparison.

Usage:
    python scripts/analysis/create_chicago_visualizations.py \
        --input analysis/chicago_2024/final/CLIPPED_CM90.parquet \
        --output analysis/chicago_2024/visualizations
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def create_heatmap(data_path: str, output_path: Path):
    """Monthly-hourly heatmap (MEAN kWh per customer)."""
    print("\n📊 Creating heatmap (MEAN kWh per customer)...")

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
        .agg(pl.col("kwh").mean().alias("mean_kwh"))
        .sort(["sample_month", "hour"])
        .collect(engine="streaming")
    )

    matrix = monthly_hourly.pivot(index="hour", columns="sample_month", values="mean_kwh").fill_null(0)

    hour_labels = matrix.select("hour").to_series().to_list()
    month_cols = sorted([c for c in matrix.columns if c != "hour"])
    data_matrix = matrix.select(month_cols).to_numpy()

    _fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(
        data_matrix,
        cmap="RdYlBu_r",
        cbar_kws={
            "label": f"Mean Energy Consumption per Customer (kWh)\n{n_customers:,} Households",
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
        "Residential Electricity Load Patterns: Temporal Heat Map\n"
        f"Chicago • {date_range} • {n_customers:,} Households",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_file = output_path / "chicago_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_hourly_profile(data_path: str, output_path: Path):
    """Average hourly profile across the year (mean and IQR)."""
    print("\n📊 Creating hourly profile...")

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

    _fig, ax = plt.subplots(figsize=(15, 8))
    hours = hourly["hour"].to_list()
    mean = hourly["mean_kwh"].to_list()
    p25 = hourly["p25_kwh"].to_list()
    p75 = hourly["p75_kwh"].to_list()

    ax.fill_between(hours, p25, p75, alpha=0.3, color="#e74c3c", label="Interquartile Range")
    sns.lineplot(x=hours, y=mean, linewidth=3.5, color="#c0392b", marker="o", markersize=8, ax=ax, label="Mean Usage")

    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Energy Consumption (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        f"Average Hourly Electricity Usage Profile\n{n_customers:,} Chicago Households",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)

    # Peak annotation
    peak_idx = mean.index(max(mean))
    peak_val = mean[peak_idx]
    ax.annotate(
        f"Peak\n{peak_val:.3f} kWh\nat {peak_idx}:00",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx - 4, peak_val + 0.04),
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#f39c12", "alpha": 0.9, "edgecolor": "#000", "linewidth": 2},
        arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#000"},
    )

    # Baseload annotation
    min_val = min(mean)
    min_idx = mean.index(min_val)
    ax.annotate(
        f"Baseload\n{min_val:.3f} kWh\nat {min_idx}:00",
        xy=(min_idx, min_val),
        xytext=(min_idx + 3, min_val + 0.06),
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#3498db", "alpha": 0.9, "edgecolor": "#000", "linewidth": 2},
        arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#000"},
    )

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#000", fancybox=True, shadow=True, fontsize=12)
    plt.tight_layout()
    output_file = output_path / "chicago_hourly_profile.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_monthly_profile(data_path: str, output_path: Path):
    """Monthly average bar chart with mean kWh annotations."""
    print("\n📊 Creating monthly profile...")

    lf = pl.scan_parquet(data_path)

    monthly = (
        lf.group_by("sample_month")
        .agg([
            pl.col("account_identifier").n_unique().alias("customers"),
            pl.col("kwh").mean().alias("mean_kwh"),
            pl.col("kwh").std().alias("std_kwh"),
        ])
        .sort("sample_month")
        .collect(engine="streaming")
    )

    _fig, ax = plt.subplots(figsize=(15, 8))
    months = monthly["sample_month"].to_list()
    mean_kwh = monthly["mean_kwh"].to_list()

    month_labels = [f"{m[4:]}/{m[2:4]}" for m in months]
    colors = ["#3498db"] * len(months)

    bars = ax.bar(range(len(months)), mean_kwh, color=colors, edgecolor="black", linewidth=1.5)

    # Annotate each bar with its average energy
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.01,
            f"{mean_kwh[i]:.3f} kWh",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Month", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Average Energy (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Monthly Average Electricity Consumption\nChicago", fontsize=18, fontweight="bold", pad=25)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(month_labels, fontsize=12)
    ax.set_ylim(0, max(mean_kwh) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_file = output_path / "chicago_monthly_profile.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_weekend_comparison(data_path: str, output_path: Path):
    """Weekday vs weekend mean kWh per 30-min."""
    print("\n📊 Creating weekend comparison...")

    lf = pl.scan_parquet(data_path)
    comparison = (
        lf.group_by(["hour", "is_weekend"])
        .agg(pl.col("kwh").mean().alias("mean_kwh"))
        .sort(["is_weekend", "hour"])
        .collect(engine="streaming")
    )

    weekday = comparison.filter(not pl.col("is_weekend"))
    weekend = comparison.filter(pl.col("is_weekend"))

    _fig, ax = plt.subplots(figsize=(15, 9))

    sns.lineplot(
        x=weekday["hour"].to_list(),
        y=weekday["mean_kwh"].to_list(),
        linewidth=3.5,
        color="#2980b9",
        marker="o",
        markersize=8,
        ax=ax,
        label="Weekday",
    )
    sns.lineplot(
        x=weekend["hour"].to_list(),
        y=weekend["mean_kwh"].to_list(),
        linewidth=3.5,
        color="#e67e22",
        marker="s",
        markersize=8,
        ax=ax,
        label="Weekend",
    )

    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Average Energy (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Weekday vs Weekend Load Profiles\nChicago", fontsize=18, fontweight="bold", pad=30)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)

    max_val = max(weekday["mean_kwh"].max(), weekend["mean_kwh"].max())
    ax.set_ylim(0, max_val * 1.25)

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#000", fancybox=True, shadow=True, fontsize=13)

    plt.tight_layout()
    output_file = output_path / "chicago_weekend_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create visualizations from Chicago smart meter data")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input parquet file (e.g., analysis/chicago_2024/final/CLIPPED_CM90.parquet)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for visualizations (e.g., analysis/chicago_2024/visualizations)",
    )

    args = parser.parse_args()

    data_path = Path(args.input)
    output_dir = Path(args.output)

    # Validate input
    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        raise SystemExit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CHICAGO SMART METER VISUALIZATIONS")
    print("=" * 80)
    print(f"Input:  {data_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Create all visualizations
    create_heatmap(str(data_path), output_dir)
    create_hourly_profile(str(data_path), output_dir)
    create_monthly_profile(str(data_path), output_dir)
    create_weekend_comparison(str(data_path), output_dir)

    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
