#!/usr/bin/env python
"""
Final visualizations: All 12 months of 2024 Chicago-wide data.
April clearly marked with caveat about limited data availability.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Seaborn style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

DATA_FILE = "analysis/chicago_2024_full_year/combined/chicago_2024_with_april_boost_CM90.parquet"
OUTPUT_DIR = Path("analysis/chicago_2024_full_year/final_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_heatmap(data_path: str, output_path: str):
    """Monthly-hourly heatmap with April footnote."""
    print("\n📊 Creating heatmap...")

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
        f"Chicago • {date_range} • {n_customers:,} Households",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    # Mark April with asterisk
    xtick_labels = ax.get_xticklabels()
    for label in xtick_labels:
        if "2024-04" in label.get_text():
            label.set_text(label.get_text() + "*")
            label.set_fontweight("bold")

    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=11)
    ax.invert_yaxis()

    # Add footnote about April
    fig.text(
        0.5,
        0.02,
        "*April 2024: Limited data availability (n=229 vs. ~950 typical) due to utility data quality issues",
        ha="center",
        fontsize=10,
        style="italic",
        color="#666666",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_hourly_profile(data_path: str, output_path: str):
    """Average hourly profile."""
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

    fig, ax = plt.subplots(figsize=(15, 8))

    hours = hourly["hour"].to_list()
    mean = hourly["mean_kwh"].to_list()
    p25 = hourly["p25_kwh"].to_list()
    p75 = hourly["p75_kwh"].to_list()

    ax.fill_between(hours, p25, p75, alpha=0.3, color="#e74c3c", label="Interquartile Range")

    sns.lineplot(x=hours, y=mean, linewidth=3.5, color="#c0392b", marker="o", markersize=8, ax=ax, label="Mean Usage")

    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Energy Consumption (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(
        f"Average Hourly Electricity Usage Profile\n{n_customers:,} Chicago Households • 2024",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)

    # Peak annotation
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

    # Baseload annotation
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


def create_monthly_profile(data_path: str, output_path: str):
    """Monthly average with April clearly marked."""
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

    fig, ax = plt.subplots(figsize=(15, 8))

    months = monthly["sample_month"].to_list()
    mean_kwh = monthly["mean_kwh"].to_list()
    customers = monthly["customers"].to_list()

    # Month labels
    month_labels = [f"{m[4:]}/{m[2:4]}" for m in months]  # MM/YY format

    # Color bars - April in different color
    colors = ["#e74c3c" if m == "202404" else "#3498db" for m in months]

    bars = ax.bar(range(len(months)), mean_kwh, color=colors, edgecolor="black", linewidth=1.5)

    # Add customer count on top of each bar
    for i, (bar, count) in enumerate(zip(bars, customers)):
        height = bar.get_height()
        label = f"n={count}"
        if months[i] == "202404":
            label += "*"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Month", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Average Energy (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Monthly Average Electricity Consumption\nChicago 2024", fontsize=18, fontweight="bold", pad=25)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(month_labels, fontsize=12)
    ax.set_ylim(0, max(mean_kwh) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="black", label="Normal months"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="April (limited data)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

    # Footnote
    fig.text(
        0.5,
        0.02,
        "*April 2024: n=229 (24% of typical) due to utility data quality issues",
        ha="center",
        fontsize=10,
        style="italic",
        color="#666666",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_weekend_comparison(data_path: str, output_path: str):
    """Weekday vs weekend."""
    print("\n📊 Creating weekend comparison...")

    lf = pl.scan_parquet(data_path)

    comparison = (
        lf.group_by(["hour", "is_weekend"])
        .agg(pl.col("kwh").mean().alias("mean_kwh"))
        .sort(["is_weekend", "hour"])
        .collect(engine="streaming")
    )

    weekday = comparison.filter(pl.col("is_weekend") == False)
    weekend = comparison.filter(pl.col("is_weekend") == True)

    fig, ax = plt.subplots(figsize=(15, 9))

    weekday_hours = weekday["hour"].to_list()
    weekday_kwh = weekday["mean_kwh"].to_list()
    weekend_hours = weekend["hour"].to_list()
    weekend_kwh = weekend["mean_kwh"].to_list()

    sns.lineplot(
        x=weekday_hours, y=weekday_kwh, linewidth=3.5, color="#2980b9", marker="o", markersize=8, ax=ax, label="Weekday"
    )
    sns.lineplot(
        x=weekend_hours, y=weekend_kwh, linewidth=3.5, color="#e67e22", marker="s", markersize=8, ax=ax, label="Weekend"
    )

    ax.set_xlabel("Hour of Day", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Average Energy (kWh per 30-min)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Weekday vs Weekend Load Profiles\nChicago 2024", fontsize=18, fontweight="bold", pad=30)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)

    max_val = max(max(weekday_kwh), max(weekend_kwh))
    ax.set_ylim(0, max_val * 1.25)

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#000", fancybox=True, shadow=True, fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("CHICAGO 2024 FINAL VISUALIZATIONS")
    print("12 Months • 10,053 Unique Customers • 55 ZIP Codes")
    print("=" * 80)

    if not Path(DATA_FILE).exists():
        print(f"❌ File not found: {DATA_FILE}")
        exit(1)

    create_heatmap(DATA_FILE, OUTPUT_DIR / "chicago_2024_heatmap.png")
    create_hourly_profile(DATA_FILE, OUTPUT_DIR / "chicago_2024_hourly_profile.png")
    create_monthly_profile(DATA_FILE, OUTPUT_DIR / "chicago_2024_monthly_profile.png")
    create_weekend_comparison(DATA_FILE, OUTPUT_DIR / "chicago_2024_weekend_comparison.png")

    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  📈 chicago_2024_heatmap.png")
    print("  📈 chicago_2024_hourly_profile.png")
    print("  📈 chicago_2024_monthly_profile.png")
    print("  📈 chicago_2024_weekend_comparison.png")
    print("\nAll figures include April 2024 data with appropriate caveats.")
    print("=" * 80)
