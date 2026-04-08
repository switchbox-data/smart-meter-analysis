"""
Rate BEST off-peak BESTEC sensitivity analysis — CUB presentation, ComEd TOU rate equity.

Rate BEST (Basic Electric Service TOU) is ComEd's proposed TOU rate. The bill has two parts:
  - Delivery: time-differentiated but fixed across all scenarios (tou_delivery_dollars)
  - Supply: computed from per-period kWh x the BESTEC (Basic Electric Service TOU Energy Charge)

Sign convention: delta = flat_bill - tou_bill. Positive delta means the household SAVES
money on Rate BEST vs. their current flat rate.

The mid-day peak BESTEC (18.080 ¢/kWh) is fixed in all scenarios; only the three
off-peak BESTECs (morning, evening, overnight) vary across scenarios.
"""

import polars as pl
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# File paths — edit these if files move
# ---------------------------------------------------------------------------
PERIOD_KWH_PATH = "/tmp/period_kwh_202301.parquet"  # noqa: S108
STOU_COMBINED_PATH = "/ebs/home/griffin_switch_box/runs/billing_output/stou_combined/stou_combined_202301.parquet"
ACCOUNT_BG_MAP_PATH = "/ebs/home/griffin_switch_box/runs/billing_output/account_bg_map_statewide.parquet"
BG_INCOME_PATH = "/ebs/home/griffin_switch_box/runs/billing_output/statewide_analysis/bg_level_stou_jan.csv"

# ---------------------------------------------------------------------------
# Rate scenarios: (label, morning_cents, evening_cents, overnight_cents)
# Mid-day peak BESTEC is 18.080 ¢/kWh — constant across all scenarios.
# ---------------------------------------------------------------------------
PEAK_BESTEC = 18.080  # ¢/kWh, mid-day peak — never changes

SCENARIOS = [
    ("Current illustrative", 4.095, 4.352, 3.278),
    ("Moderate (5.5¢ off-peak)", 5.5, 5.5, 5.5),
    ("Higher (7.0¢ off-peak)", 7.0, 7.0, 7.0),
    ("Near-flat (8.5¢ off-peak)", 8.5, 8.5, 8.5),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=== Loading data ===\n")

# Use PyArrow to read, then convert to Polars
period_kwh = pl.from_arrow(pq.read_table(PERIOD_KWH_PATH))
print(f"period_kwh:       {len(period_kwh):,} rows")

stou = pl.from_arrow(
    pq.read_table(
        STOU_COMBINED_PATH,
        columns=["account_identifier", "total_bill_a_dollars", "tou_delivery_dollars"],
    )
)
print(f"stou_combined:    {len(stou):,} rows")

account_bg = pl.from_arrow(pq.read_table(ACCOUNT_BG_MAP_PATH))
print(f"account_bg_map:   {len(account_bg):,} rows")

bg_income = pl.read_csv(BG_INCOME_PATH)
print(f"bg_level_income:  {len(bg_income):,} rows\n")


# ---------------------------------------------------------------------------
# Normalize join keys to Utf8 (some parquets use large_string)
# ---------------------------------------------------------------------------
def to_utf8(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(pl.col(col).cast(pl.Utf8))


period_kwh = to_utf8(period_kwh, "account_identifier")
stou = to_utf8(stou, "account_identifier")
account_bg = to_utf8(account_bg, "account_identifier")
account_bg = to_utf8(account_bg, "geoid_bg")
bg_income = to_utf8(bg_income, "geoid_bg")

# ---------------------------------------------------------------------------
# Join: period_kwh + stou billing + account→BG map + BG income
# ---------------------------------------------------------------------------
print("=== Joining ===\n")

df = period_kwh.join(stou, on="account_identifier", how="inner")
print(f"After join period_kwh x stou:          {len(df):,} rows")

df = df.join(account_bg.select(["account_identifier", "geoid_bg"]), on="account_identifier", how="inner")
print(f"After join x account_bg_map:           {len(df):,} rows")

df = df.join(bg_income.select(["geoid_bg", "median_household_income"]), on="geoid_bg", how="left")
print(f"After left-join x bg_income (all kept): {len(df):,} rows")
print(
    f"  Rows with non-null income:            {df.filter(pl.col('median_household_income').is_not_null()).shape[0]:,}\n"
)

# ---------------------------------------------------------------------------
# Assign income quintiles
# Method mirrors analysis.qmd: sort BGs by income, assign quintile = (i*5)//n + 1
# We sort at the household level by their BG's median income.
# ---------------------------------------------------------------------------
df_income = df.filter(pl.col("median_household_income").is_not_null())
n = len(df_income)

df_income = (
    df_income.sort("median_household_income")
    .with_row_index(name="row_i")
    .with_columns(((pl.col("row_i") * 5) // n + 1).alias("quintile"))
)

print("=== Income quintile boundaries ===\n")
quintile_bounds = (
    df_income.group_by("quintile")
    .agg(
        pl.col("median_household_income").min().alias("min_income"),
        pl.col("median_household_income").max().alias("max_income"),
        pl.len().alias("n_households"),
    )
    .sort("quintile")
)
print(quintile_bounds)
print()

# ---------------------------------------------------------------------------
# Compute savings for each scenario
# new_supply (dollars) = sum of period_kwh x BESTEC (c/kWh) / 100
# new_total  = new_supply + tou_delivery_dollars
# delta      = total_bill_a_dollars - new_total  (positive = saves money)
# ---------------------------------------------------------------------------


def compute_scenario_metrics(
    data: pl.DataFrame,
    morning: float,
    evening: float,
    overnight: float,
) -> pl.DataFrame:
    return (
        data.with_columns(
            (
                pl.col("kwh_morning") * morning / 100
                + pl.col("kwh_midday_peak") * PEAK_BESTEC / 100
                + pl.col("kwh_evening") * evening / 100
                + pl.col("kwh_overnight") * overnight / 100
            ).alias("new_supply"),
        )
        .with_columns(
            (pl.col("new_supply") + pl.col("tou_delivery_dollars")).alias("new_total"),
        )
        .with_columns(
            (pl.col("total_bill_a_dollars") - pl.col("new_total")).alias("delta"),
        )
    )


def summarize(data: pl.DataFrame, label: str) -> dict:
    """Return per-quintile + All summary metrics for one scenario."""
    # Filter zero/null flat bills before pct savings to avoid div-by-zero
    valid = data.filter(pl.col("total_bill_a_dollars").is_not_null() & (pl.col("total_bill_a_dollars") != 0))

    rows = {}
    for q in range(1, 6):
        chunk = valid.filter(pl.col("quintile") == q)
        rows[f"Q{q}"] = {
            "pct_savings": (chunk["delta"] / chunk["total_bill_a_dollars"] * 100).mean(),
            "dollar_savings": chunk["delta"].mean(),
            "pct_positive": (chunk["delta"] > 0).mean() * 100,
        }

    rows["All"] = {
        "pct_savings": (valid["delta"] / valid["total_bill_a_dollars"] * 100).mean(),
        "dollar_savings": valid["delta"].mean(),
        "pct_positive": (valid["delta"] > 0).mean() * 100,
    }
    return rows


# Collect metrics for all scenarios
all_metrics: dict[str, dict] = {}
for scenario_label, morning, evening, overnight in SCENARIOS:
    enriched = compute_scenario_metrics(df_income, morning, evening, overnight)
    all_metrics[scenario_label] = summarize(enriched, scenario_label)

# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
COLS = [f"Q{i}" for i in range(1, 6)] + ["All"]
SCENARIO_LABELS = [s[0] for s in SCENARIOS]


def print_table(title: str, metric_key: str, fmt: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    header = f"{'Scenario':<30}" + "".join(f"{c:>10}" for c in COLS)
    print(header)
    print("-" * len(header))
    for label in SCENARIO_LABELS:
        row = f"{label:<30}"
        for col in COLS:
            val = all_metrics[label][col][metric_key]
            row += f"{val:{fmt}}"
        print(row)


print_table(
    "Mean % savings vs. flat rate (positive = TOU is cheaper)",
    "pct_savings",
    ">9.1f",
)
print_table(
    "Mean dollar savings vs. flat rate (positive = TOU is cheaper)",
    "dollar_savings",
    ">9.2f",
)
print_table(
    "% of households saving money on Rate BEST vs. flat rate",
    "pct_positive",
    ">9.1f",
)
print()
