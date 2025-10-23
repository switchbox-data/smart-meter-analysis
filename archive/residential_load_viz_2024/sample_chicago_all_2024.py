#!/usr/bin/env python
"""
Sample ALL 12 months of 2024 from multiple Chicago ZIP codes.
"""

import os
import subprocess

# All 53 Chicago ZIPs with substantial data
ALL_CHICAGO_ZIPS = [
    "60601",
    "60605",
    "60606",
    "60607",
    "60608",
    "60609",
    "60610",
    "60611",
    "60612",
    "60613",
    "60614",
    "60615",
    "60616",
    "60617",
    "60618",
    "60619",
    "60620",
    "60621",
    "60622",
    "60623",
    "60624",
    "60625",
    "60626",
    "60628",
    "60629",
    "60630",
    "60631",
    "60632",
    "60633",
    "60634",
    "60636",
    "60637",
    "60638",
    "60639",
    "60640",
    "60641",
    "60642",
    "60643",
    "60644",
    "60645",
    "60646",
    "60647",
    "60649",
    "60651",
    "60652",
    "60653",
    "60654",
    "60655",
    "60656",
    "60657",
    "60659",
    "60660",
    "60661",
]

# Pilot: 10 diverse neighborhoods
PILOT_ZIPS = ["60622", "60614", "60615", "60625", "60629", "60639", "60647", "60653", "60657", "60660"]

# Configuration
USE_PILOT = True  # Set to False for all 53 ZIPs
TARGET_PER_ZIP = 100  # 100 customers per ZIP per month
START_MONTH = "202412"  # December (descending order, so this is "start")
END_MONTH = "202401"  # January - ALL 12 MONTHS OF 2024
BASE_OUTPUT = "analysis/chicago_2024_full_year"

ZIPS_TO_SAMPLE = PILOT_ZIPS if USE_PILOT else ALL_CHICAGO_ZIPS

print("=" * 80)
print("CHICAGO CITY-WIDE: ALL 12 MONTHS OF 2024")
print("=" * 80)
print(f"Mode: {'PILOT (10 ZIPs)' if USE_PILOT else 'FULL CITY (53 ZIPs)'}")
print(f"Customers per ZIP: {TARGET_PER_ZIP}")
print(f"Total target: {len(ZIPS_TO_SAMPLE) * TARGET_PER_ZIP:,} customers per month")
print("Months: January 2024 through December 2024 (ALL 12 MONTHS)")
print(f"Expected total rows: ~{len(ZIPS_TO_SAMPLE) * TARGET_PER_ZIP * 12 * 48 * 30:,}")
print("=" * 80)

input(f"\nThis will sample {len(ZIPS_TO_SAMPLE)} ZIPs. Press ENTER to start or Ctrl+C to cancel...")

successful = []
failed = []

for i, zipcode in enumerate(ZIPS_TO_SAMPLE, 1):
    print(f"\n{'=' * 80}")
    print(f"[{i}/{len(ZIPS_TO_SAMPLE)}] Processing ZIP {zipcode}...")
    print(f"{'=' * 80}")

    output_dir = f"{BASE_OUTPUT}/zip{zipcode}"

    env = os.environ.copy()
    env.update({
        "START_YYYYMM": START_MONTH,
        "END_YYYYMM": END_MONTH,
        "OUTPUT_DIR": output_dir,
        "TARGET_CUSTOMERS": str(TARGET_PER_ZIP),
        "ZIP5": zipcode,
    })

    try:
        result = subprocess.run(
            ["python", "sample_customers_60622.py"],
            env=env,
        )

        if result.returncode == 0:
            print(f"\n✅ ZIP {zipcode} completed successfully")
            successful.append(zipcode)
        else:
            print(f"\n⚠️  ZIP {zipcode} failed (code: {result.returncode})")
            failed.append(zipcode)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        print(f"Completed: {len(successful)} ZIPs")
        print(f"Failed: {len(failed)} ZIPs")
        print(f"Remaining: {len(ZIPS_TO_SAMPLE) - i} ZIPs")
        exit(1)
    except Exception as e:
        print(f"\n❌ ZIP {zipcode} exception: {e}")
        failed.append(zipcode)

print("\n" + "=" * 80)
print("SAMPLING COMPLETE")
print("=" * 80)
print(f"✅ Successful: {len(successful)}/{len(ZIPS_TO_SAMPLE)} ZIPs")
if successful:
    print(f"   {', '.join(successful)}")

if failed:
    print(f"\n❌ Failed: {len(failed)} ZIPs")
    print(f"   {', '.join(failed)}")

print("\nNext step: python combine_chicago_zips.py")
print("=" * 80)
