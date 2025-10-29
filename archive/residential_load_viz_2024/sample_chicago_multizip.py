#!/usr/bin/env python
"""
Sample from multiple Chicago ZIP codes for city-wide analysis.
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

# Pilot: representative sample of neighborhoods
PILOT_ZIPS = ["60622", "60614", "60615", "60625", "60629", "60639", "60647", "60653", "60657", "60660"]

# Configuration
USE_PILOT = True  # Set to False for full city
TARGET_PER_ZIP = 500 if USE_PILOT else 100
START_MONTH = "202412"  # Dec 2024 (descending, so this is start)
END_MONTH = "202406"  # Jun 2024
BASE_OUTPUT = "analysis/chicago_citywide"

ZIPS_TO_SAMPLE = PILOT_ZIPS if USE_PILOT else ALL_CHICAGO_ZIPS

print("=" * 80)
print("CHICAGO CITY-WIDE ELECTRICITY LOAD SAMPLING")
print("=" * 80)
print(f"Mode: {'PILOT (10 ZIPs)' if USE_PILOT else 'FULL CITY (53 ZIPs)'}")
print(f"Customers per ZIP: {TARGET_PER_ZIP}")
print(f"Total target: {len(ZIPS_TO_SAMPLE) * TARGET_PER_ZIP:,} customers")
print(f"Months: {END_MONTH} through {START_MONTH}")
print("=" * 80)

successful = []
failed = []

for i, zipcode in enumerate(ZIPS_TO_SAMPLE, 1):
    print(f"\n[{i}/{len(ZIPS_TO_SAMPLE)}] Sampling ZIP {zipcode}...")

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
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"   ✅ ZIP {zipcode} completed")
            successful.append(zipcode)
        else:
            print(f"   ⚠️  ZIP {zipcode} failed (code: {result.returncode})")
            print(f"   Error: {result.stderr[-200:]}")
            failed.append(zipcode)

    except Exception as e:
        print(f"   ❌ ZIP {zipcode} exception: {e}")
        failed.append(zipcode)

print("\n" + "=" * 80)
print("SAMPLING COMPLETE")
print("=" * 80)
print(f"✅ Successful: {len(successful)}/{len(ZIPS_TO_SAMPLE)} ZIPs")
print(f"❌ Failed: {len(failed)} ZIPs")

if failed:
    print(f"\nFailed ZIPs: {', '.join(failed)}")

print("\nNext step: Run combine script to merge all ZIPs into one dataset")
print("=" * 80)
