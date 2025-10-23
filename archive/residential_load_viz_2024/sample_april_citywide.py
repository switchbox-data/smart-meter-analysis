#!/usr/bin/env python
"""
Sample ALL 55 Chicago ZIPs specifically for April 2024.
Target: Get enough customers so that even with 90% CM90 dropout, we still have ~1000.
"""

import os
import subprocess

# All 55 ZIPs with substantial April data (from investigation)
APRIL_ZIPS = [
    "60601",
    "60602",
    "60603",
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

# Already sampled in pilot
PILOT_ZIPS = ["60622", "60614", "60615", "60625", "60629", "60639", "60647", "60653", "60657", "60660"]

# Only sample new ZIPs
NEW_ZIPS = [z for z in APRIL_ZIPS if z not in PILOT_ZIPS]

# Sample 200 per ZIP to account for ~90% CM90 dropout
TARGET_PER_ZIP = 200

print("=" * 80)
print("APRIL 2024 CITY-WIDE BOOST")
print("=" * 80)
print(f"Sampling {len(NEW_ZIPS)} additional ZIPs for April only")
print(f"Target: {TARGET_PER_ZIP} customers per ZIP")
print(f"Expected raw: {len(NEW_ZIPS) * TARGET_PER_ZIP:,} samples")
print(f"Expected after CM90 (~10% pass): {len(NEW_ZIPS) * TARGET_PER_ZIP * 0.1:,.0f} customers")
print(f"Combined with pilot April: ~{len(NEW_ZIPS) * TARGET_PER_ZIP * 0.1 + 12:.0f} total April customers")
print("=" * 80)

input("\nPress ENTER to start or Ctrl+C to cancel...")

successful = []
failed = []

for i, zipcode in enumerate(NEW_ZIPS, 1):
    print(f"\n{'=' * 80}")
    print(f"[{i}/{len(NEW_ZIPS)}] ZIP {zipcode} - April 2024 only")
    print(f"{'=' * 80}")

    output_dir = f"analysis/chicago_april_citywide/zip{zipcode}"

    env = os.environ.copy()
    env.update({
        "START_YYYYMM": "202404",  # April only
        "END_YYYYMM": "202404",  # April only
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
            print(f"\n✅ ZIP {zipcode} completed")
            successful.append(zipcode)
        else:
            print(f"\n⚠️  ZIP {zipcode} failed")
            failed.append(zipcode)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted!")
        print(f"Completed: {len(successful)}")
        print(f"Remaining: {len(NEW_ZIPS) - i}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ZIP {zipcode} error: {e}")
        failed.append(zipcode)

print("\n" + "=" * 80)
print("APRIL BOOST COMPLETE")
print("=" * 80)
print(f"✅ Successful: {len(successful)}/{len(NEW_ZIPS)}")
if failed:
    print(f"❌ Failed: {len(failed)} ZIPs")
    print(f"   {', '.join(failed)}")

print("\nNext: python combine_with_april_boost.py")
print("=" * 80)
