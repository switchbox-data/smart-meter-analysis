# Preflight Validation Checklist: 200-File Run (202307)

Target: Validate 200-file shard run before scaling to full ~30k month.

## Prerequisites

- 25-file batch (shard 100) completed successfully
- Output at: `/ebs/home/griffin_switch_box/runs/out_test_output_ec2/`
- Run artifacts at: `/ebs/home/griffin_switch_box/runs/out_test_output_ec2/_runs/202307/<run_id>/`

---

## Step 1: Run the 200-file migration

```bash
# Prepare input list (200 files from the sorted CSV inventory)
head -200 /path/to/all_csvs_202307_sorted.txt > /tmp/shard_200.txt
wc -l /tmp/shard_200.txt  # confirm 200

# Run migration (adjust paths as needed)
python scripts/csv_to_parquet/migrate_month_runner.py \
  --input-list /tmp/shard_200.txt \
  --out-root /ebs/home/griffin_switch_box/runs/out_200_preflight \
  --year-month 202307 \
  --shard-id 200 \
  --batch-size 50 \
  --workers 4 \
  --exec-mode lazy_sink \
  --fail-fast
```

Expect: 4 batch files (200 / 50 = 4 batches).

---

## Step 2: Quick sanity (before full validation)

```bash
# Confirm output structure
find /ebs/home/griffin_switch_box/runs/out_200_preflight/year=2023/month=07/ \
  -name '*.parquet' | sort

# Expected: 4 files named shard_200_batch_0000.parquet through shard_200_batch_0003.parquet

# Confirm run completed cleanly
cat /ebs/home/griffin_switch_box/runs/out_200_preflight/_runs/202307/*/run_summary.json \
  | python -m json.tool | grep -E '"total_(success|failure|skip)|batches_written|stop_requested"'

# Expected: total_failure=0, total_success=200, batches_written=4, stop_requested=false
```

---

## Step 3: Identify the run-dir

```bash
# List run directories to find the run_id
ls /ebs/home/griffin_switch_box/runs/out_200_preflight/_runs/202307/

# Set variable for convenience (replace <run_id> with actual)
RUN_DIR="/ebs/home/griffin_switch_box/runs/out_200_preflight/_runs/202307/<run_id>"
OUT_ROOT="/ebs/home/griffin_switch_box/runs/out_200_preflight"
```

---

## Step 4: Full validation (all checks)

```bash
python scripts/csv_to_parquet/validate_month_output.py \
  --out-root "$OUT_ROOT" \
  --check-mode full \
  --dst-month-check \
  --run-dir "$RUN_DIR" \
  --output-report "$RUN_DIR/validation_report_200.json"
```

This single command validates all of the following:

| Check | What it verifies |
|---|---|
| Schema contract | All 10 columns present, exact dtypes |
| Partition integrity | year=2023, month=7 in every file |
| No duplicates | No duplicate (zip_code, account_identifier, datetime) within any batch |
| Datetime invariants | No nulls, min=00:00, max=23:30, no spillover |
| DST Option B | Exactly 48 slots/day, no timestamps beyond 23:30 |
| Sortedness (full) | Lexicographic order by (zip_code, account_identifier, datetime) |
| Run artifact integrity | plan.json valid, run_summary.json clean, manifests 0 failures |
| Row counts | Per-file and total row counts reported |

Expected output on success:
```
OK: validated 4 parquet files across 1 partitions (discovered total parquet files=4, total rows validated=NNNNNN).
Validation report written to: .../validation_report_200.json
```

---

## Step 5: Review the validation report

```bash
python -m json.tool "$RUN_DIR/validation_report_200.json"
```

Checklist for the report JSON:

- [ ] `"status": "pass"`
- [ ] `"files_validated": 4`
- [ ] `"total_rows_validated"` is reasonable (expect ~200 files * ~N accounts * 48 slots * 31 days)
- [ ] `"checks_passed"` contains all 7 checks:
  - `schema_contract`
  - `partition_integrity`
  - `no_duplicates`
  - `datetime_invariants`
  - `sortedness_full`
  - `dst_option_b`
  - `run_artifact_integrity`
- [ ] `"per_file_rows"` shows all 4 batch files with non-zero row counts
- [ ] `"run_artifacts"."summary_total_failure"` is 0
- [ ] `"run_artifacts"."manifest_success_count"` is 200

---

## Step 6: Spot-check a parquet file interactively

```python
import polars as pl

f = "/ebs/home/griffin_switch_box/runs/out_200_preflight/year=2023/month=07/shard_200_batch_0000.parquet"
df = pl.read_parquet(f)

print("Shape:", df.shape)
print("Schema:", df.schema)
print("Head:\n", df.head(5))
print("Tail:\n", df.tail(5))

# Verify sort order visually
print("Sorted check:", df.select([
    pl.col("zip_code"),
    pl.col("account_identifier"),
    pl.col("datetime"),
]).head(20))

# Unique accounts
print("Unique accounts:", df["account_identifier"].n_unique())
print("Date range:", df["datetime"].min(), "to", df["datetime"].max())
```

---

## Step 7: Cross-check with 25-file run (optional determinism)

If the 200-file input list's first 25 files overlap with the original 25-file shard:

```bash
python scripts/csv_to_parquet/validate_month_output.py \
  --out-root "$OUT_ROOT" \
  --compare-root /ebs/home/griffin_switch_box/runs/out_test_output_ec2 \
  --check-mode sample
```

Note: This will only work if both roots share identical partition structure.
If shard IDs differ, compare individual batch files manually instead.

---

## Go/No-Go Decision

| Criterion | Required |
|---|---|
| Step 4 prints `OK` | YES |
| Validation report `status: pass` | YES |
| All 7 checks in `checks_passed` | YES |
| `total_rows_validated > 0` | YES |
| `run_artifacts.summary_total_failure == 0` | YES |
| `run_artifacts.manifest_success_count == 200` | YES |
| No unexpected files in output directory | YES |
| Spot-check schema + sort order looks correct | YES |

If all criteria pass: proceed to full-month sharded run.
If any fail: investigate, fix, re-run the 200-file batch.
