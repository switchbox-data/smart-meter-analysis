# Justfile for ComEd Smart Meter Analysis
# Run `just --list` to see all available commands

default:
    @just --list

# =============================================================================
# 🏗️  DEVELOPMENT ENVIRONMENT SETUP
# =============================================================================

install:
    uv sync
    uv run pre-commit install

update:
    uv lock --upgrade

# =============================================================================
# 🔍 AWS
# =============================================================================

aws:
    .devcontainer/devpod/aws.sh

# =============================================================================
# 🚀 DEVELOPMENT ENVIRONMENT
# =============================================================================

_terraform: aws
    bash infra/install-terraform.sh

dev-setup: _terraform
    bash infra/dev-setup.sh

dev-teardown: _terraform
    bash infra/dev-teardown.sh

dev-teardown-all: _terraform
    bash infra/dev-teardown-all.sh

dev-login: aws
    bash infra/dev-login.sh

# =============================================================================
# 🔄 DATA PIPELINE (ANALYTICS)
# =============================================================================

pipeline YEAR_MONTH:
    uv run python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --source s3

pipeline-skip-download YEAR_MONTH:
    uv run python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --skip-download --source s3

pipeline-debug YEAR_MONTH:
    uv run python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --debug --source s3

# =============================================================================
# 🔄 CSV → PARQUET MIGRATION (PORTABLE + OPEN SOURCE SAFE)
# =============================================================================
# Operator configuration:
#   .env.comed (gitignored) may define:
#     COMED_S3_PREFIX
#     COMED_MIGRATE_OUT_BASE
#     COMED_MIGRATE_BATCH_SIZE
#     COMED_MIGRATE_WORKERS
#     CONTINUE_ON_ERROR
#     COMED_ORCHESTRATOR_LOG_DIR

S3_PREFIX            := env_var_or_default("COMED_S3_PREFIX", "")
MIGRATE_OUT_BASE     := env_var_or_default("COMED_MIGRATE_OUT_BASE", "")
MIGRATE_BATCH_SIZE   := env_var_or_default("COMED_MIGRATE_BATCH_SIZE", "100")
MIGRATE_WORKERS      := env_var_or_default("COMED_MIGRATE_WORKERS", "6")
CONTINUE_ON_ERROR    := env_var_or_default("CONTINUE_ON_ERROR", "")
ORCHESTRATOR_LOG_DIR := env_var_or_default("COMED_ORCHESTRATOR_LOG_DIR", "")
OUT_ROOT_TEMPLATE    := env_var_or_default("COMED_OUT_ROOT_TEMPLATE", "")

# -----------------------------------------------------------------------------
# List available YYYYMM months from S3
# -----------------------------------------------------------------------------

months-from-s3 OUT_FILE PREFIX=S3_PREFIX:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f ".env.comed" ]; then source ".env.comed"; fi

    prefix="{{PREFIX}}"
    if [ -z "$prefix" ]; then prefix="${COMED_S3_PREFIX:-}"; fi
    if [ -z "$prefix" ]; then
        echo "ERROR: S3 prefix not set. Use COMED_S3_PREFIX or PREFIX=..." >&2
        exit 1
    fi
    prefix="${prefix%/}/"

    AWS_PAGER="" aws s3 ls "$prefix" \
      | awk '/PRE/ {gsub(/\//,"",$2); if ($2 ~ /^[0-9]{6}$/) print $2}' \
      | sort -u > "{{OUT_FILE}}"

    echo "Wrote $(wc -l < "{{OUT_FILE}}") months to {{OUT_FILE}}"

# -----------------------------------------------------------------------------
# Single-month migration (EC2 only)
# -----------------------------------------------------------------------------

migrate-month YEAR_MONTH:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f ".env.comed" ]; then source ".env.comed"; fi

    if [ ! -d /ebs ]; then
        echo "ERROR: /ebs not found. Must run on EC2 with EBS mounted." >&2
        exit 1
    fi

    prefix="{{S3_PREFIX}}"
    if [ -z "$prefix" ]; then prefix="${COMED_S3_PREFIX:-}"; fi
    if [ -z "$prefix" ]; then
        echo "ERROR: S3 prefix not set. Use COMED_S3_PREFIX or S3_PREFIX=..." >&2
        exit 1
    fi
    prefix="${prefix%/}/"

    bucket=$(echo "$prefix" | sed 's|^s3://||' | cut -d/ -f1)

    out_base="{{MIGRATE_OUT_BASE}}"
    if [ -z "$out_base" ]; then out_base="${COMED_MIGRATE_OUT_BASE:-}"; fi
    if [ -z "$out_base" ]; then out_base="/ebs/home/$(whoami)/runs"; fi

    INPUT_LIST="$HOME/s3_paths_{{YEAR_MONTH}}_full.txt"
    OUT_ROOT="${out_base}/out_{{YEAR_MONTH}}_production"

    AWS_PAGER="" aws s3 ls "${prefix}{{YEAR_MONTH}}/" --recursive \
        | awk -v b="s3://${bucket}/" -v m="{{YEAR_MONTH}}" 'match($4,/ANONYMOUS_DATA_([0-9]{6})_/,a) && a[1]==m {print b $4}' \
        | sort -u > "$INPUT_LIST"

    if [ "$(wc -l < "$INPUT_LIST")" -eq 0 ]; then
        echo "ERROR: No CSVs found for {{YEAR_MONTH}}" >&2
        exit 1
    fi

    echo "Wrote $(wc -l < "$INPUT_LIST") CSVs to $INPUT_LIST"

    uv run python scripts/csv_to_parquet/migrate_month_runner.py \
      --input-list "$INPUT_LIST" \
      --out-root "$OUT_ROOT" \
      --year-month "{{YEAR_MONTH}}" \
      --batch-size "{{MIGRATE_BATCH_SIZE}}" \
      --workers "{{MIGRATE_WORKERS}}" \
      --resume \
      --exec-mode lazy_sink

# -----------------------------------------------------------------------------
# Multi-month migration (sequential)
# -----------------------------------------------------------------------------

migrate-months MONTHS_FILE:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f ".env.comed" ]; then source ".env.comed"; fi

    if [ ! -d /ebs ]; then
        echo "ERROR: /ebs not found. Must run on EC2." >&2
        exit 1
    fi

    log_dir="{{ORCHESTRATOR_LOG_DIR}}"
    if [ -z "$log_dir" ]; then log_dir="/ebs/home/$(whoami)/runs/_orchestrator_logs"; fi
    mkdir -p "$log_dir"

    ts=$(date -u +%Y%m%dT%H%M%SZ)
    log_file="$log_dir/migrate_${ts}.log"

    succeeded=0; failed=0; skipped=0; failures=""

    log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log_file"; }

    while IFS= read -r line || [ -n "$line" ]; do
        month=$(echo "$line" | sed 's/#.*//' | tr -d '[:space:]')
        [ -z "$month" ] && continue
        if ! echo "$month" | grep -qE '^[0-9]{6}$'; then
            log "SKIP invalid month: $month"
            skipped=$((skipped + 1))
            continue
        fi

        rc=0
        log "START $month"
        just migrate-month "$month" 2>&1 | tee -a "$log_file" || rc=$?
        log "END $month rc=$rc"

        if [ "$rc" -eq 0 ]; then
            succeeded=$((succeeded + 1))
        else
            failed=$((failed + 1))
            failures="$failures $month"
            if [ "{{CONTINUE_ON_ERROR}}" != "1" ]; then
                log "ABORT on first failure"
                break
            fi
        fi
    done < "{{MONTHS_FILE}}"

    log "DONE succeeded=$succeeded failed=$failed skipped=$skipped"
    [ "$failed" -eq 0 ]

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

validate-month YEAR_MONTH OUT_ROOT MAX_FILES="50" CHECK_MODE="sample" DST="1":
    #!/usr/bin/env bash
    set -euo pipefail
    run_base="{{OUT_ROOT}}/_runs/{{YEAR_MONTH}}"
    run_dir=$(ls -1dt "$run_base"/*/ 2>/dev/null | head -1 || true)
    run_dir="${run_dir%/}"

    if [ -z "$run_dir" ]; then
        run_dir="$run_base/_unknown"
        mkdir -p "$run_dir"
    fi

    ts=$(date -u +%Y%m%dT%H%M%SZ)
    report="$run_dir/validation_${ts}.json"

    dst_flag=""
    if [ "{{DST}}" = "1" ]; then dst_flag="--dst-month-check"; fi

    python3 scripts/csv_to_parquet/validate_month_output.py \
      --out-root "{{OUT_ROOT}}" \
      --check-mode "{{CHECK_MODE}}" \
      --max-files "{{MAX_FILES}}" \
      $dst_flag \
      --run-dir "$run_dir" \
      --output-report "$report"

    echo "Report: $report"

validate-months MONTHS_FILE OUT_BASE_DIR="/ebs/home/$(whoami)/runs":
    #!/usr/bin/env bash
    set -euo pipefail

    log_dir="{{ORCHESTRATOR_LOG_DIR}}"
    if [ -z "$log_dir" ]; then log_dir="$OUT_BASE_DIR/_orchestrator_logs"; fi
    mkdir -p "$log_dir"

    ts=$(date -u +%Y%m%dT%H%M%SZ)
    log_file="$log_dir/validate_${ts}.log"

    while read -r month; do
        [ -z "$month" ] && continue
        out_root="$OUT_BASE_DIR/out_${month}_production"
        just validate-month "$month" "$out_root" 2>&1 | tee -a "$log_file"
    done < "{{MONTHS_FILE}}"

# -----------------------------------------------------------------------------
# Status dashboard
# -----------------------------------------------------------------------------

migration-status OUT_BASE_DIR="/ebs/home/$(whoami)/runs":
    #!/usr/bin/env bash
    for d in "$OUT_BASE_DIR"/out_*_production; do
        [ -d "$d" ] || continue
        m=$(basename "$d" | grep -oE '[0-9]{6}')
        files=$(find "$d" -name "*.parquet" | wc -l)
        run=$(ls -1dt "$d/_runs/$m/"* 2>/dev/null | head -1)
        if [ -f "$run/run_summary.json" ]; then
            python3 -c 'import json; s=json.load(open("$run/run_summary.json")); print(f"{m} files={files} success={s['total_success']} failure={s['total_failure']}")'
        else
            echo "$m files=$files (no run_summary.json)"
        fi
    done
