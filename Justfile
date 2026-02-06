# Justfile for ComEd Smart Meter Analysis
# Run `just --list` to see all available commands

default:
    @just --list

# =============================================================================
# 🏗️  DEVELOPMENT ENVIRONMENT SETUP
# =============================================================================

install:
    echo "🚀 Creating virtual environment using uv"
    uv sync
    uv run pre-commit install

update:
    uv lock --upgrade

# =============================================================================
# 🔍 AWS
# =============================================================================

# Authenticate with AWS via SSO (for manual AWS CLI usage like S3 access)
# Automatically configures SSO if not already configured
aws:
    .devcontainer/devpod/aws.sh

# =============================================================================
# 🚀 DEVELOPMENT ENVIRONMENT
# =============================================================================

# Ensure Terraform is installed (internal dependency). Depends on aws so credentials
# are valid before any Terraform or infra script runs.
_terraform: aws
    bash infra/install-terraform.sh

# Set up EC2 instance (run once by admin)
# Idempotent: safe to run multiple times
dev-setup: _terraform
    bash infra/dev-setup.sh

# Destroy EC2 instance but preserve data volume (to recreate, run dev-setup again)
dev-teardown: _terraform
    bash infra/dev-teardown.sh

# Destroy everything including data volume (WARNING: destroys all data!)
dev-teardown-all: _terraform
    bash infra/dev-teardown-all.sh

# User login (run by any authorized user)
dev-login: aws
    bash infra/dev-login.sh

# =============================================================================
# 🔄 DATA PIPELINE
# =============================================================================

test-pipeline-local:
    uv run python scripts/run_comed_pipeline.py --source local

pipeline YEAR_MONTH:
    uv run python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --source s3

test-pipeline YEAR_MONTH MAX_FILES="10":
    uv run mprof run scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --max-files {{MAX_FILES}} --source s3

pipeline-skip-download YEAR_MONTH:
    uv run python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --skip-download --source s3

pipeline-debug YEAR_MONTH:
    uv run python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --debug --source s3

download-transform YEAR_MONTH MAX_FILES="":
    uv run python -m smart_meter_analysis.aws_loader {{YEAR_MONTH}} {{MAX_FILES}}

# CSV-to-Parquet migration (EC2 production). Usage: just migrate-month 202307
migrate-month YEAR_MONTH:
    #!/usr/bin/env bash
    set -euo pipefail
    OUT_ROOT="/ebs/home/griffin_switch_box/runs/out_{{YEAR_MONTH}}_production"
    INPUT_LIST="$HOME/s3_paths_{{YEAR_MONTH}}_full.txt"
    if [ ! -f "$INPUT_LIST" ]; then
        echo "ERROR: Input list not found: $INPUT_LIST" >&2
        exit 1
    fi
    echo "migrate-month {{YEAR_MONTH}}: $(wc -l < "$INPUT_LIST") inputs -> $OUT_ROOT"
    python scripts/csv_to_parquet/migrate_month_runner.py \
        --input-list "$INPUT_LIST" \
        --out-root "$OUT_ROOT" \
        --year-month {{YEAR_MONTH}} \
        --batch-size 100 \
        --workers 6 \
        --resume \
        --exec-mode lazy_sink

# =============================================================================
# 🧪 SAMPLE DATA (S3 + Synthetic)
# =============================================================================

download-samples YEAR_MONTH="202308" NUM_FILES="5":
    uv run python scripts/testing/download_samples_from_s3.py --year-month {{YEAR_MONTH}} --num-files {{NUM_FILES}}

download-samples-small YEAR_MONTH="202308":
    uv run python scripts/testing/download_samples_from_s3.py --year-month {{YEAR_MONTH}} --num-files 3

download-samples-large YEAR_MONTH="202308":
    uv run python scripts/testing/download_samples_from_s3.py --year-month {{YEAR_MONTH}} --num-files 10

generate-samples:
    uv run python scripts/testing/generate_sample_data.py

generate-samples-custom ACCOUNTS DAYS START_DATE:
    uv run python scripts/testing/generate_sample_data.py --num-accounts {{ACCOUNTS}} --num-days {{DAYS}} --start-date {{START_DATE}}

validate-local:
    uv run python scripts/diagnostics/validate_pipeline.py --input data/processed/comed_samples.parquet

inspect-dst-local:
    uv run python scripts/diagnostics/inspect_dst_days.py --input data/processed/comed_samples.parquet --start 2023-11-01 --end 2023-11-10

view-sample:
    @ls data/samples/*.csv 2>/dev/null | head -1 | xargs head -n 5 || echo "No samples found. Run: just download-samples"

clean-samples:
    rm -rf data/samples/*.csv
    @echo "Sample data cleaned"

# =============================================================================
# 🗄️  DATA COLLECTION
# =============================================================================

download-ameren:
    uv run python scripts/data_collection/ameren_scraper.py

download-ameren-force:
    uv run python scripts/data_collection/ameren_scraper.py --force

download-ameren-debug:
    uv run python scripts/data_collection/ameren_scraper.py --debug

# =============================================================================
# 🏙️ CHICAGO-WIDE SAMPLER
# =============================================================================

sample-city zips start end out bucket prefix target="200" cm90="":
    #!/usr/bin/env bash
    set -euo pipefail
    CM90="{{cm90}}"
    if [ -n "$CM90" ]; then EXTRA="--cm90 $CM90"; else EXTRA=""; fi
    python scripts/tasks/task_runner.py sample \
      --zips "{{zips}}" \
      --start "{{start}}" \
      --end "{{end}}" \
      --bucket "{{bucket}}" \
      --prefix-base "{{prefix}}" \
      --target-per-zip {{target}} \
      --out "{{out}}" \
      $EXTRA

sample-city-file zips_file start end out bucket prefix target="100" cm90="":
    #!/usr/bin/env bash
    set -euo pipefail
    CM90="{{cm90}}"
    if [ -n "$CM90" ]; then EXTRA="--cm90 $CM90"; else EXTRA=""; fi
    python scripts/tasks/task_runner.py sample \
      --zips-file "{{zips_file}}" \
      --start "{{start}}" \
      --end "{{end}}" \
      --bucket "{{bucket}}" \
      --prefix-base "{{prefix}}" \
      --target-per-zip {{target}} \
      --out "{{out}}" \
      $EXTRA

viz inp out:
    python scripts/tasks/task_runner.py viz --inp "{{inp}}" --out "{{out}}"

# =============================================================================
# 📊 BENCHMARKS (eager vs lazy)
# =============================================================================

# Run a specific benchmark: N in {100, 1000, 10000}
bench-run N MODE="lazy":
    uv run python scripts/bench/eager_vs_lazy_benchmarks.py run \
        --mode {{MODE}} \
        --n {{N}}

# Build summary CSV from stored profiles
bench-summary:
    uv run python scripts/bench/eager_vs_lazy_benchmarks.py summary

# Plot memory curves (requires existing profiles)
bench-plot:
    uv run python scripts/bench/eager_vs_lazy_benchmarks.py plot

# Run all benchmarks for eager + lazy
bench-all:
    just bench-run 100 eager
    just bench-run 100 lazy
    just bench-run 1000 eager
    just bench-run 1000 lazy
    just bench-run 10000 eager
    # lazy 10k intentionally omitted (8+ hrs)
    @echo "✔ Benchmark suite complete"

# =============================================================================
# 🔍 CODE QUALITY & TESTING
# =============================================================================

check:
    echo "🚀 Checking lock file consistency with 'pyproject.toml'"
    uv lock --locked
    echo "🚀 Linting code: Running pre-commit"
    uv run pre-commit run -a
    echo "🚀 Static type checking: Running mypy"
    uv run mypy
    echo "🚀 Checking for obsolete dependencies: Running deptry"
    uv run deptry .

test:
    echo "🚀 Testing code: Running pytest"
    uv run python -m pytest --doctest-modules

lint:
    uv run ruff check .

lint-fix:
    uv run ruff check --fix .

format:
    uv run ruff format .

typecheck:
    uv run mypy smart_meter_analysis

test-coverage:
    uv run pytest --cov=smart_meter_analysis --cov-report=html

# =============================================================================
# 📚 DOCUMENTATION
# =============================================================================

docs-test:
    uv run mkdocs build -s

docs:
    uv run mkdocs serve

docs-serve:
    uv run pdoc smart_meter_analysis

# =============================================================================
# 📊 DATA EXPLORATION
# =============================================================================

notebook:
    uv run jupyter notebook

lab:
    uv run jupyter lab

inspect-data FILE N="10":
    uv run python -c "import polars as pl; df = pl.scan_parquet('{{FILE}}').limit({{N}}).collect(); print(df)"

inspect-schema FILE:
    uv run python -c "import polars as pl; print(pl.scan_parquet('{{FILE}}').collect_schema())"

count-rows FILE:
    uv run python -c "import polars as pl; print(pl.scan_parquet('{{FILE}}').select(pl.len()).collect())"

# =============================================================================
# 🧹 UTILITIES
# =============================================================================

clean:
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf .ruff_cache
    rm -rf htmlcov
    rm -rf dist
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

clean-data:
    #!/usr/bin/env bash
    echo "This will delete processed data files!"
    echo "Raw data in S3 will not be affected."
    read -p "Are you sure? (y/N) " -n 1 -r
    if [[ $$REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/processed/*
        echo "Data cleaned"
    fi

du:
    @echo "Data directory sizes:"
    @du -sh data/* 2>/dev/null || echo "No data directories found"

# =============================================================================
# 📦 BUILD & RELEASE
# =============================================================================

clean-build:
    #!/usr/bin/env bash
    echo "🚀 Removing build artifacts"
    rm -rf dist
    echo "Removed 'dist' (if it existed)."

build: clean-build
    echo "🚀 Creating wheel file"
    uvx --from build pyproject-build --installer uv

publish:
    echo "🚀 Publishing."
    uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

build-and-publish: build publish

# =============================================================================
# 💡 EXAMPLES
# =============================================================================

example-quick:
    @echo "Step 1: Download 5 sample files from S3..."
    just download-samples-small 202308
    @echo ""
    @echo "Step 2: Run pipeline on samples..."
    just test-pipeline-local
    @echo ""
    @echo "Step 3: Inspect results..."
    just inspect-data data/processed/comed_samples.parquet 10

example-quick-offline:
    @echo "Step 1: Generate synthetic sample data..."
    just generate-samples
    @echo ""
    @echo "Step 2: Run pipeline on samples..."
    just test-pipeline-local
    @echo ""
    @echo "Step 3: Inspect results..."
    just inspect-data data/processed/comed_samples.parquet 10

example-test:
    @echo "Running test pipeline with 10 files from S3..."
    just test-pipeline 202308 10

example-full:
    @echo "Running full pipeline for August 2023..."
    @echo "This will take approximately 5-8 hours."
    just pipeline 202308

example-rerun:
    @echo "Re-running analysis on existing August 2023 data..."
    just pipeline-skip-download 202308
