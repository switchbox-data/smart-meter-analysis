# Justfile for ComEd Smart Meter Analysis
# Run `just --list` to see all available commands

# Default recipe - show help
default:
    @just --list

# =============================================================================
# 🏗️  DEVELOPMENT ENVIRONMENT SETUP
# =============================================================================

install:
    echo "🚀 Creating virtual environment using uv"
    uv sync
    uv run pre-commit install

# Update dependencies
update:
    uv lock --upgrade

# =============================================================================
# 🔄 DATA PIPELINE
# =============================================================================

# Run pipeline on local sample data (fast, self-contained test)
test-pipeline-local:
    python scripts/run_comed_pipeline.py --source local

# Run full pipeline for a specific month from S3 (e.g., just pipeline 202308)
pipeline YEAR_MONTH:
    python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --source s3

# Test pipeline with limited S3 files (e.g., just test-pipeline 202308 10)
test-pipeline YEAR_MONTH MAX_FILES="10":
    python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --max-files {{MAX_FILES}} --source s3

# Run pipeline skipping download step (if data already processed)
pipeline-skip-download YEAR_MONTH:
    python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --skip-download --source s3

# Run pipeline with debug logging
pipeline-debug YEAR_MONTH:
    python scripts/run_comed_pipeline.py --year-month {{YEAR_MONTH}} --debug --source s3

# Step 1: Download and transform ComEd data from S3
download-transform YEAR_MONTH MAX_FILES="":
    python -m smart_meter_analysis.step0_aws {{YEAR_MONTH}} {{MAX_FILES}}

# =============================================================================
# 🧪 SAMPLE DATA (for testing)
# =============================================================================

# Download real sample files from S3 (default: 5 files from Aug 2023)
download-samples YEAR_MONTH="202308" NUM_FILES="5":
    python scripts/testing/download_samples_from_s3.py --year-month {{YEAR_MONTH}} --num-files {{NUM_FILES}}

# Download small sample set (3 files) - quick test
download-samples-small YEAR_MONTH="202308":
    python scripts/testing/download_samples_from_s3.py --year-month {{YEAR_MONTH}} --num-files 3

# Download larger sample set (10 files) - thorough test
download-samples-large YEAR_MONTH="202308":
    python scripts/testing/download_samples_from_s3.py --year-month {{YEAR_MONTH}} --num-files 10

# Generate synthetic sample data (for testing without S3 access)
generate-samples:
    python scripts/testing/generate_sample_data.py

# Generate sample data with custom parameters
generate-samples-custom ACCOUNTS DAYS START_DATE:
    python scripts/testing/generate_sample_data.py --num-accounts {{ACCOUNTS}} --num-days {{DAYS}} --start-date {{START_DATE}}

# View a sample CSV file
view-sample:
    @ls data/samples/*.csv 2>/dev/null | head -1 | xargs head -n 5 || echo "No samples found. Run: just download-samples"

# Clean sample data
clean-samples:
    rm -rf data/samples/*.csv
    @echo "Sample data cleaned"

# =============================================================================
# 🗄️  DATA COLLECTION
# =============================================================================

# Download Ameren CSV files and upload to S3 (interactive mode)
# Uses default bucket 'smart-meter-data-sb' unless specified
# Note: Requires AWS credentials configured; script will fail without S3 access
# May require 2-3 runs due to server rate limiting
# Automatically skips files already in S3
download-ameren:
    uv run python scripts/data_collection/ameren_scraper.py

# Download Ameren files with force flag (skip all prompts, overwrite existing)
download-ameren-force:
    uv run python scripts/data_collection/ameren_scraper.py --force

# Download Ameren files with debug logging
download-ameren-debug:
    uv run python scripts/data_collection/ameren_scraper.py --debug

# =============================================================================
# 🏗️  CHICAGO-WIDE SAMPLER
# =============================================================================
# Usage:
#   just sample-city zips=60622,60614 start=202412 end=202401 \
#        out=analysis/chicago_2024_full_year \
#        bucket=smart-meter-data-sb prefix=sharepoint-files/Zip4 \
#        target=200 cm90=0.90
#
#   just sample-city-file zips_file=zips.txt start=202412 end=202401 \
#        out=analysis/chicago_2024_full_year \
#        bucket=smart-meter-data-sb prefix=sharepoint-files/Zip4 \
#        target=100

# City-wide sample (comma-separated ZIPs)
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

# City-wide sample from a file (one ZIP per line)
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

# Build visuals from a parquet you choose
viz inp out:
    python scripts/tasks/task_runner.py viz --inp "{{inp}}" --out "{{out}}"

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

# Run linter (ruff) only
lint:
    uv run ruff check .

# Run linter and auto-fix issues
lint-fix:
    uv run ruff check --fix .

# Run formatter (ruff)
format:
    uv run ruff format .

# Run type checker (mypy) only
typecheck:
    uv run mypy smart_meter_analysis

# Run tests with coverage
test-coverage:
    uv run pytest --cov=smart_meter_analysis --cov-report=html

# =============================================================================
# 📚 DOCUMENTATION
# =============================================================================

docs-test:
    uv run mkdocs build -s

docs:
    uv run mkdocs serve

# Serve API documentation locally
docs-serve:
    uv run pdoc smart_meter_analysis

# =============================================================================
# 📊 DATA EXPLORATION
# =============================================================================

# Start Jupyter notebook
notebook:
    uv run jupyter notebook

# Start JupyterLab
lab:
    uv run jupyter lab

# Quick data inspection (shows first N rows of a parquet file)
inspect-data FILE N="10":
    python -c "import polars as pl; df = pl.scan_parquet('{{FILE}}').limit({{N}}).collect(); print(df)"

# Show schema of a parquet file
inspect-schema FILE:
    python -c "import polars as pl; print(pl.scan_parquet('{{FILE}}').collect_schema())"

# Count rows in a parquet file
count-rows FILE:
    python -c "import polars as pl; print(pl.scan_parquet('{{FILE}}').select(pl.len()).collect())"

# =============================================================================
# 🧹 UTILITIES
# =============================================================================

# Clean generated files
clean:
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf .ruff_cache
    rm -rf htmlcov
    rm -rf dist
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Clean data files (use with caution!)
clean-data:
    #!/usr/bin/env bash
    echo "This will delete processed data files!"
    echo "Raw data in S3 will not be affected."
    read -p "Are you sure? (y/N) " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/processed/*
        echo "Data cleaned"
    fi

# Show disk usage of data directories
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

# Example: Quick test with real sample data from S3
example-quick:
    @echo "Step 1: Download 5 sample files from S3..."
    just download-samples-small 202308
    @echo ""
    @echo "Step 2: Run pipeline on samples..."
    just test-pipeline-local
    @echo ""
    @echo "Step 3: Inspect results..."
    just inspect-data data/processed/comed_samples.parquet 10

# Example: Quick test with synthetic data (no S3 needed)
example-quick-offline:
    @echo "Step 1: Generate synthetic sample data..."
    just generate-samples
    @echo ""
    @echo "Step 2: Run pipeline on samples..."
    just test-pipeline-local
    @echo ""
    @echo "Step 3: Inspect results..."
    just inspect-data data/processed/comed_samples.parquet 10

# Example: Process August 2023 with 10 files for testing (requires S3)
example-test:
    @echo "Running test pipeline with 10 files from S3..."
    just test-pipeline 202308 10

# Example: Process full August 2023 (requires S3)
example-full:
    @echo "Running full pipeline for August 2023..."
    @echo "This will take approximately 5-8 hours."
    just pipeline 202308

# Example: Re-run analysis on existing data
example-rerun:
    @echo "Re-running analysis on existing August 2023 data..."
    just pipeline-skip-download 202308
