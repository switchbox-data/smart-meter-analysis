# Use bash
set shell := ["bash", "-cu"]

# Load .env automatically if you ever add one
set dotenv-load := true

# Default task when you run plain `just`
default: check

# -------------------- Python env & quality --------------------

# Install/sync dependencies via uv
sync:
    uv sync

# Lint & format (Ruff)
format:
    uv run ruff check --fix .
    uv run ruff format .

# Lint only
lint:
    uv run ruff check .

# Run tests (pytest)
test:
    uv run pytest -q

# Run all pre-commit hooks
precommit:
    uv run pre-commit run --all-files

# CI-style check: deps, lint, format check, tests, hooks
check:
    just sync
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q
    uv run pre-commit run --all-files

# Editable install of your package
install-dev:
    uv pip install -e .

# Clean caches/builds
clean:
    rm -rf .pytest_cache .ruff_cache build dist *.egg-info **/__pycache__

# -------------------- Ameren Data Collection --------------------

# Download Ameren CSV files and upload to S3 (interactive mode)
# Note: May require 2-3 runs due to server rate limiting
# The script automatically skips files already in S3
download-ameren:
    uv run python scripts/data_collection/ameren_scraper.py

# Download Ameren files with force flag (skip all prompts, overwrite existing)
download-ameren-force:
    uv run python scripts/data_collection/ameren_scraper.py --force

# Download Ameren files to custom S3 bucket
download-ameren-bucket BUCKET:
    uv run python scripts/data_collection/ameren_scraper.py --bucket-name {{BUCKET}} --force

echo "" >> Justfile
