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

# -------------------- Project-specific tasks --------------------

# Download all Ameren CSVs (your Selenium+requests script)
download-ameren:
    uv run python scripts/data_collection/ameren_scraper.py
