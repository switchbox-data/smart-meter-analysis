
# List available commands
default:
    @just --list

# Run quality checks
check:
    make check

# Fetch ComEd data
fetch-data:
    python scripts/fetch_comed_data.py

# Run tests
test:
    uv run pytest tests/

# Run pre-commit on all files
format:
    uv run pre-commit run --all-files

# Validate cookies for ComEd download
validate-cookies:
    python scripts/fetch_comed_data.py validate
