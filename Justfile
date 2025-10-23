# ---------------- CODE QUALITY & TESTING ----------------

check:
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	uv run deptry .

test:
	@echo "🚀 Testing code: Running pytest"
	uv run python -m pytest --doctest-modules

# ---------------- DOCUMENTATION ----------------

docs-test:
	uv run mkdocs build -s

docs:
	uv run mkdocs serve

# ---------------- BUILD & RELEASE ----------------

clean-build:
	@echo "🚀 Removing build artifacts"
	rm -rf dist

build: clean-build
	@echo "🚀 Creating wheel file"
	uvx --from build pyproject-build --installer uv

publish:
	@echo "🚀 Publishing."
	uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

build-and-publish: build publish

# ---------------- DEV ENV SETUP ----------------

install:
	@echo "🚀 Creating virtual environment using uv"
	uv sync
	uv run pre-commit install

# ---------------- UTILITIES FOR YOUR PIPELINE ----------------

# List a few S3 keys for a month
list MONTH="202311" BUCKET="smart-meter-data-sb" PREFIX="sharepoint-files/Zip4/":
	uv run python - <<'PY'
	from smart_meter_analysis.aws_loader import list_s3_files
	keys = list_s3_files("{{MONTH}}", bucket="{{BUCKET}}", prefix="{{PREFIX}}", max_files=5)
	print("found:", len(keys))
	for k in keys:
	print(" •", k)
	PY

# Build BG features (strict month) with a small cap for smoke testing
smoke-features YEAR="2023" MONTH="11" CROSSWALK="data/zip4_to_bg_smoke_202311.parquet" MAX='1' MINDAYS='1' BUCKET="smart-meter-data-sb" PREFIX="sharepoint-files/Zip4/":
	uv run python -m smart_meter_analysis.bg_usage_features \
	--mode month \
	--year "{{YEAR}}" --month "{{MONTH}}" \
	--zip4_to_bg {{CROSSWALK}} \
	--bucket {{BUCKET}} \
	--prefix {{PREFIX}} \
	--out out/features_bg_{{YEAR}}{{MONTH}}.parquet \
	--quality_out out/features_bg_{{YEAR}}{{MONTH}}.quality.json \
	--max_files {{MAX}} \
	--min_days {{MINDAYS}} \
	--strict_month

# Generate the quality HTML
quality YEAR="2023" MONTH="11":
	uv run python -m smart_meter_analysis.bg_quality_report \
	--features out/features_bg_{{YEAR}}{{MONTH}}.parquet \
	--quality_json out/features_bg_{{YEAR}}{{MONTH}}.quality.json \
	--html_out out/features_bg_{{YEAR}}{{MONTH}}.report.html

# -------------------- Ameren Data Collection --------------------

# Download Ameren CSV files and upload to S3 (interactive mode)
# Uses default bucket 'smart-meter-data-sb' unless specified with download-ameren-bucket
# Note: Requires AWS credentials configured; script will fail without S3 access
# May require 2-3 runs due to server rate limiting
# Automatically skips files already in S3

download-ameren:
    uv run python scripts/data_collection/ameren_scraper.py

# Download Ameren files with force flag (skip all prompts, overwrite existing)
download-ameren-force:
    uv run python scripts/data_collection/ameren_scraper.py --force
