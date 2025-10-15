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

# =============================================================================
# 📚 DOCUMENTATION
# =============================================================================

docs-test:
    uv run mkdocs build -s

docs:
    uv run mkdocs serve

# =============================================================================
# 📦 BUILD & RELEASE
# =============================================================================

clean-build:
    echo "🚀 Removing build artifacts"
    uv run python - <<'PY'
import shutil, os, pathlib
p = pathlib.Path("dist")
shutil.rmtree(p, ignore_errors=True)
print("Removed 'dist' (if it existed).")
PY

build: clean-build
    echo "🚀 Creating wheel file"
    uvx --from build pyproject-build --installer uv

publish:
    echo "🚀 Publishing."
    uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

build-and-publish: build publish

# =============================================================================
# 🏗️  DEVELOPMENT ENVIRONMENT SETUP
# =============================================================================

install:
    echo "🚀 Creating virtual environment using uv"
    uv sync
    uv run pre-commit install

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
sample-city zips start end out bucket prefix target:=200 cm90:=""
    set -euo pipefail
    CM90="{{cm90}}"
    if [ -n "$$CM90" ]; then EXTRA="--cm90 $$CM90"; else EXTRA=""; fi
    python scripts/tasks/task_runner.py sample \
      --zips "{{zips}}" \
      --start "{{start}}" \
      --end "{{end}}" \
      --bucket "{{bucket}}" \
      --prefix-base "{{prefix}}" \
      --target-per-zip {{target}} \
      --out "{{out}}" \
      $$EXTRA

# City-wide sample from a file (one ZIP per line)
sample-city-file zips_file start end out bucket prefix target:=100 cm90:=""
    set -euo pipefail
    CM90="{{cm90}}"
    if [ -n "$$CM90" ]; then EXTRA="--cm90 $$CM90"; else EXTRA=""; fi
    python scripts/tasks/task_runner.py sample \
      --zips-file "{{zips_file}}" \
      --start "{{start}}" \
      --end "{{end}}" \
      --bucket "{{bucket}}" \
      --prefix-base "{{prefix}}" \
      --target-per-zip {{target}} \
      --out "{{out}}" \
      $$EXTRA

# Build visuals from a parquet you choose
viz inp out
    python scripts/tasks/task_runner.py viz --inp "{{inp}}" --out "{{out}}"
