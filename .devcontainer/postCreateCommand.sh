#!/usr/bin/env bash
set -euo pipefail

echo "Starting post-create setup..."

# Always work from the workspace root inside the container
cd /workspaces/smart-meter-analysis

# Install uv only if it's not already available
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
else
  echo "uv already installed, skipping install."
fi

# Ensure uv is on PATH for this script run (covers common install locations)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Clean any corrupted venv from previous runs
echo "Cleaning any existing venv..."
rm -rf .venv

# Install dependencies (creates & manages .venv)
echo "Installing dependencies with uv sync..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install --install-hooks

# -------------------------------------------------------------------
# Auto-activate .venv for ALL future interactive bash sessions
# -------------------------------------------------------------------
BASHRC="/home/vscode/.bashrc"
SNIPPET_START="# >>> smart-meter-analysis venv auto-activate >>>"
SNIPPET_END="# <<< smart-meter-analysis venv auto-activate <<<"

# Remove old snippet if it exists (idempotent)
if grep -q "$SNIPPET_START" "$BASHRC" 2>/dev/null; then
  echo "Removing existing venv auto-activate snippet from .bashrc..."
  # delete lines between markers
  sed -i "/$SNIPPET_START/,/$SNIPPET_END/d" "$BASHRC"
fi

echo "Adding venv auto-activate snippet to .bashrc..."
cat << 'EOF' >> "$BASHRC"
# >>> smart-meter-analysis venv auto-activate >>>
if [ -d "/workspaces/smart-meter-analysis/.venv" ]; then
  cd /workspaces/smart-meter-analysis
  # only activate if not already in this venv
  if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "/workspaces/smart-meter-analysis/.venv" ]; then
    . /workspaces/smart-meter-analysis/.venv/bin/activate
  fi
fi
# <<< smart-meter-analysis venv auto-activate <<<
EOF

echo "Post-create setup completed successfully!"
