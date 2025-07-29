#!/usr/bin/env bash
set -e

echo "Starting post-create setup..."

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the shell to get uv in PATH
# export PATH="$HOME/.cargo/bin:$PATH"

# Install Dependencies
echo "Installing dependencies with uv sync..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install --install-hooks

echo "Post-create setup completed successfully!"
