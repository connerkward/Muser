#!/bin/bash
# Setup script for image-bucketing project using uv
# Run from repo root

set -e

echo "Setting up image-bucketing with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Sync dependencies
echo "Installing dependencies..."
uv sync

echo ""
echo "Setup complete! To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use uv run to execute commands:"
echo "  uv run python -m image_bucketing.cli --help"

