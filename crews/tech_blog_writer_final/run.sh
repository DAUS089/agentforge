#!/bin/bash
# Run script for tech_blog_writer_final crew

set -e

echo "Starting tech_blog_writer_final crew..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Install dependencies
echo "Installing dependencies with uv..."
uv sync

# Run the crew
echo "Running crew with arguments: $@"
uv run python -m src.tech_blog_writer_final.main "$@"
