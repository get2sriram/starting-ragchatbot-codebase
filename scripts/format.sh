#!/bin/bash
# Format code with black
echo "Formatting code with black..."
uv run black backend/ scripts/ tests/
echo "Code formatting completed!"