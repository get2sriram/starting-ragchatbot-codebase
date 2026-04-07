#!/bin/bash
# Install development dependencies
echo "Installing development dependencies..."
uv pip install -e .[dev]
echo "Development dependencies installed successfully!"