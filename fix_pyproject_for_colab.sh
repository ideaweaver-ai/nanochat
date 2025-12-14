#!/bin/bash

# Fix pyproject.toml to use CUDA 11.8 instead of 12.8
# This prevents uv sync from installing the wrong PyTorch version

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

if [ ! -f "pyproject.toml" ]; then
    echo "❌ ERROR: pyproject.toml not found!"
    exit 1
fi

echo "Fixing pyproject.toml to use CUDA 11.8..."

# Backup original
cp pyproject.toml pyproject.toml.backup

# Replace CUDA 12.8 with CUDA 11.8
sed -i 's/pytorch-cu128/pytorch-cu118/g' pyproject.toml
sed -i 's/cu128/cu118/g' pyproject.toml

echo "✅ Fixed pyproject.toml"
echo "   Changed: pytorch-cu128 → pytorch-cu118"
echo "   Changed: cu128 → cu118"
echo ""
echo "Backup saved to: pyproject.toml.backup"
echo ""
echo "Now run: uv sync --extra gpu"
echo "It will install PyTorch with CUDA 11.8 from the start!"
