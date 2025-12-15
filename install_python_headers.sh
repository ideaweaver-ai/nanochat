#!/bin/bash
# Quick fix: Install Python development headers

set -e

echo "=== Installing Python Development Headers ==="
echo ""

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "3.10")
echo "Detected Python version: $PYTHON_VERSION"

# Check if headers already exist
if [ -f "/usr/include/python${PYTHON_VERSION}/Python.h" ] || \
   [ -f "/usr/include/python3.10/Python.h" ] || \
   [ -f "/usr/include/python3.11/Python.h" ] || \
   [ -f "/usr/include/python3.12/Python.h" ]; then
    echo "✓ Python headers already installed"
    exit 0
fi

# Install headers
if command -v apt-get &> /dev/null; then
    echo "Installing python3-dev and build-essential..."
    apt-get update -qq
    apt-get install -y python3-dev build-essential
    echo "✓ Installation complete"
elif command -v yum &> /dev/null; then
    echo "Installing python3-devel and build tools..."
    yum install -y python3-devel gcc gcc-c++
    echo "✓ Installation complete"
else
    echo "✗ Cannot determine package manager. Please install manually:"
    echo "  apt-get install -y python3-dev build-essential"
    exit 1
fi

# Verify
if [ -f "/usr/include/python${PYTHON_VERSION}/Python.h" ] || \
   [ -f "/usr/include/python3.10/Python.h" ] || \
   [ -f "/usr/include/python3.11/Python.h" ] || \
   [ -f "/usr/include/python3.12/Python.h" ]; then
    echo "✓ Python headers verified"
else
    echo "⚠ Warning: Headers may not be in expected location"
    echo "  Training will skip torch.compile() and use eager mode"
fi
