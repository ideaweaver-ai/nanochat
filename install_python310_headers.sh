#!/bin/bash
# Install Python 3.10 headers (venv is using Python 3.10)

set -e

echo "=== Installing Python 3.10 Development Headers ==="
echo ""

# Check if already installed
if [ -f "/usr/include/python3.10/Python.h" ]; then
    echo "✓ Python 3.10 headers already installed"
    exit 0
fi

# Install Python 3.10 headers
if command -v apt-get &> /dev/null; then
    echo "Installing python3.10-dev..."
    apt-get update -qq
    apt-get install -y python3.10-dev build-essential
    echo "✓ Installation complete"
elif command -v yum &> /dev/null; then
    echo "Installing python3.10-devel..."
    yum install -y python3.10-devel gcc gcc-c++
    echo "✓ Installation complete"
else
    echo "✗ Cannot determine package manager"
    exit 1
fi

# Verify
if [ -f "/usr/include/python3.10/Python.h" ]; then
    echo "✓ Python 3.10 headers verified at /usr/include/python3.10/Python.h"
else
    echo "✗ Headers not found after installation"
    exit 1
fi
