#!/bin/bash
# Find where Python headers actually are

echo "=== Finding Python Headers ==="
echo ""

# System Python
echo "System Python:"
python3 --version 2>/dev/null || echo "  Not found"
SYSTEM_PYTHON=$(python3 --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "")
echo ""

# Venv Python (if exists)
if [ -d ".venv" ]; then
    echo "Venv Python:"
    .venv/bin/python --version 2>/dev/null || echo "  Not found"
    VENV_PYTHON=$(.venv/bin/python --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "")
    echo ""
fi

# Find all Python.h files
echo "Searching for Python.h files:"
find /usr/include -name "Python.h" 2>/dev/null | head -10
find /usr/local/include -name "Python.h" 2>/dev/null | head -10
echo ""

# Check common locations
echo "Checking common locations:"
for version in "3.10" "3.11" "3.12" "$SYSTEM_PYTHON" "$VENV_PYTHON"; do
    if [ -n "$version" ]; then
        for base in "/usr/include" "/usr/local/include"; do
            path="${base}/python${version}/Python.h"
            if [ -f "$path" ]; then
                echo "  ✓ Found: $path"
            fi
        done
    fi
done
echo ""

# Check what Python version the venv is using
if [ -d ".venv" ]; then
    echo "Venv Python executable:"
    .venv/bin/python -c "import sys; print(f'  {sys.executable}')"
    .venv/bin/python -c "import sys; print(f'  Version: {sys.version_info.major}.{sys.version_info.minor}')"
    .venv/bin/python -c "import sysconfig; print(f'  Include dir: {sysconfig.get_path(\"include\")}')" 2>/dev/null || echo "  Could not get include dir"
fi
