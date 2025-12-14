#!/bin/bash
# Install CUDA toolkit in container (if missing)

echo "=== Checking CUDA Toolkit Installation ==="
echo ""

# Check if CUDA toolkit is installed
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found"
    nvcc --version
else
    echo "✗ nvcc not found - CUDA toolkit may not be installed"
    echo ""
    echo "The container has the NVIDIA driver but may be missing CUDA toolkit."
    echo ""
    echo "To install CUDA toolkit 12.4 (matching your system):"
    echo "  apt-get update"
    echo "  apt-get install -y cuda-toolkit-12-4"
    echo ""
    read -p "Install CUDA toolkit now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing CUDA toolkit..."
        apt-get update
        apt-get install -y cuda-toolkit-12-4
        
        # Update library path
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
        echo "export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:/usr/local/cuda/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        
        echo ""
        echo "✓ CUDA toolkit installed"
        echo "  You may need to restart the container or source ~/.bashrc"
    fi
fi

echo ""
echo "Current CUDA library locations:"
find /usr/local/cuda* -name "libcudart.so*" 2>/dev/null | head -5 || echo "  No CUDA runtime libraries found in /usr/local/cuda*"
