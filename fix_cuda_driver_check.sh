#!/bin/bash
# Check CUDA driver compatibility and try different PyTorch versions

set -e

source .venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "=== CUDA Driver Compatibility Check ==="
echo ""

# Check driver version
echo "1. System Information:"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "   Driver version: $DRIVER_VERSION"
    
    # Check CUDA version supported by driver
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1)
    echo "   CUDA version (driver): $CUDA_VERSION"
fi

if command -v nvcc &> /dev/null; then
    CUDA_TOOLKIT=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "   CUDA toolkit: $CUDA_TOOLKIT"
fi

echo ""
echo "2. Current PyTorch:"
python -c "import torch; print(f'   Version: {torch.__version__}'); print(f'   CUDA compiled: {torch.version.cuda}')" 2>/dev/null || echo "   PyTorch not installed"

echo ""
echo "3. Testing direct CUDA driver access..."
python << 'EOF'
import ctypes
import os

# Try to load libcuda directly
libcuda_paths = [
    '/usr/local/nvidia/lib64/libcuda.so.1',
    '/usr/local/nvidia/lib/libcuda.so.1',
    '/usr/lib/x86_64-linux-gnu/libcuda.so.1',
    'libcuda.so.1'
]

libcuda = None
for path in libcuda_paths:
    try:
        libcuda = ctypes.CDLL(path)
        print(f"   ✓ Loaded libcuda from: {path}")
        break
    except:
        continue

if libcuda:
    try:
        # Try to call cuInit
        result = libcuda.cuInit(0)
        if result == 0:
            print("   ✓ cuInit() succeeded - CUDA driver is accessible")
        else:
            print(f"   ✗ cuInit() failed with error code: {result}")
    except Exception as e:
        print(f"   ✗ Could not call cuInit: {e}")
else:
    print("   ✗ Could not load libcuda.so")
EOF

echo ""
echo "4. Trying PyTorch with CUDA 11.8 (more compatible with older drivers)..."
read -p "   Install PyTorch 2.4.0 with CUDA 11.8? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Removing current PyTorch..."
    uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    
    echo "   Installing PyTorch 2.4.0 with CUDA 11.8..."
    uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu118 \
        --no-cache-dir
    
    echo ""
    echo "   Testing CUDA 11.8 PyTorch..."
    python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA compiled: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"   ✓ SUCCESS! CUDA is available")
    print(f"   GPU count: {torch.cuda.device_count()}")
else:
    print(f"   ✗ CUDA still not available")
PYEOF
fi
