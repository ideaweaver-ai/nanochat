#!/bin/bash
# Nuclear option: Reinstall PyTorch with system CUDA libraries

set -e

echo "=== Nuclear PyTorch CUDA Fix ==="
echo ""

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "✗ Virtual environment not found"
    exit 1
fi

# Check system CUDA
echo "1. Checking system CUDA..."
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "   System CUDA: $CUDA_VERSION"
else
    echo "   ⚠ nvcc not found, inferring from driver..."
    if command -v nvidia-smi &> /dev/null; then
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
        echo "   Driver version: $DRIVER"
        # Driver 570 supports CUDA 12.x
        CUDA_VERSION="12.4"
    fi
fi

echo ""
echo "2. Completely removing PyTorch..."
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
# Also remove CUDA packages
uv pip list | grep -i "nvidia\|cuda" | awk '{print $1}' | xargs -r uv pip uninstall -y 2>/dev/null || true

echo ""
echo "3. Installing PyTorch with CUDA 12.1 (most compatible)..."
# Use the official PyTorch index with CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "4. Verifying installation..."
python << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

# Set library path before checking
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    # Force reload
    import importlib
    importlib.reload(torch.cuda)
    
    if torch.cuda.is_available():
        print(f"✓ SUCCESS! CUDA is available")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        sys.exit(0)
    else:
        print(f"✗ CUDA still not available")
        print(f"\nThis might be a container GPU access issue.")
        print(f"Check if /dev/nvidia* devices exist in the container.")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ PyTorch CUDA is now working! ==="
else
    echo ""
    echo "=== ✗ PyTorch CUDA still not working ==="
    echo ""
    echo "This is likely a container GPU access issue."
    echo "Run: bash check_container_gpu.sh"
    echo ""
    echo "If /dev/nvidia* devices don't exist, the container needs"
    echo "to be restarted with --gpus all flag"
fi
