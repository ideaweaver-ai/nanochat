#!/bin/bash
# Deep fix for CUDA initialization issues

set -e

echo "=== Deep CUDA Fix ==="
echo ""

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "✗ Virtual environment not found"
    exit 1
fi

# Step 1: Set proper library paths
echo "1. Setting up CUDA library paths..."
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Step 2: Unset problematic environment variables
echo ""
echo "2. Cleaning environment variables..."
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging if needed

# Step 3: Verify CUDA libraries are accessible
echo ""
echo "3. Verifying CUDA libraries..."
if [ -f "/usr/local/nvidia/lib64/libcudart.so" ] || [ -f "/usr/local/nvidia/lib/libcudart.so" ]; then
    echo "   ✓ Found CUDA runtime libraries"
else
    echo "   ⚠ CUDA runtime libraries not in standard location"
    echo "   Searching..."
    find /usr -name "libcudart.so*" 2>/dev/null | head -3 || echo "   ✗ Could not find libcudart"
fi

# Step 4: Test PyTorch with explicit library loading
echo ""
echo "4. Testing PyTorch CUDA access..."
python << 'PYTHON_EOF'
import os
import sys

# Set library path before importing torch
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA compiled: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

# Try to access CUDA
try:
    if torch.cuda.is_available():
        print(f"   ✓ CUDA is available!")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        sys.exit(0)
    else:
        print(f"   ✗ CUDA is NOT available")
        # Try to get more info
        try:
            count = torch.cuda.device_count()
            print(f"   (device_count returned: {count})")
        except Exception as e:
            print(f"   Error getting device count: {e}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

CUDA_WORKS=$?

if [ $CUDA_WORKS -eq 0 ]; then
    echo ""
    echo "=== SUCCESS! CUDA is now working ==="
    echo ""
    echo "To make this permanent, add to your ~/.bashrc or script:"
    echo "export LD_LIBRARY_PATH=\"/usr/local/nvidia/lib64:/usr/local/nvidia/lib:\$LD_LIBRARY_PATH\""
else
    echo ""
    echo "=== CUDA still not working ==="
    echo ""
    echo "This might be a container/Docker issue. Try:"
    echo "1. Ensure container has GPU access: --gpus all"
    echo "2. Check if nvidia-container-toolkit is installed on host"
    echo "3. Try running: nvidia-smi (should show GPUs)"
    echo ""
    echo "Or it might be a driver/runtime mismatch. Check:"
    echo "  nvidia-smi  # Driver version"
    echo "  nvcc --version  # CUDA toolkit version"
    echo "  python -c 'import torch; print(torch.version.cuda)'  # PyTorch CUDA version"
fi
