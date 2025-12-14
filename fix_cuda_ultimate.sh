#!/bin/bash
# Ultimate fix: Complete PyTorch reinstall with proper CUDA setup

set -e

echo "=== Ultimate CUDA Fix ==="
echo ""

source .venv/bin/activate

# Set library path from the start
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "1. Complete PyTorch removal..."
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
uv pip list | grep -E "^nvidia-|^cuda" | awk '{print $1}' | xargs -r uv pip uninstall -y 2>/dev/null || true

echo ""
echo "2. Installing PyTorch 2.5.1 with CUDA 12.1..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir

echo ""
echo "3. Installing CUDA runtime libraries..."
uv pip install \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvtx-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    --no-cache-dir

echo ""
echo "4. Testing CUDA with proper environment..."
# Create a test script that sets environment before import
cat > /tmp/test_cuda.py << 'PYEOF'
import os
import sys

# CRITICAL: Set library path BEFORE any torch import
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Also try setting CUDA paths
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_ROOT'] = '/usr/local/cuda'

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

# Try CUDA
try:
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    
    if available:
        count = torch.cuda.device_count()
        print(f"✓ SUCCESS! GPU count: {count}")
        for i in range(count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test operation
        x = torch.randn(2, 2).cuda()
        print(f"  ✓ GPU tensor creation successful: {x.device}")
        sys.exit(0)
    else:
        print("✗ CUDA not available")
        
        # Try to see what's wrong
        try:
            # Check if we can at least import the C extension
            import torch._C
            print("  torch._C imported successfully")
            
            # Try direct C call
            try:
                torch._C._cuda_getDeviceCount()
            except Exception as e:
                print(f"  C extension error: {e}")
        except Exception as e:
            print(f"  Could not import torch._C: {e}")
        
        sys.exit(1)
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

python /tmp/test_cuda.py
RESULT=$?

rm -f /tmp/test_cuda.py

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ SUCCESS! CUDA is working! ==="
    echo ""
    echo "The key was:"
    echo "1. Complete reinstall of PyTorch and CUDA runtime libraries"
    echo "2. Setting LD_LIBRARY_PATH before Python imports torch"
    echo ""
    echo "You can now run speedrun.sh"
else
    echo ""
    echo "=== Still not working ==="
    echo ""
    echo "This might be a deeper compatibility issue."
    echo "Try checking:"
    echo "  - Driver version: nvidia-smi"
    echo "  - CUDA toolkit: nvcc --version"
    echo "  - PyTorch CUDA: python -c 'import torch; print(torch.version.cuda)'"
    echo ""
    echo "Or try PyTorch with CUDA 11.8 (more compatible):"
    echo "  uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118"
fi
