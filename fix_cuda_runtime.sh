#!/bin/bash
# Install CUDA runtime libraries that PyTorch needs

set -e

echo "=== Installing CUDA Runtime Libraries for PyTorch ==="
echo ""

source .venv/bin/activate

# Check what CUDA runtime PyTorch expects
echo "1. Checking PyTorch CUDA requirements..."
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA compiled: {torch.version.cuda}")
EOF

echo ""
echo "2. PyTorch 2.5.1 with CUDA 12.1 needs these runtime libraries:"
echo "   - nvidia-cuda-runtime-cu12"
echo "   - nvidia-cudnn-cu12"
echo "   - nvidia-cublas-cu12"
echo "   - And other nvidia-* packages"

echo ""
echo "3. Checking current nvidia packages..."
uv pip list | grep -E "^nvidia-|^cuda" || echo "   None found"

echo ""
echo "4. Installing/updating CUDA runtime packages..."
# Install the exact versions that PyTorch 2.5.1+cu121 needs
uv pip install --upgrade \
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
    nvidia-cuda-nvrtc-cu12

echo ""
echo "5. Verifying installation..."
python << 'PYEOF'
import os
# Set library path
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

# Try to import CUDA
try:
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("✗ CUDA still not available")
        # Try to get more info
        import sys
        print("\nTrying to get detailed error...")
        try:
            # Force initialization
            torch.cuda.init()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
PYEOF
