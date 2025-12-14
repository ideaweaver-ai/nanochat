#!/bin/bash
# Fix PyTorch to match RunPod container's CUDA 12.4.1

set -e

echo "=== Fixing PyTorch to Match RunPod Container CUDA 12.4.1 ==="
echo ""

source .venv/bin/activate

# Set library paths
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"

echo "1. Container has CUDA 12.4.1 (from runpod/pytorch:2.4.0 image)"
echo "2. Current PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "3. Installing PyTorch 2.4.0 with CUDA 12.1 (compatible with CUDA 12.4.1 runtime)..."
echo ""

# Remove current PyTorch
echo "Removing incompatible PyTorch..."
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install PyTorch 2.4.0 with CUDA 12.1 (compatible with container's CUDA 12.4.1)
echo "Installing PyTorch 2.4.0 + CUDA 12.1..."
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir

echo ""
echo "4. Testing CUDA with matching versions..."
python << 'PYEOF'
import os
import sys

# Set all paths
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'

# Force reload
if 'torch' in sys.modules:
    del sys.modules['torch']
if 'torch.cuda' in sys.modules:
    del sys.modules['torch.cuda']

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"✓ SUCCESS! CUDA is available!")
    print(f"  GPU count: {count}")
    for i in range(count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test operation
    x = torch.randn(2, 2).cuda()
    y = torch.randn(2, 2).cuda()
    z = x @ y
    print(f"  ✓ GPU computation successful: {z.shape}")
    sys.exit(0)
else:
    print("✗ CUDA still not available")
    
    # Try direct driver test
    import ctypes
    libcuda = ctypes.CDLL('libcuda.so.1')
    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    result = libcuda.cuInit(0)
    print(f"  cuInit result: {result} (0=success)")
    
    sys.exit(1)
PYEOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ SUCCESS! CUDA IS NOW WORKING! ==="
    echo "PyTorch 2.4.0 with CUDA 12.1 is compatible with your container's CUDA 12.4.1"
    echo ""
    echo "You can now run speedrun.sh and it will use all 8 H100 GPUs!"
else
    echo ""
    echo "=== Still not working ==="
    echo "There may be a deeper container issue. Contact RunPod support."
fi
