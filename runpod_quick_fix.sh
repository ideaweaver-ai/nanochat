#!/bin/bash
# QUICK FIX for RunPod - Install compatible PyTorch NOW

set -e

echo "=== RunPod Quick Fix - Get GPUs Working! ==="
echo ""

# Make sure uv is in PATH
if ! command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:${PATH}"
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:${PATH}"
fi

source .venv/bin/activate

# Set library paths
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "1. Removing incompatible PyTorch 2.8.0 (CUDA 12.8)..."
if command -v uv &> /dev/null; then
    uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
else
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
fi

echo ""
echo "2. Installing PyTorch 2.4.0 with CUDA 12.1 (compatible with your container's CUDA 12.4.1)..."
if command -v uv &> /dev/null; then
    uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --no-cache-dir
else
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --no-cache-dir
fi

echo ""
echo "3. Testing CUDA..."
python << 'EOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

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
    print(f"  ✓ GPU computation successful!")
    exit(0)
else:
    print("✗ CUDA still not available")
    exit(1)
EOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ CUDA IS NOW WORKING! ==="
    echo ""
    echo "Your 8x H100 GPUs are now accessible!"
    echo "Run speedrun.sh again - it will use all GPUs now."
else
    echo ""
    echo "=== Still not working ==="
    echo "This may require RunPod pod restart or support contact."
fi
