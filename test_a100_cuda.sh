#!/bin/bash
# Quick test for A100 - verify CUDA works before training

set -e

echo "=== A100 CUDA Quick Test ==="
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Set LD_LIBRARY_PATH early
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "1. Hardware check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

echo ""
echo "2. Installing PyTorch 2.4.0 with CUDA 12.1 (compatible with container's CUDA 12.4.1)..."
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
python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"✓ CUDA is available!")
    print(f"  GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # Test actual GPU operation
    print("\n  Testing GPU tensor operation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"  ✓ GPU computation successful! Result shape: {z.shape}")
    print("\n=== ✓ A100 IS READY FOR TRAINING! ===")
else:
    print("✗ CUDA is NOT available")
    print("\nTroubleshooting:")
    print("1. Check nvidia-smi output above")
    print("2. Verify container has GPU access")
    print("3. Try: bash runpod_cuda_workaround.sh")
    exit(1)
PYEOF
