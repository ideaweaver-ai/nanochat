#!/bin/bash
# Final fix: Install PyTorch with CUDA 12.4 compatibility

set -e

echo "=== Final CUDA Fix: PyTorch for CUDA 12.4 ==="
echo ""

source .venv/bin/activate

# Set library path
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "1. Removing all PyTorch and CUDA packages..."
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
# Remove all nvidia packages
uv pip list | grep -E "^nvidia-|^cuda" | awk '{print $1}' | xargs -r uv pip uninstall -y 2>/dev/null || true

echo ""
echo "2. Installing PyTorch 2.5.1 with CUDA 12.1 (compatible with CUDA 12.4 runtime)..."
# CUDA 12.1 PyTorch builds work with CUDA 12.4 runtime
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "3. Verifying installation with library path set..."
python << 'PYEOF'
import os
import sys

# CRITICAL: Set library path BEFORE importing torch
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

# Force reload CUDA module
import importlib
if 'torch.cuda' in sys.modules:
    del sys.modules['torch.cuda']
import torch.cuda

try:
    # Try CUDA access
    print("\nAttempting CUDA access...")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    
    if available:
        count = torch.cuda.device_count()
        print(f"✓ SUCCESS! CUDA is available")
        print(f"  GPU count: {count}")
        for i in range(count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test a simple operation
        print("\n  Testing GPU operation...")
        x = torch.randn(2, 2).cuda()
        y = torch.randn(2, 2).cuda()
        z = x @ y
        print(f"  ✓ GPU computation successful: {z.shape}")
        sys.exit(0)
    else:
        print("✗ CUDA still not available")
        
        # Try to get more info
        try:
            # This might give us the actual error
            torch.cuda.init()
        except Exception as e:
            print(f"  Error during init: {e}")
            import traceback
            traceback.print_exc()
        
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ SUCCESS! CUDA is now working! ==="
    echo ""
    echo "The fix was:"
    echo "1. Setting LD_LIBRARY_PATH before importing torch"
    echo "2. Using PyTorch 2.5.1 with CUDA 12.1 (compatible with CUDA 12.4)"
    echo ""
    echo "You can now run speedrun.sh"
else
    echo ""
    echo "=== Still not working ==="
    echo ""
    echo "This might require checking PyTorch's internal CUDA loading."
    echo "Run: bash debug_pytorch_cuda.sh"
fi
