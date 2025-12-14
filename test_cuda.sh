#!/bin/bash
# Simple CUDA test after applying fixes

echo "=== Testing CUDA Access ==="
echo ""

# Set library path
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "✗ Virtual environment not found"
    exit 1
fi

# Test CUDA
echo "Testing PyTorch CUDA..."
python << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

try:
    if torch.cuda.is_available():
        print(f"✓ SUCCESS! CUDA is available")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Try a simple operation
        print("\n  Testing GPU operation...")
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x @ y
        print(f"  ✓ GPU computation successful: {z.shape}")
        sys.exit(0)
    else:
        print("✗ FAILED: CUDA is not available")
        sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ CUDA is working correctly! ==="
    echo "You can now run speedrun.sh"
else
    echo ""
    echo "=== ✗ CUDA is still not working ==="
    echo ""
    echo "Try running the deep diagnostic:"
    echo "  bash deep_cuda_diagnose.sh"
    echo ""
    echo "Or the deep fix:"
    echo "  bash fix_cuda_deep.sh"
fi
