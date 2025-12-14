#!/bin/bash
# Check if container has proper GPU access

echo "=== Container GPU Access Check ==="
echo ""

# Check if we're in Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running in Docker container"
    
    # Check if /dev/nvidia* devices exist
    echo ""
    echo "Checking for NVIDIA devices..."
    if ls /dev/nvidia* 1> /dev/null 2>&1; then
        echo "✓ Found NVIDIA devices:"
        ls -la /dev/nvidia* | head -5
    else
        echo "✗ NO NVIDIA devices found in /dev/"
        echo ""
        echo "⚠ CRITICAL: Container does not have GPU access!"
        echo ""
        echo "This container was likely started WITHOUT --gpus flag"
        echo ""
        echo "Solution: Restart the container with GPU access:"
        echo "  docker run --gpus all ..."
        echo "  OR"
        echo "  docker run --runtime=nvidia ..."
        echo ""
        exit 1
    fi
else
    echo "ℹ Not in Docker (or detection failed)"
    if ls /dev/nvidia* 1> /dev/null 2>&1; then
        echo "✓ Found NVIDIA devices"
        ls -la /dev/nvidia* | head -5
    else
        echo "⚠ No NVIDIA devices found"
    fi
fi

echo ""
echo "Checking nvidia-smi access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | head -3
    echo "✓ nvidia-smi works"
else
    echo "✗ nvidia-smi not found"
fi

echo ""
echo "Checking if PyTorch can see devices..."
if [ -f ".venv/bin/python" ]; then
    source .venv/bin/activate
    python << 'EOF'
import os
import sys

# Check environment
print("Environment check:")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Try importing torch
try:
    import torch
    print(f"\nPyTorch import: ✓")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA compiled: {torch.version.cuda}")
    
    # Try to access CUDA
    print(f"\nAttempting CUDA access...")
    try:
        available = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {available}")
        
        if not available:
            print(f"\n  ✗ CUDA not available")
            print(f"  This usually means:")
            print(f"    1. Container doesn't have GPU access (check /dev/nvidia*)")
            print(f"    2. Driver/runtime mismatch")
            print(f"    3. PyTorch CUDA libraries incompatible")
            
            # Try to get more info
            try:
                count = torch.cuda.device_count()
                print(f"    (device_count() returned: {count})")
            except Exception as e:
                print(f"    device_count() error: {e}")
        else:
            print(f"  ✓ CUDA is available!")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"  ✗ Exception during CUDA check: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"✗ Could not import torch: {e}")
    sys.exit(1)
EOF
else
    echo "✗ Virtual environment not found"
fi
