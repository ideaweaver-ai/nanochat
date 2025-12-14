#!/bin/bash
# RunPod-specific CUDA fix

set -e

echo "=== RunPod CUDA Fix ==="
echo ""

source .venv/bin/activate

# RunPod specific environment setup
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"

echo "1. Checking RunPod GPU configuration..."
echo "   NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-not set}"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# RunPod sometimes uses different environment variables
if [ -z "$NVIDIA_VISIBLE_DEVICES" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "   ⚠ Neither NVIDIA_VISIBLE_DEVICES nor CUDA_VISIBLE_DEVICES is set"
    echo "   Setting CUDA_VISIBLE_DEVICES=all"
    export CUDA_VISIBLE_DEVICES=all
fi

echo ""
echo "2. Checking if this is a RunPod template issue..."
echo "   RunPod pods need GPU enabled in the template settings."
echo "   If cuInit fails, the pod template may not have GPU access enabled."

echo ""
echo "3. Testing CUDA with RunPod-specific settings..."

# Try with explicit device selection
python << 'EOF'
import os
import sys

# Set all paths
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'

# RunPod specific: try setting CUDA_VISIBLE_DEVICES if not set
if 'CUDA_VISIBLE_DEVICES' not in os.environ and 'NVIDIA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = 'all'

# Force reload torch
if 'torch' in sys.modules:
    del sys.modules['torch']
if 'torch.cuda' in sys.modules:
    del sys.modules['torch.cuda']

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

# Try CUDA
try:
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    
    if available:
        count = torch.cuda.device_count()
        print(f"✓ SUCCESS! Found {count} GPU(s)")
        for i in range(count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test operation
        x = torch.randn(2, 2).cuda()
        print(f"  ✓ GPU tensor creation successful!")
        sys.exit(0)
    else:
        print("✗ CUDA still not available")
        
        # Try direct driver test
        print("\nTesting direct CUDA driver...")
        import ctypes
        try:
            libcuda = ctypes.CDLL('libcuda.so.1')
            libcuda.cuInit.argtypes = [ctypes.c_uint]
            libcuda.cuInit.restype = ctypes.c_int
            result = libcuda.cuInit(0)
            print(f"  cuInit result: {result} (0=success)")
            if result != 0:
                print(f"  ✗ cuInit failed - RunPod pod may not have GPU access enabled")
                print(f"  Check your RunPod template settings!")
        except Exception as e:
            print(f"  Error testing driver: {e}")
        
        sys.exit(1)
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ CUDA IS WORKING! ==="
    echo "You can now use your GPUs!"
else
    echo ""
    echo "=== ✗ CUDA STILL NOT WORKING ==="
    echo ""
    echo "This is a RunPod pod configuration issue."
    echo ""
    echo "SOLUTION:"
    echo "1. Go to your RunPod dashboard"
    echo "2. Check your pod template settings"
    echo "3. Ensure 'GPU' is enabled in the template"
    echo "4. Restart the pod with GPU enabled"
    echo ""
    echo "OR if using RunPod API:"
    echo "  Make sure 'gpuCount' > 0 in your pod template"
    echo ""
    echo "The pod may have been started without GPU access enabled in the template."
fi
