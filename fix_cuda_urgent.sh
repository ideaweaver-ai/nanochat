#!/bin/bash
# URGENT: Fix CUDA to use expensive GPUs!

set -e

echo "=== URGENT CUDA FIX - We need those GPUs! ==="
echo ""

source .venv/bin/activate

# Set all necessary paths
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"

echo "1. Checking CUDA driver access..."
python << 'EOF'
import ctypes
import os

# Try to load and initialize CUDA driver
libcuda_paths = [
    '/usr/local/nvidia/lib64/libcuda.so.1',
    '/usr/local/nvidia/lib/libcuda.so.1',
    '/usr/lib/x86_64-linux-gnu/libcuda.so.1',
    'libcuda.so.1'
]

libcuda = None
for path in libcuda_paths:
    try:
        libcuda = ctypes.CDLL(path)
        print(f"   ✓ Loaded libcuda from: {path}")
        break
    except Exception as e:
        continue

if not libcuda:
    print("   ✗ CRITICAL: Cannot load libcuda.so.1")
    print("   The container doesn't have CUDA driver access!")
    exit(1)

# Try cuInit
libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int

result = libcuda.cuInit(0)
if result == 0:
    print("   ✓ cuInit() succeeded!")
    
    # Get device count
    libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDeviceGetCount.restype = ctypes.c_int
    count = ctypes.c_int()
    result = libcuda.cuDeviceGetCount(ctypes.byref(count))
    if result == 0:
        print(f"   ✓ Found {count.value} GPU(s)")
        exit(0)
    else:
        print(f"   ✗ cuDeviceGetCount failed: {result}")
        exit(1)
else:
    print(f"   ✗ cuInit() failed with error: {result}")
    print(f"   Error 999 = CUDA_ERROR_UNKNOWN")
    print(f"   This usually means driver/runtime mismatch or container GPU access issue")
    exit(1)
EOF

CUDA_DRIVER_WORKS=$?

if [ $CUDA_DRIVER_WORKS -ne 0 ]; then
    echo ""
    echo "=== CRITICAL: CUDA driver cannot initialize ==="
    echo ""
    echo "The container has GPU hardware but CUDA driver API cannot initialize."
    echo "This is a container configuration issue, not a PyTorch issue."
    echo ""
    echo "Possible fixes:"
    echo "1. Check if container was started with: --gpus all --runtime=nvidia"
    echo "2. Verify nvidia-container-toolkit is installed on HOST"
    echo "3. Try setting: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    echo "4. Check container logs for GPU initialization errors"
    echo ""
    echo "For now, trying workaround: Set CUDA_VISIBLE_DEVICES explicitly..."
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

echo ""
echo "2. Testing PyTorch with explicit environment..."
python << 'PYEOF'
import os
import sys

# Set all paths
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

# Force reload CUDA module
if 'torch.cuda' in sys.modules:
    del sys.modules['torch.cuda']

import torch.cuda

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
    echo "=== ✓ CUDA IS NOW WORKING! ==="
    echo "You can now use your expensive GPUs!"
else
    echo ""
    echo "=== ✗ CUDA STILL NOT WORKING ==="
    echo ""
    echo "This requires container-level fixes. The container needs:"
    echo "1. Proper GPU runtime (--gpus all --runtime=nvidia)"
    echo "2. nvidia-container-toolkit on the host"
    echo "3. Container restart with GPU flags"
    echo ""
    echo "Contact your cloud provider or check container startup configuration."
fi
