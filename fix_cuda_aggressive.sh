#!/bin/bash
# AGGRESSIVE CUDA FIX - Try everything to get GPUs working

set -e

echo "=== AGGRESSIVE CUDA FIX - We MUST get GPUs working! ==="
echo ""

source .venv/bin/activate

# Set all possible environment variables
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

echo "1. Checking container GPU runtime..."
echo "   NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES}"
echo "   NVIDIA_DRIVER_CAPABILITIES: ${NVIDIA_DRIVER_CAPABILITIES}"
echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

echo ""
echo "2. Testing direct CUDA driver access..."
python << 'EOF'
import ctypes
import os
import sys

# Try multiple ways to load libcuda
libcuda = None
for path in ['/usr/local/nvidia/lib64/libcuda.so.1', '/usr/local/nvidia/lib/libcuda.so.1', 
             '/usr/lib/x86_64-linux-gnu/libcuda.so.1', 'libcuda.so.1']:
    try:
        libcuda = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        print(f"   ✓ Loaded: {path}")
        break
    except:
        continue

if not libcuda:
    print("   ✗ Cannot load libcuda - container GPU access broken!")
    sys.exit(1)

# Try cuInit with error checking
libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int

# Get error string function
try:
    libcuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    libcuda.cuGetErrorString.restype = ctypes.c_int
    has_error_string = True
except:
    has_error_string = False

result = libcuda.cuInit(0)
if result == 0:
    print("   ✓ cuInit() SUCCEEDED!")
    # Get device count
    libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDeviceGetCount.restype = ctypes.c_int
    count = ctypes.c_int()
    result2 = libcuda.cuDeviceGetCount(ctypes.byref(count))
    if result2 == 0:
        print(f"   ✓ Found {count.value} GPU(s)")
        sys.exit(0)
    else:
        print(f"   ✗ cuDeviceGetCount failed: {result2}")
else:
    print(f"   ✗ cuInit() FAILED: {result}")
    if has_error_string:
        err_str = ctypes.c_char_p()
        libcuda.cuGetErrorString(result, ctypes.byref(err_str))
        print(f"   Error: {err_str.value.decode() if err_str.value else 'Unknown'}")
    else:
        error_codes = {
            1: "CUDA_ERROR_INVALID_VALUE",
            999: "CUDA_ERROR_UNKNOWN",
        }
        print(f"   Error code {result}: {error_codes.get(result, 'Unknown error')}")
    sys.exit(1)
EOF

DRIVER_RESULT=$?

if [ $DRIVER_RESULT -ne 0 ]; then
    echo ""
    echo "=== CRITICAL: CUDA driver cannot initialize ==="
    echo ""
    echo "The container's GPU runtime is NOT properly configured."
    echo ""
    echo "This requires the container to be restarted with:"
    echo "  docker run --gpus all --runtime=nvidia \\"
    echo "    -e NVIDIA_VISIBLE_DEVICES=all \\"
    echo "    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\"
    echo "    ..."
    echo ""
    echo "OR the host needs nvidia-container-toolkit installed and Docker restarted."
    echo ""
    echo "Since you can't restart the container, let's try one more thing..."
    echo ""
    
    # Try reinstalling PyTorch with a different method
    echo "3. Trying PyTorch reinstall with explicit CUDA paths..."
    uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --no-cache-dir
    
    echo ""
    echo "4. Final test with all environment set..."
    python << 'PYEOF'
import os
import sys

# Set everything
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_ROOT'] = '/usr/local/cuda'
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'

# Force reload
if 'torch' in sys.modules:
    del sys.modules['torch']
if 'torch.cuda' in sys.modules:
    del sys.modules['torch.cuda']

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"✓ SUCCESS! CUDA works!")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    sys.exit(0)
else:
    print("✗ CUDA still not available")
    print("")
    print("CONTAINER MUST BE RESTARTED WITH PROPER GPU RUNTIME!")
    sys.exit(1)
PYEOF

    FINAL_RESULT=$?
    
    if [ $FINAL_RESULT -eq 0 ]; then
        echo ""
        echo "=== ✓ CUDA IS NOW WORKING! ==="
    else
        echo ""
        echo "=== ✗ CUDA STILL NOT WORKING ==="
        echo ""
        echo "THIS REQUIRES CONTAINER RESTART WITH PROPER GPU FLAGS!"
        echo "Contact your cloud provider or check container startup configuration."
    fi
else
    echo ""
    echo "✓ CUDA driver works! Testing PyTorch..."
    python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import torch
if torch.cuda.is_available():
    print(f"✓ PyTorch CUDA works! {torch.cuda.device_count()} GPUs")
else:
    print("✗ PyTorch still can't see CUDA (PyTorch issue, not driver)")
PYEOF
fi
