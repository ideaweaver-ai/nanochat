#!/bin/bash
# RunPod CUDA Workaround - Try to fix cuInit error 999

set -e

echo "=== RunPod CUDA Workaround - Fixing cuInit Error 999 ==="
echo ""

source .venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "1. Checking nvidia-uvm module (required for CUDA)..."
if [ -c /dev/nvidia-uvm ]; then
    echo "   ✓ /dev/nvidia-uvm exists"
else
    echo "   ✗ /dev/nvidia-uvm NOT found - this may be the problem!"
    echo "   The nvidia-uvm kernel module may not be loaded on the host"
fi

echo ""
echo "2. Checking all NVIDIA devices..."
ls -la /dev/nvidia* 2>/dev/null | head -10

echo ""
echo "3. Trying workaround: Use CUDA_VISIBLE_DEVICES and explicit device selection..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo ""
echo "4. Testing with explicit environment..."
python << 'PYEOF'
import os
import sys
import ctypes

# Set everything explicitly
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# Try loading libcuda with RTLD_GLOBAL
print("   Loading CUDA driver with RTLD_GLOBAL...")
libcuda = ctypes.CDLL('libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

# Try cuInit with explicit error handling
libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int

# Try multiple times (sometimes works on retry)
for attempt in range(5):
    result = libcuda.cuInit(0)
    if result == 0:
        print(f"   ✓ cuInit() succeeded on attempt {attempt+1}!")
        break
    print(f"   Attempt {attempt+1}: cuInit failed with {result}")
    if attempt < 4:
        import time
        time.sleep(0.5)

if result != 0:
    print(f"   ✗ cuInit() still fails after 5 attempts")
    print("")
    print("   THIS IS A RUNPOD CONTAINER GPU RUNTIME ISSUE!")
    print("   The container's GPU runtime is not properly configured.")
    print("   This requires RunPod support to fix.")
    print("")
    print("   Contact RunPod support with:")
    print(f"   - Pod ID: {os.uname().nodename}")
    print("   - Issue: cuInit() fails with error 999")
    print("   - GPUs allocated: 8x H100")
    print("   - nvidia-smi works but CUDA runtime cannot initialize")
    sys.exit(1)

# Get device count
libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
libcuda.cuDeviceGetCount.restype = ctypes.c_int
count = ctypes.c_int()
result = libcuda.cuDeviceGetCount(ctypes.byref(count))
if result == 0:
    print(f"   ✓ Found {count.value} GPU(s) via driver API")
else:
    print(f"   ✗ cuDeviceGetCount failed: {result}")
    sys.exit(1)

# Now test PyTorch
print("\n   Testing PyTorch...")
if 'torch' in sys.modules:
    del sys.modules['torch']
if 'torch.cuda' in sys.modules:
    del sys.modules['torch.cuda']

import torch

if torch.cuda.is_available():
    pytorch_count = torch.cuda.device_count()
    print(f"   ✓ PyTorch CUDA works! {pytorch_count} GPU(s)")
    for i in range(pytorch_count):
        print(f"     GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test operation
    x = torch.randn(2, 2).cuda()
    print(f"     ✓ GPU tensor creation successful!")
    sys.exit(0)
else:
    print("   ✗ PyTorch still can't see CUDA")
    sys.exit(1)
PYEOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ SUCCESS! CUDA IS NOW WORKING! ==="
    echo "Run speedrun.sh - it will use all 8 GPUs now!"
else
    echo ""
    echo "=== ✗ STILL NOT WORKING ==="
    echo ""
    echo "This is a RunPod container GPU runtime configuration issue."
    echo "It CANNOT be fixed from inside the container."
    echo ""
    echo "REQUIRED ACTIONS:"
    echo "1. Contact RunPod Support IMMEDIATELY"
    echo "   - Pod: $(hostname)"
    echo "   - Issue: cuInit error 999 - CUDA runtime cannot initialize"
    echo "   - You're paying \$19/hour for GPUs you can't use"
    echo ""
    echo "2. OR try recreating the pod"
    echo "   - Sometimes pods get into a bad state"
    echo "   - Stop and recreate with same template"
fi
