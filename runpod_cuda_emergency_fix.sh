#!/bin/bash
# RunPod Emergency CUDA Fix - Get GPUs Working NOW

set -e

echo "=== RunPod Emergency CUDA Fix ==="
echo "You're paying \$19/hour - we MUST fix this NOW!"
echo ""

source .venv/bin/activate

# Set all possible paths
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"

echo "1. RunPod Pod Information:"
echo "   Container ID: $(hostname)"
echo "   nvidia-smi GPUs: $(nvidia-smi --list-gpus 2>/dev/null | wc -l)"

echo ""
echo "2. The Problem:"
echo "   - nvidia-smi works (GPUs are accessible)"
echo "   - But cuInit() fails (CUDA runtime can't initialize)"
echo "   - This is a RunPod pod configuration issue"
echo ""

echo "3. Checking RunPod-specific settings..."
# RunPod sometimes needs explicit device selection
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
        export CUDA_VISIBLE_DEVICES=$DEVICES
        echo "   Set CUDA_VISIBLE_DEVICES=$DEVICES"
    fi
fi

echo ""
echo "4. CRITICAL: This pod may need to be recreated with GPU enabled"
echo "   Check your RunPod dashboard:"
echo "   - Is GPU enabled in the pod template?"
echo "   - Is GPU count > 0?"
echo "   - Is the correct GPU type selected?"
echo ""

echo "5. Trying workaround: Force CUDA initialization..."
python << 'PYEOF'
import os
import sys
import ctypes

# Set everything
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Try to get GPU count from nvidia-smi and set CUDA_VISIBLE_DEVICES
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
    gpu_count = len([l for l in result.stdout.strip().split('\n') if l.strip()])
    if gpu_count > 0:
        devices = ','.join(str(i) for i in range(gpu_count))
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
        print(f"   Detected {gpu_count} GPUs, set CUDA_VISIBLE_DEVICES={devices}")
except:
    pass

# Try direct driver access first
print("\n   Testing CUDA driver directly...")
libcuda = ctypes.CDLL('libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int

# Try cuInit multiple times (sometimes it works on retry)
for attempt in range(3):
    result = libcuda.cuInit(0)
    if result == 0:
        print(f"   ✓ cuInit() succeeded on attempt {attempt+1}!")
        break
    else:
        print(f"   Attempt {attempt+1}: cuInit failed with {result}")
        if attempt < 2:
            import time
            time.sleep(1)

if result != 0:
    print(f"   ✗ cuInit() still fails after 3 attempts")
    print(f"   This pod needs to be recreated with GPU enabled in RunPod template!")
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
# Force reload
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
    print(f"   ✗ PyTorch still can't see CUDA")
    sys.exit(1)
PYEOF

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== ✓ SUCCESS! CUDA IS NOW WORKING! ==="
    echo "You can now use your expensive GPUs!"
    echo ""
    echo "Run speedrun.sh again - it should detect GPUs now."
else
    echo ""
    echo "=== ✗ CUDA STILL NOT WORKING ==="
    echo ""
    echo "THIS IS A RUNPOD POD CONFIGURATION ISSUE!"
    echo ""
    echo "ACTION REQUIRED:"
    echo "1. Go to RunPod dashboard: https://www.runpod.io/console/pods"
    echo "2. Check your pod template - GPU MUST be enabled"
    echo "3. If GPU is not enabled:"
    echo "   - STOP the current pod (save money!)"
    echo "   - Create NEW pod with GPU enabled"
    echo "   - Select GPU count: 8"
    echo "   - Select GPU type: H100 (or your GPU)"
    echo ""
    echo "The pod was likely created without GPU access enabled in the template."
    echo "This is a RunPod configuration issue, not a code issue."
fi
