#!/bin/bash
# Workaround: Use CPU mode or check if there's a container-specific issue

echo "=== CUDA Workaround Options ==="
echo ""

echo "The persistent 'CUDA unknown error' suggests a deep compatibility issue."
echo ""
echo "Possible causes:"
echo "1. Container was started without proper GPU runtime (--gpus all)"
echo "2. CUDA driver version mismatch"
echo "3. Missing CUDA driver libraries in container"
echo ""
echo "Quick checks:"
echo ""

# Check if libcuda exists
echo "1. Checking for CUDA driver library..."
if [ -f "/usr/local/nvidia/lib64/libcuda.so.1" ] || [ -f "/usr/local/nvidia/lib/libcuda.so.1" ]; then
    echo "   ✓ libcuda.so found"
else
    echo "   ✗ libcuda.so NOT found - this is the problem!"
    echo ""
    echo "   The container is missing the CUDA driver library."
    echo "   Solution: Restart container with:"
    echo "     docker run --gpus all --runtime=nvidia ..."
    echo "   OR ensure nvidia-container-toolkit is installed on host"
    exit 1
fi

# Check container runtime
echo ""
echo "2. Checking container runtime..."
if [ -f /.dockerenv ]; then
    echo "   In Docker container"
    # Check if nvidia runtime is being used
    if [ -n "$NVIDIA_VISIBLE_DEVICES" ]; then
        echo "   NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
    else
        echo "   ⚠ NVIDIA_VISIBLE_DEVICES not set"
    fi
fi

# Try a minimal CUDA test
echo ""
echo "3. Testing minimal CUDA access..."
source .venv/bin/activate 2>/dev/null || true

python << 'EOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Try to load CUDA driver directly
try:
    import ctypes
    libcuda = ctypes.CDLL('libcuda.so.1')
    
    # Define cuInit
    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    
    result = libcuda.cuInit(0)
    if result == 0:
        print("   ✓ Direct CUDA driver access works!")
        print("   This means the driver is accessible, but PyTorch can't use it")
        print("   Likely a PyTorch CUDA runtime compatibility issue")
    else:
        print(f"   ✗ cuInit failed with code: {result}")
        print("   CUDA driver is not accessible")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("   Cannot access CUDA driver")
EOF

echo ""
echo "=== Recommendations ==="
echo ""
echo "If libcuda.so is missing or cuInit fails:"
echo "  → Container needs to be restarted with proper GPU access"
echo ""
echo "If libcuda.so works but PyTorch doesn't:"
echo "  → Try PyTorch with CUDA 11.8:"
echo "    uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "  → Or check if you need to install CUDA toolkit in container:"
echo "    apt-get update && apt-get install -y cuda-toolkit-12-4"
