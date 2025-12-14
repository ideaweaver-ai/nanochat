#!/bin/bash
# Test direct CUDA driver access to find the root cause

source .venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "=== Direct CUDA Driver Test ==="
echo ""

python << 'EOF'
import ctypes
import os
import sys

print("1. Testing libcuda.so access...")
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
        print(f"   ✓ Loaded: {path}")
        break
    except Exception as e:
        continue

if not libcuda:
    print("   ✗ Could not load libcuda.so.1")
    print("   This is the problem - CUDA driver library is missing!")
    sys.exit(1)

print("\n2. Testing cuInit...")
try:
    # Define cuInit function
    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    
    result = libcuda.cuInit(0)
    if result == 0:
        print("   ✓ cuInit() succeeded - CUDA driver is accessible!")
    else:
        print(f"   ✗ cuInit() failed with error code: {result}")
        print(f"   Error codes: 0=success, 1=missing driver, 999=unknown")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Exception calling cuInit: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing cuDeviceGetCount...")
try:
    # Define cuDeviceGetCount
    libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDeviceGetCount.restype = ctypes.c_int
    
    count = ctypes.c_int()
    result = libcuda.cuDeviceGetCount(ctypes.byref(count))
    if result == 0:
        print(f"   ✓ cuDeviceGetCount() succeeded: {count.value} GPUs")
    else:
        print(f"   ✗ cuDeviceGetCount() failed: {result}")
except Exception as e:
    print(f"   ✗ Exception: {e}")

print("\n4. Conclusion:")
print("   If cuInit works but PyTorch doesn't, it's a PyTorch-specific issue.")
print("   If cuInit fails, the container doesn't have proper CUDA driver access.")
EOF

echo ""
echo "=== Checking PyTorch's CUDA library loading ==="
python << 'EOF'
import torch
import os

# Check what libraries PyTorch is trying to load
torch_lib = os.path.dirname(torch.__file__)
print(f"PyTorch location: {torch_lib}")

# Check for CUDA libraries in PyTorch
cuda_lib_path = os.path.join(torch_lib, 'lib')
if os.path.exists(cuda_lib_path):
    print(f"\nPyTorch CUDA libraries:")
    for f in sorted(os.listdir(cuda_lib_path)):
        if 'cuda' in f.lower() or 'cudnn' in f.lower():
            full_path = os.path.join(cuda_lib_path, f)
            size = os.path.getsize(full_path) / (1024*1024)
            print(f"  {f} ({size:.1f} MB)")

# Try to see what error PyTorch gets
print("\nAttempting PyTorch CUDA access with error capture...")
try:
    import torch._C
    # Try to call the C function directly
    try:
        # This is what PyTorch calls internally
        count = torch._C._cuda_getDeviceCount()
        print(f"  Direct C call succeeded: {count}")
    except Exception as e:
        print(f"  Direct C call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
except Exception as e:
    print(f"  Could not access torch._C: {e}")
EOF
