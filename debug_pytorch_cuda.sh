#!/bin/bash
# Debug PyTorch CUDA library loading

source .venv/bin/activate

echo "=== PyTorch CUDA Library Debug ==="
echo ""

python << 'EOF'
import torch
import os
import sys
import ctypes

print("1. PyTorch Info:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA compiled: {torch.version.cuda}")
print(f"   cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

print("\n2. Environment:")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")

print("\n3. Checking CUDA library locations...")
# Check where PyTorch expects CUDA libraries
torch_lib = os.path.dirname(torch.__file__)
cuda_lib_path = os.path.join(torch_lib, 'lib')
print(f"   PyTorch lib path: {torch_lib}")
print(f"   PyTorch CUDA lib path: {cuda_lib_path}")

if os.path.exists(cuda_lib_path):
    cuda_libs = [f for f in os.listdir(cuda_lib_path) if 'cuda' in f.lower() or 'cudnn' in f.lower()]
    print(f"   Found CUDA-related libs: {len(cuda_libs)}")
    for lib in sorted(cuda_libs)[:10]:
        print(f"     - {lib}")

print("\n4. Checking system CUDA libraries...")
lib_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
lib_paths.extend(['/usr/local/nvidia/lib64', '/usr/local/nvidia/lib', '/usr/local/cuda/lib64', '/usr/local/cuda/lib'])

cudart_found = []
for path in lib_paths:
    if path and os.path.isdir(path):
        for f in os.listdir(path):
            if 'libcudart' in f:
                full_path = os.path.join(path, f)
                if os.path.isfile(full_path):
                    cudart_found.append(full_path)
                    break

if cudart_found:
    print(f"   Found libcudart:")
    for lib in cudart_found:
        print(f"     - {lib}")
else:
    print("   ✗ No libcudart found in library paths")

print("\n5. Attempting CUDA initialization with detailed error...")
try:
    # Try to get more detailed error
    import torch._C
    print("   Attempting torch.cuda.is_available()...")
    result = torch.cuda.is_available()
    print(f"   Result: {result}")
    
    if not result:
        print("\n   Trying to get device count to see error...")
        try:
            count = torch.cuda.device_count()
            print(f"   device_count() returned: {count}")
        except Exception as e:
            print(f"   device_count() error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
except Exception as e:
    print(f"   ✗ Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Checking if we can load CUDA libraries directly...")
try:
    # Try to load libcudart directly
    for lib_path in cudart_found:
        try:
            lib = ctypes.CDLL(lib_path)
            print(f"   ✓ Successfully loaded: {lib_path}")
            break
        except Exception as e:
            print(f"   ✗ Failed to load {lib_path}: {e}")
except Exception as e:
    print(f"   Error checking libraries: {e}")

EOF
