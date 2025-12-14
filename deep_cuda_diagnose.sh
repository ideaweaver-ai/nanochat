#!/bin/bash
# Deep CUDA diagnostic to find the root cause

echo "=== Deep CUDA Diagnostic ==="
echo ""

# Check if we're in a container
echo "1. Environment Check:"
if [ -f /.dockerenv ]; then
    echo "  ✓ Running in Docker container"
elif [ -n "$container" ]; then
    echo "  ✓ Running in container (detected via \$container)"
else
    echo "  ℹ Not in container (or detection failed)"
fi
echo ""

# Check CUDA libraries
echo "2. CUDA Library Check:"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo "  Searching for CUDA libraries..."

# Check common CUDA library locations
CUDA_LIB_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/nvidia/lib64"
    "/usr/local/nvidia/lib"
)

for path in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "  ✓ Found: $path"
        if [ -f "$path/libcudart.so" ] || [ -f "$path/libcudart.so.12" ] || [ -f "$path/libcudart.so.11" ]; then
            echo "    → Contains libcudart"
        fi
        if [ -f "$path/libcublas.so" ] || [ -f "$path/libcublas.so.12" ] || [ -f "$path/libcublas.so.11" ]; then
            echo "    → Contains libcublas"
        fi
    fi
done
echo ""

# Check what PyTorch is trying to load
echo "3. PyTorch CUDA Library Loading:"
if [ -f ".venv/bin/python" ]; then
    source .venv/bin/activate
    python << 'EOF'
import torch
import os
import sys

print(f"  PyTorch version: {torch.__version__}")
print(f"  Python executable: {sys.executable}")

# Try to get more info about CUDA
try:
    print(f"  CUDA compiled version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
except Exception as e:
    print(f"  Error getting CUDA info: {e}")

# Check environment
print(f"\n  Environment variables:")
cuda_vars = [k for k in os.environ.keys() if 'CUDA' in k.upper()]
for var in sorted(cuda_vars):
    print(f"    {var}={os.environ[var]}")

# Try to initialize CUDA with more verbose output
print(f"\n  Attempting CUDA initialization...")
try:
    if torch.cuda.is_available():
        print(f"    ✓ CUDA is available")
        print(f"    GPU count: {torch.cuda.device_count()}")
    else:
        print(f"    ✗ CUDA is NOT available")
        # Try to get the actual error
        try:
            torch.cuda.device_count()
        except Exception as e:
            print(f"    Error details: {e}")
            import traceback
            traceback.print_exc()
except Exception as e:
    print(f"    ✗ Exception during CUDA check: {e}")
    import traceback
    traceback.print_exc()
EOF
fi
echo ""

# Check nvidia-smi
echo "4. NVIDIA Driver Check:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
    echo ""
    echo "  Testing GPU access:"
    nvidia-smi -q -d MEMORY | head -10
else
    echo "  ✗ nvidia-smi not found"
fi
echo ""

# Check for common issues
echo "5. Common Issue Checks:"
echo "  Checking for CUDA_VISIBLE_DEVICES conflicts..."
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "    ⚠ CUDA_VISIBLE_DEVICES is set: $CUDA_VISIBLE_DEVICES"
    echo "    Try: unset CUDA_VISIBLE_DEVICES"
else
    echo "    ✓ CUDA_VISIBLE_DEVICES not set"
fi

echo ""
echo "  Checking for library conflicts..."
if [ -f ".venv/bin/python" ]; then
    source .venv/bin/activate
    python -c "
import sys
import os
# Check if there are multiple CUDA libraries
lib_paths = []
if 'LD_LIBRARY_PATH' in os.environ:
    lib_paths.extend(os.environ['LD_LIBRARY_PATH'].split(':'))
lib_paths.extend(['/usr/local/cuda/lib64', '/usr/local/cuda/lib', '/usr/lib/x86_64-linux-gnu'])

cudart_found = []
for path in lib_paths:
    if os.path.isdir(path):
        for f in os.listdir(path):
            if 'libcudart' in f:
                cudart_found.append(os.path.join(path, f))
                break

if len(cudart_found) > 1:
    print(f'    ⚠ Multiple libcudart found: {cudart_found}')
    print(f'    This can cause conflicts')
else:
    print(f'    ✓ Found libcudart: {cudart_found[0] if cudart_found else \"none\"}')
" 2>/dev/null || echo "    Could not check library conflicts"
fi

echo ""
echo "=== Recommendations ==="
echo ""
echo "If CUDA still doesn't work, try:"
echo "1. Set LD_LIBRARY_PATH explicitly:"
echo "   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:\$LD_LIBRARY_PATH"
echo ""
echo "2. Try setting CUDA_LAUNCH_BLOCKING=1 for debugging:"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo ""
echo "3. Check if this is a container issue - may need to pass through devices:"
echo "   docker run --gpus all ..."
echo ""
echo "4. Try a simple CUDA test:"
echo "   python -c \"import torch; x = torch.randn(1).cuda(); print(x)\""
