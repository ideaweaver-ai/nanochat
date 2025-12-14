#!/bin/bash
# Final diagnostic - this is a RunPod container GPU runtime issue

echo "=== FINAL DIAGNOSTIC - RunPod GPU Runtime Issue ==="
echo ""

source .venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo "1. Hardware Check:"
echo "   GPUs visible via nvidia-smi: $(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
echo "   /dev/nvidia* devices: $(ls /dev/nvidia* 2>/dev/null | wc -l)"

echo ""
echo "2. CUDA Driver Test:"
python << 'EOF'
import ctypes
import sys

libcuda = ctypes.CDLL('libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int

result = libcuda.cuInit(0)
print(f"   cuInit() result: {result}")
if result == 0:
    print("   ✓ Driver API works!")
    sys.exit(0)
else:
    print(f"   ✗ cuInit fails with error {result}")
    print("   This is a CONTAINER GPU RUNTIME issue, not PyTorch!")
    print("   The container's GPU runtime is not properly configured.")
    sys.exit(1)
EOF

DRIVER_RESULT=$?

echo ""
echo "3. Conclusion:"
if [ $DRIVER_RESULT -ne 0 ]; then
    echo "   ✗ CRITICAL: CUDA driver cannot initialize"
    echo ""
    echo "   This is NOT a PyTorch issue - it's a RunPod container GPU runtime issue."
    echo ""
    echo "   REQUIRED ACTIONS:"
    echo "   1. Contact RunPod support immediately"
    echo "      - Pod ID: $(hostname)"
    echo "      - Issue: cuInit() fails with error 999"
    echo "      - GPUs allocated but CUDA runtime cannot initialize"
    echo ""
    echo "   2. OR try recreating the pod:"
    echo "      - Stop current pod"
    echo "      - Create new pod with same template"
    echo "      - Ensure GPU is enabled in template"
    echo ""
    echo "   You're paying \$19/hour - this needs RunPod support to fix!"
else
    echo "   ✓ Driver works - testing PyTorch..."
    python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import torch
if torch.cuda.is_available():
    print(f"   ✓ PyTorch CUDA works! {torch.cuda.device_count()} GPUs")
else:
    print("   ✗ PyTorch still can't see CUDA (PyTorch-specific issue)")
PYEOF
fi
