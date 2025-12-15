#!/bin/bash
# Check if training is running and using GPUs

echo "=== Training Status Check ==="
echo ""

# Check for running Python processes
echo "1. Python processes:"
ps aux | grep -E "python.*base_train|torchrun.*base_train" | grep -v grep || echo "  No training processes found"

echo ""
echo "2. GPU processes (from nvidia-smi):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "  No GPU processes found"

echo ""
echo "3. GPU utilization:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "4. PyTorch CUDA status:"
cd /workspace/nanochat 2>/dev/null || cd $(dirname $0)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
    python << 'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"  CUDA available: Yes")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(f"  GPU {i} ({props.name}):")
        print(f"    Allocated: {allocated / 1e9:.2f} GB")
        print(f"    Reserved: {reserved / 1e9:.2f} GB")
        print(f"    Total: {props.total_memory / 1e9:.2f} GB")
else:
    print("  CUDA available: No")
PYEOF
else
    echo "  Venv not found"
fi
