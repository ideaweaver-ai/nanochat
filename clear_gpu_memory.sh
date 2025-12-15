#!/bin/bash
# Clear GPU memory by killing any Python processes using CUDA

echo "=== Clearing GPU Memory ==="
echo ""

# Find and kill any Python processes using CUDA
echo "Checking for Python processes using GPU..."
PYTHON_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)

if [ -n "$PYTHON_PIDS" ]; then
    echo "Found processes using GPU: $PYTHON_PIDS"
    for pid in $PYTHON_PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            CMD=$(ps -p $pid -o cmd= 2>/dev/null | head -1)
            if echo "$CMD" | grep -q python; then
                echo "  Killing Python process $pid: $CMD"
                kill -9 $pid 2>/dev/null || true
            fi
        fi
    done
    sleep 2
else
    echo "No Python processes found using GPU"
fi

# Clear PyTorch cache
echo ""
echo "Clearing PyTorch CUDA cache..."
python << 'PYEOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        free = props.total_memory - allocated
        print(f"  GPU {i}: {free / 1e9:.2f} GB free / {props.total_memory / 1e9:.2f} GB total")
PYEOF

echo ""
echo "✓ GPU memory cleared"
