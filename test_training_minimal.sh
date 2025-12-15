#!/bin/bash
# Minimal test - run just a few training steps to verify everything works

set -e

echo "=== Minimal Training Test ==="
echo "This will run 5 training steps to verify everything works"
echo ""

cd /workspace/nanochat
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

# Run minimal training - just 5 steps
python -m scripts.base_train \
    --depth=16 \
    --max_seq_len=512 \
    --device_batch_size=4 \
    --total_batch_size=1024 \
    --num_iterations=5 \
    --eval_every=10 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    --save_every=-1

echo ""
echo "=== Test completed successfully! ==="
