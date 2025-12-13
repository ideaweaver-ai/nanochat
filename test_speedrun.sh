#!/bin/bash

# TEST VERSION of speedrun.sh - Quick test to verify everything works
# This runs a minimal version that should complete in ~10-30 minutes
# Use this to verify your setup before running the full 4-hour training

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Setup Qwen tokenizer (downloads from HuggingFace, adds special tokens, computes token_bytes.pt)
echo "Setting up Qwen tokenizer..."
python -m scripts.setup_qwen_tokenizer

# -----------------------------------------------------------------------------
# Dataset Download (MINIMAL - only 2 shards for testing)
# Each shard is ~100MB, so this is only ~200MB of data
echo "Downloading minimal dataset for testing (2 shards, ~200MB)..."
python -m nanochat.dataset -n 2

# -----------------------------------------------------------------------------
# Base model (pretraining) - TEST VERSION
# Using very small settings to complete quickly

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NPROC_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$NPROC_PER_NODE" -eq 0 ] || [ -z "$NPROC_PER_NODE" ]; then
        echo "Warning: Could not detect GPUs, defaulting to 1"
        NPROC_PER_NODE=1
    fi
else
    echo "Warning: nvidia-smi not found, defaulting to 1 GPU"
    NPROC_PER_NODE=1
fi

echo "Detected $NPROC_PER_NODE GPU(s), using that for training"

# Verify GPUs are accessible
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; assert torch.cuda.device_count() >= $NPROC_PER_NODE, f'Only {torch.cuda.device_count()} GPU(s) available, but trying to use $NPROC_PER_NODE'" || {
        echo "Error: GPU count mismatch. Available GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'unknown')"
        echo "Falling back to single GPU mode"
        NPROC_PER_NODE=1
    }
fi

echo "Starting TEST pretraining (d16 model, 50 iterations only)..."
echo "This should take ~5-10 minutes instead of 2-3 hours"

# TEST: Only 50 iterations, smaller eval settings
# Note: --run must come after -- to avoid conflict with torchrun's --run-path
# Use torchrun only if multiple GPUs, otherwise run directly
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=16 \
    --run=$WANDB_RUN \
    --num_iterations=50 \
    --eval_every=25 \
    --eval_tokens=524288 \
    --core_metric_every=-1 \
    --sample_every=50
else
    python -m scripts.base_train -- \
    --depth=16 \
    --run=$WANDB_RUN \
    --num_iterations=50 \
    --eval_every=25 \
    --eval_tokens=524288 \
    --core_metric_every=-1 \
    --sample_every=50
fi

echo "TEST pretraining complete!"

# Quick evaluation (minimal)
# Note: base_loss.py uses split_tokens, not eval_tokens
echo "Running quick evaluation..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss -- \
    --split_tokens=524288

# Skip base_eval (CORE metric) for speed - it's expensive
echo "Skipping CORE evaluation for speed (test mode)"

# -----------------------------------------------------------------------------
# Midtraining - TEST VERSION

# download identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

echo "Starting TEST midtraining (50 iterations only)..."
echo "This should take ~2-5 minutes instead of 30-45 minutes"

# TEST: Only 50 iterations
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- \
        --run=$WANDB_RUN \
        --num_iterations=50 \
        --eval_every=25 \
        --eval_tokens=524288
else
    python -m scripts.mid_train -- \
        --run=$WANDB_RUN \
        --num_iterations=50 \
        --eval_every=25 \
        --eval_tokens=524288
fi

# Quick eval (minimal problems)
echo "Running quick chat evaluation..."
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
        --source=mid \
        --max-problems=10
else
    python -m scripts.chat_eval -- \
        --source=mid \
        --max-problems=10
fi

# -----------------------------------------------------------------------------
# Supervised Finetuning - TEST VERSION

echo "Starting TEST SFT (50 iterations only)..."
echo "This should take ~2-5 minutes instead of 30-45 minutes"

# TEST: Only 50 iterations
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
        --run=$WANDB_RUN \
        --num_iterations=50 \
        --eval_steps=25 \
        --eval_metrics_max_problems=10
else
    python -m scripts.chat_sft -- \
        --run=$WANDB_RUN \
        --num_iterations=50 \
        --eval_steps=25 \
        --eval_metrics_max_problems=10
fi

# Quick eval
echo "Running quick SFT evaluation..."
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
        --source=sft \
        --max-problems=10
else
    python -m scripts.chat_eval -- \
        --source=sft \
        --max-problems=10
fi

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "TEST RUN COMPLETE!"
echo "=========================================="
echo ""
echo "If this worked, you can now run the full training with:"
echo "  bash speedrun.sh"
echo ""
echo "Or in a screen session:"
echo "  screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh"
echo ""
echo "To test the web UI:"
echo "  source .venv/bin/activate"
echo "  python -m scripts.chat_web"
echo ""
