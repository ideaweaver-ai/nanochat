#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Fix CUDA library path issues (common in containers)
# This ensures PyTorch can find CUDA runtime libraries
if [ -d "/usr/local/nvidia/lib64" ] || [ -d "/usr/local/nvidia/lib" ]; then
    export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH if it was just installed
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not available in PATH. Please ensure ~/.local/bin is in your PATH."
    exit 1
fi

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
# First, regenerate lock file if transformers was added to pyproject.toml
if grep -q "transformers" pyproject.toml; then
    echo "Regenerating lock file to include transformers..."
    uv lock || echo "Warning: uv lock failed, continuing..."
fi

uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Verify PyTorch CUDA support after installation
# CRITICAL: Set LD_LIBRARY_PATH before importing torch
echo "Verifying PyTorch CUDA installation..."
python << 'PYEOF'
import os
# Set library path BEFORE importing torch
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"}')
PYEOF

# Verify transformers is installed in the venv
echo "Verifying dependencies..."
.venv/bin/python -c "import transformers; print(f'✓ transformers {transformers.__version__}')" || {
    echo "WARNING: transformers not found in venv. Installing directly..."
    # Use uv pip to install directly (bypasses lock file)
    uv pip install transformers || {
        echo "ERROR: Failed to install transformers. Exiting."
        exit 1
    }
    # Verify it's installed
    .venv/bin/python -c "import transformers; print(f'✓ transformers {transformers.__version__} installed')" || {
        echo "ERROR: transformers still not found after installation. Exiting."
        exit 1
    }
}

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# NOTE: Using Qwen3 tokenizer instead of custom Rust BPE tokenizer
# Rust/Cargo and rustbpe build are no longer needed
# If you want to use the original Rust BPE tokenizer, uncomment the lines below:
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source "$HOME/.cargo/env"
# uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background
# See comment below for why 160 is the right number here (reduced from 240 for d16 model)
python -m nanochat.dataset -n 160 &
DATASET_DOWNLOAD_PID=$!

# ============================================================================
# TOKENIZER: Using Qwen3 tokenizer instead of training custom tokenizer
# ============================================================================
# Setup Qwen tokenizer (downloads from HuggingFace, adds special tokens, computes token_bytes.pt)
# This replaces the custom Rust BPE tokenizer training which saves ~30-60 minutes
python -m scripts.setup_qwen_tokenizer
# Note: tok_eval is skipped since we're using a pre-trained tokenizer

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d16 model with GQA is ~380M parameters (reduced from d20's 561M for efficiency).
# Chinchilla says #tokens = 20X #params, so we need 380e6 * 20 = 7.6B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 7.6B * 4.8 ~= 36.5B chars.
# At 250M chars/shard, this is 36.5B / 250M ~= 146 shards needed for pretraining.
# Round up to 160 for safety. At ~100MB/shard, this downloads ~16GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Auto-detect number of GPUs and verify PyTorch can access them
echo "Checking GPU availability..."

# Check for CUDA_VISIBLE_DEVICES - if set incorrectly, it can cause issues
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚠ CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
    echo "  If you're seeing CUDA errors, try unsetting it: unset CUDA_VISIBLE_DEVICES"
fi

# First check if nvidia-smi can see GPUs
NVIDIA_GPU_COUNT=0
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$NVIDIA_GPU_COUNT" -gt 0 ]; then
        echo "✓ nvidia-smi detected $NVIDIA_GPU_COUNT GPU(s)"
        # Show CUDA version from nvidia-smi
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo "  Driver version: $CUDA_VERSION"
    else
        echo "⚠ nvidia-smi found but no GPUs detected"
    fi
else
    echo "⚠ nvidia-smi not found"
fi

# Now check if PyTorch can access CUDA
echo "Checking PyTorch CUDA support..."
PYTORCH_CUDA_AVAILABLE=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "error")
PYTORCH_CUDA_COUNT=0

if [ "$PYTORCH_CUDA_AVAILABLE" = "yes" ]; then
    PYTORCH_CUDA_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    PYTORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo "✓ PyTorch CUDA is available"
    echo "  PyTorch can see $PYTORCH_CUDA_COUNT GPU(s)"
    echo "  PyTorch CUDA version: $PYTORCH_CUDA_VERSION"
    
    # Use the minimum of what nvidia-smi sees and what PyTorch sees
    if [ "$NVIDIA_GPU_COUNT" -gt 0 ] && [ "$PYTORCH_CUDA_COUNT" -gt 0 ]; then
        NPROC_PER_NODE=$((NVIDIA_GPU_COUNT < PYTORCH_CUDA_COUNT ? NVIDIA_GPU_COUNT : PYTORCH_CUDA_COUNT))
    else
        NPROC_PER_NODE=$PYTORCH_CUDA_COUNT
    fi
elif [ "$PYTORCH_CUDA_AVAILABLE" = "error" ]; then
    echo "✗ Error checking PyTorch CUDA availability"
    echo "  This might indicate PyTorch is not installed correctly"
    NPROC_PER_NODE=1
else
    echo "✗ PyTorch CUDA is NOT available"
    if [ "$NVIDIA_GPU_COUNT" -gt 0 ]; then
        echo "  ⚠ CRITICAL: nvidia-smi sees $NVIDIA_GPU_COUNT GPU(s) but PyTorch cannot access them"
        echo ""
        echo "  Root cause: CUDA driver cannot initialize (cuInit fails with error 999)"
        echo "  This indicates a container GPU access configuration issue."
        echo ""
        echo "  Solution: Restart container with proper GPU runtime:"
        echo "    docker run --gpus all --runtime=nvidia \\"
        echo "      -e NVIDIA_VISIBLE_DEVICES=all \\"
        echo "      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\"
        echo "      ..."
        echo ""
        echo "  For now, falling back to CPU mode (very slow)."
        echo "  See CUDA_FIX_SUMMARY.md for details."
        echo ""
        echo "  Diagnostic info:"
        python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA compiled version: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"N/A\"}')" 2>/dev/null || echo "  Could not get PyTorch version"
        echo ""
        echo "  Attempting to fix CUDA version mismatch (may not work if container issue)..."
        # Detect system CUDA version and install compatible PyTorch
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' | head -1)
            echo "  System CUDA version: $CUDA_VERSION"
            # CUDA 12.4 is compatible with PyTorch cu121
            if [[ "$CUDA_VERSION" == "12."* ]]; then
                PYTORCH_CUDA="cu121"
            else
                PYTORCH_CUDA="cu121"  # Default fallback
            fi
        else
            PYTORCH_CUDA="cu121"  # Default for H100
        fi
        echo "  Installing PyTorch for CUDA $PYTORCH_CUDA..."
        uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
        uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA" 2>/dev/null || {
            echo "  Failed to reinstall. Falling back to CPU mode."
        }
        # Re-check after reinstall
        PYTORCH_CUDA_AVAILABLE=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
        if [ "$PYTORCH_CUDA_AVAILABLE" = "yes" ]; then
            PYTORCH_CUDA_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
            NPROC_PER_NODE=$PYTORCH_CUDA_COUNT
            echo "  ✓ PyTorch CUDA is now available with $PYTORCH_CUDA_COUNT GPU(s)"
        else
            NPROC_PER_NODE=1
            echo "  ✗ Still cannot access CUDA. Will use CPU mode (single process)"
        fi
    else
        NPROC_PER_NODE=1
        echo "  No GPUs detected. Using CPU mode (single process)"
    fi
fi

echo ""
echo "Using $NPROC_PER_NODE process(es) for training"

# pretrain the d16 model (reduced from d20 for parameter efficiency: ~561M params vs ~911M)
# GQA is enabled in base_train.py (num_kv_heads = num_heads // 2) for additional parameter reduction
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=16 --run=$WANDB_RUN
else
    "$(pwd)/.venv/bin/python" -m scripts.base_train --depth=16 --run=$WANDB_RUN
fi

# evaluate the model on a larger chunk of train/val data and draw some samples
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
else
    "$(pwd)/.venv/bin/python" -m scripts.base_loss
fi

# evaluate the model on CORE tasks
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
else
    "$(pwd)/.venv/bin/python" -m scripts.base_eval
fi

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
else
    "$(pwd)/.venv/bin/python" -m scripts.mid_train --run=$WANDB_RUN
fi

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
else
    "$(pwd)/.venv/bin/python" -m scripts.chat_eval -i mid
fi

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
else
    "$(pwd)/.venv/bin/python" -m scripts.chat_sft --run=$WANDB_RUN
fi

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
else
    "$(pwd)/.venv/bin/python" -m scripts.chat_eval -i sft
fi

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
