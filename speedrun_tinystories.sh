#!/bin/bash

# TinyStories-style training script for nanochat
# This trains a small model (5M params) with restricted vocabulary (2K tokens)
# on simplified text data, similar to the TinyStories approach

# Usage:
#   bash speedrun_tinystories.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not available in PATH. Please ensure ~/.local/bin is in your PATH."
    exit 1
fi
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=tinystories
fi

# -----------------------------------------------------------------------------
# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# IMPORTANT: Prepare your simplified dataset first!
# 
# You need to create a simplified dataset with:
# - Limited vocabulary (~1,500-2,000 words)
# - Simple stories/texts (50-200 words each)
# - Domain-specific (children's stories, simple conversations, etc.)
#
# Options:
# 1. Use TinyStories dataset if available
# 2. Generate synthetic data with GPT-4/Claude
# 3. Filter existing dataset to simple texts only
#
# Place your dataset in: $NANOCHAT_BASE_DIR/tinystories_data/
# Format: One text per line, plain text files

echo "=========================================="
echo "TinyStories-Style Training"
echo "=========================================="
echo ""
echo "IMPORTANT: You need to prepare a simplified dataset first!"
echo "Expected location: $NANOCHAT_BASE_DIR/tinystories_data/"
echo ""
echo "The dataset should contain:"
echo "  - Simple texts (50-200 words each)"
echo "  - Limited vocabulary (~1,500-2,000 words)"
echo "  - One text per line"
echo ""
read -p "Press Enter when your dataset is ready, or Ctrl+C to exit..."

# Check if dataset exists
TINYSTORIES_DATA_DIR="$NANOCHAT_BASE_DIR/tinystories_data"
if [ ! -d "$TINYSTORIES_DATA_DIR" ] || [ -z "$(ls -A $TINYSTORIES_DATA_DIR 2>/dev/null)" ]; then
    echo "ERROR: Dataset not found at $TINYSTORIES_DATA_DIR"
    echo ""
    echo "Please create a simplified dataset first. Options:"
    echo "  1. Download TinyStories dataset"
    echo "  2. Generate synthetic simple stories"
    echo "  3. Filter existing dataset to simple texts"
    echo ""
    echo "See TINYSTORIES_APPROACH_GUIDE.md for details."
    exit 1
fi

# Train tokenizer with SMALL vocabulary (2,000 tokens instead of 65,536)
echo "Training tokenizer with vocabulary size 2,000..."
echo "(This will take ~10-15 minutes)"
python -m scripts.tok_train \
    --vocab_size=2000 \
    --max_chars=500000000 \
    --doc_cap=5000

# Evaluate tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# Train small model (depth=3 gives ~6M parameters)
# This is much smaller than the default depth=20 (561M params)
echo ""
echo "Training base model (depth=3, ~6M parameters)..."
echo "(This will take ~30-45 minutes on 8×H100)"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.base_train \
    --depth=3 \
    --run=$WANDB_RUN \
    --device_batch_size=32 \
    --total_batch_size=131072  # Smaller batch for smaller model

# Evaluate base model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (optional - teaches conversation format)

echo ""
echo "Midtraining (teaching conversation format)..."
echo "(This will take ~10-15 minutes)"

# Download identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run midtraining
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.mid_train \
    --run=$WANDB_RUN \
    --device_batch_size=32

# Evaluate
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_eval \
    -- -i mid

# -----------------------------------------------------------------------------
# Supervised Fine-Tuning (optional)

echo ""
echo "Supervised Fine-Tuning..."
echo "(This will take ~5-10 minutes)"

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_sft \
    --run=$WANDB_RUN \
    --device_batch_size=4

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_eval \
    -- -i sft

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model: TinyStories-style (depth=3, ~6M params, 2K vocab)"
echo "Location: $NANOCHAT_BASE_DIR/chatsft_checkpoints/d3/"
echo ""
echo "To test your model:"
echo "  python -m scripts.chat_cli -p 'Once upon a time, there was a'"
echo ""
echo "Note: This model will ONLY work well for simple stories/texts"
echo "      in the domain you trained on. It cannot handle:"
echo "      - Complex vocabulary"
echo "      - General knowledge questions"
echo "      - Code generation"
echo "      - Other domains"
echo ""
echo "See TINYSTORIES_APPROACH_GUIDE.md for details on limitations."

