#!/bin/bash

# Single A100 GPU Training Script for NanoChat
# Designed to run on a single A100 GPU (40GB or 80GB)
# This is adapted from speedrun.sh but optimized for single GPU

# Configuration options:
# - For 40GB A100: Use CONFIG="conservative" or CONFIG="balanced"
# - For 80GB A100: Use CONFIG="aggressive" or CONFIG="maximum"
CONFIG=${CONFIG:-"balanced"}  # conservative|balanced|aggressive|maximum

# 1) Example launch (simplest):
# bash run_single_a100.sh
# 2) Example launch with specific config:
# CONFIG=conservative bash run_single_a100.sh
# 3) Example launch in a screen session:
# screen -L -Logfile single_a100.log -S single_a100 bash run_single_a100.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not available in PATH. Please ensure ~/.local/bin is in your PATH."
    exit 1
fi
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
# Configuration selection based on CONFIG variable
case $CONFIG in
    conservative)
        # For 40GB A100 - smaller model, smaller batches
        DEPTH=12
        MAX_SEQ_LEN=1024
        DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=131072
        DATA_SHARDS=80
        echo "Using CONSERVATIVE config (40GB A100): depth=$DEPTH, batch=$DEVICE_BATCH_SIZE, seq_len=$MAX_SEQ_LEN"
        ;;
    balanced)
        # For 40GB A100 - balanced approach
        DEPTH=14
        MAX_SEQ_LEN=1536
        DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=262144
        DATA_SHARDS=120
        echo "Using BALANCED config (40GB A100): depth=$DEPTH, batch=$DEVICE_BATCH_SIZE, seq_len=$MAX_SEQ_LEN"
        ;;
    aggressive)
        # For 80GB A100 - larger model
        DEPTH=16
        MAX_SEQ_LEN=2048
        DEVICE_BATCH_SIZE=8
        TOTAL_BATCH_SIZE=524288
        DATA_SHARDS=160
        echo "Using AGGRESSIVE config (80GB A100): depth=$DEPTH, batch=$DEVICE_BATCH_SIZE, seq_len=$MAX_SEQ_LEN"
        ;;
    maximum)
        # For 80GB A100 - maximum size
        DEPTH=18
        MAX_SEQ_LEN=2048
        DEVICE_BATCH_SIZE=8
        TOTAL_BATCH_SIZE=524288
        DATA_SHARDS=180
        echo "Using MAXIMUM config (80GB A100): depth=$DEPTH, batch=$DEVICE_BATCH_SIZE, seq_len=$MAX_SEQ_LEN"
        ;;
    *)
        echo "Unknown config: $CONFIG. Using balanced as default."
        CONFIG=balanced
        DEPTH=14
        MAX_SEQ_LEN=1536
        DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=262144
        DATA_SHARDS=120
        ;;
esac

# Calculate expected training tokens based on model size
# Chinchilla scaling: tokens = 20 × parameters
# Approximate params: depth × 64 × depth × 64 × 12 (rough estimate)
APPROX_PARAMS=$((DEPTH * 64 * DEPTH * 64 * 12 / 1000000))  # in millions
EXPECTED_TOKENS=$((APPROX_PARAMS * 20))  # in millions
echo "Approximate model size: ~${APPROX_PARAMS}M parameters"
echo "Expected training tokens: ~${EXPECTED_TOKENS}M tokens"

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset for tokenizer training
python -m nanochat.dataset -n 8
# Download the rest of the shards needed for pretraining in the background
python -m nanochat.dataset -n $DATA_SHARDS &
DATASET_DOWNLOAD_PID=$!

# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Calculate gradient accumulation steps
TOKENS_PER_FWDBWD=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / TOKENS_PER_FWDBWD))
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"

# NOTE: We run WITHOUT torchrun - single GPU training
# pretrain the model
python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --run=$WANDB_RUN

# evaluate the model on a larger chunk of train/val data and draw some samples
python -m scripts.base_loss --device_batch_size=$DEVICE_BATCH_SIZE
# evaluate the model on CORE tasks
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
# NOTE: Use same device_batch_size as base training
python -m scripts.mid_train \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --max_seq_len=$MAX_SEQ_LEN \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --run=$WANDB_RUN
python -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
# SFT uses smaller batch size by default, but we'll use a reasonable value
SFT_BATCH_SIZE=$((DEVICE_BATCH_SIZE / 2))
if [ $SFT_BATCH_SIZE -lt 2 ]; then
    SFT_BATCH_SIZE=2
fi
python -m scripts.chat_sft \
    --device_batch_size=$SFT_BATCH_SIZE \
    --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "Training complete!"
echo "Model: d${DEPTH} (~${APPROX_PARAMS}M parameters)"
echo "Config: $CONFIG"
echo "To chat with your model, run:"
echo "  python -m scripts.chat_web"
echo "=========================================="
