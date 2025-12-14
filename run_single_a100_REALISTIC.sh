#!/bin/bash

# REALISTIC Single A100 GPU Training Script for NanoChat
# This version uses LIMITED iterations and smaller datasets for practical training times
# Designed for learning/demo purposes, not full production training

# ⚠️ REALITY CHECK:
# - Full training would take DAYS/WEEKS on single GPU
# - This script uses LIMITED iterations for reasonable training times
# - Model quality will be lower than full training
# - Use this for learning the pipeline, not production models

# Configuration options:
CONFIG=${CONFIG:-"demo"}  # demo|small|medium

# 1) Example launch:
# bash run_single_a100_REALISTIC.sh
# 2) With specific config:
# CONFIG=small bash run_single_a100_REALISTIC.sh

# -----------------------------------------------------------------------------
# IMPORTANT: Get the script's directory and ensure we're in the right place
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Check if we're in the nanochat directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ ERROR: pyproject.toml not found!"
    echo "   This script must be run from the nanochat directory."
    echo "   Current directory: $(pwd)"
    echo ""
    echo "   If you're in Colab, first run:"
    echo "     git clone https://github.com/ideaweaver-ai/nanochat.git"
    echo "     cd nanochat"
    echo "     bash run_single_a100_REALISTIC.sh"
    exit 1
fi

# CRITICAL: Fix pyproject.toml for Colab/L4 GPU compatibility
# This prevents uv sync from installing CUDA 12.8 PyTorch
if grep -q "pytorch-cu128" pyproject.toml 2>/dev/null; then
    echo "⚠️  Detected CUDA 12.8 in pyproject.toml - fixing for L4 GPU compatibility..."
    if [ ! -f "pyproject.toml.backup" ]; then
        cp pyproject.toml pyproject.toml.backup
        echo "   Backup saved to pyproject.toml.backup"
    fi
    sed -i 's/pytorch-cu128/pytorch-cu118/g' pyproject.toml
    sed -i 's/cu128/cu118/g' pyproject.toml
    echo "✅ Fixed pyproject.toml to use CUDA 11.8"
    echo "   Now uv sync will install the correct PyTorch version from the start"
fi

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup
echo "Setting up Python environment..."

# Check for CUDA/PyTorch issues and fix if needed
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch is already installed, checking CUDA..."
    if ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        echo "⚠️  CUDA not available, but continuing..."
    fi
else
    echo "PyTorch not found, will be installed by uv sync"
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not available in PATH. Please ensure ~/.local/bin is in your PATH."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate venv first
source .venv/bin/activate

# Fix PyTorch CUDA issues if in Colab (detect by checking for libnvshmem error)
if [ -f "/etc/os-release" ] && grep -q "Ubuntu" /etc/os-release; then
    echo "Detected Ubuntu environment, checking for CUDA issues..."
    if python -c "import torch" 2>&1 | grep -q "libnvshmem"; then
        echo "⚠️  Detected libnvshmem error, fixing PyTorch installation..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
fi

# Install dependencies (but skip PyTorch if already installed correctly)
echo "Installing dependencies..."
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "✅ PyTorch already installed and working, skipping PyTorch in uv sync..."
    # Install everything except torch
    uv sync --extra gpu --no-install-project || {
        echo "Installing dependencies without PyTorch..."
        # Manually install other dependencies
        pip install datasets fastapi files-to-prompt psutil regex setuptools tiktoken tokenizers uvicorn wandb
    }
else
    echo "Installing all dependencies (will fix PyTorch after)..."
    uv sync --extra gpu
fi

# CRITICAL: Fix PyTorch AFTER uv sync (it may have installed wrong version)
echo "Fixing PyTorch installation for L4 GPU compatibility..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install compatible PyTorch version
echo "Installing PyTorch with CUDA 11.8 (compatible with L4 GPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch works
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "❌ PyTorch still failing. Trying CUDA 12.1..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print('✅ PyTorch fixed with CUDA 12.1!')" || {
        echo "❌ PyTorch installation failed. Please install manually:"
        echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        exit 1
    }
}

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# REALISTIC Configuration - Limited iterations for practical training times
case $CONFIG in
    demo)
        # DEMO MODE: Very limited, just to see the pipeline work
        DEPTH=8
        MAX_SEQ_LEN=512
        DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=8192
        BASE_ITERATIONS=500        # ~30-60 minutes
        MID_ITERATIONS=200         # ~15-30 minutes
        SFT_ITERATIONS=100         # ~10-20 minutes
        DATA_SHARDS=10             # Minimal data
        echo "DEMO MODE: depth=$DEPTH, ~1 hour total training"
        ;;
    small)
        # SMALL: Limited but more realistic
        DEPTH=10
        MAX_SEQ_LEN=1024
        DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=16384
        BASE_ITERATIONS=2000       # ~2-4 hours
        MID_ITERATIONS=500         # ~1 hour
        SFT_ITERATIONS=200         # ~30 minutes
        DATA_SHARDS=20             # Small subset
        echo "SMALL MODE: depth=$DEPTH, ~3-6 hours total training"
        ;;
    medium)
        # MEDIUM: More iterations but still limited
        DEPTH=12
        MAX_SEQ_LEN=1024
        DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=32768
        BASE_ITERATIONS=5000       # ~6-10 hours
        MID_ITERATIONS=1000        # ~2 hours
        SFT_ITERATIONS=500         # ~1 hour
        DATA_SHARDS=40             # Moderate subset
        echo "MEDIUM MODE: depth=$DEPTH, ~9-13 hours total training"
        ;;
    *)
        echo "Unknown config: $CONFIG. Using demo as default."
        CONFIG=demo
        DEPTH=8
        MAX_SEQ_LEN=512
        DEVICE_BATCH_SIZE=4
        TOTAL_BATCH_SIZE=8192
        BASE_ITERATIONS=500
        MID_ITERATIONS=200
        SFT_ITERATIONS=100
        DATA_SHARDS=10
        ;;
esac

echo ""
echo "=========================================="
echo "REALISTIC TRAINING CONFIGURATION"
echo "=========================================="
echo "Model depth: $DEPTH"
echo "Max seq len: $MAX_SEQ_LEN"
echo "Batch size: $DEVICE_BATCH_SIZE"
echo "Base iterations: $BASE_ITERATIONS"
echo "Mid iterations: $MID_ITERATIONS"
echo "SFT iterations: $SFT_ITERATIONS"
echo ""
echo "⚠️  NOTE: This is LIMITED training for demo purposes"
echo "    Full training would require 10-100x more iterations"
echo "    Model quality will be lower than production"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer (minimal - just enough to work)
echo "Setting up Rust and tokenizer..."

# Install Rust if not available
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env" 2>/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
fi

# Ensure cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Build Rust tokenizer
echo "Building Rust tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# CRITICAL: Fix PyTorch AGAIN after maturin (it may have reinstalled wrong version)
echo "Fixing PyTorch after maturin build..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch works before proceeding
echo "Verifying PyTorch works..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}')" || {
    echo "❌ PyTorch still broken after maturin. Trying CUDA 12.1..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print('✅ PyTorch fixed with CUDA 12.1!')" || {
        echo "❌ CRITICAL: PyTorch installation failed!"
        echo "Please run this manually:"
        echo "  pip uninstall -y torch torchvision torchaudio"
        echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        exit 1
    }
}

# Download minimal data for tokenizer
echo "Downloading data for tokenizer..."
python -m nanochat.dataset -n 4

# Download limited shards for training
echo "Downloading training data ($DATA_SHARDS shards)..."
python -m nanochat.dataset -n $DATA_SHARDS &
DATASET_DOWNLOAD_PID=$!

# Train tokenizer on smaller dataset
echo "Training tokenizer..."
python -m scripts.tok_train --max_chars=500000000  # 500M chars instead of 2B
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Stage 1: Base Training - LIMITED ITERATIONS
echo "Waiting for dataset download..."
wait $DATASET_DOWNLOAD_PID

echo ""
echo "Starting BASE TRAINING (limited to $BASE_ITERATIONS iterations)..."
echo "This will take approximately $((BASE_ITERATIONS / 100))-$(($BASE_ITERATIONS / 50)) minutes"
echo ""

# Use num_iterations to LIMIT training (not Chinchilla scaling)
python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=$BASE_ITERATIONS \
    --target_param_data_ratio=-1 \
    --eval_every=100 \
    --eval_tokens=8192 \
    --core_metric_every=-1 \
    --sample_every=200 \
    --run=$WANDB_RUN

python -m scripts.base_loss --device_batch_size=$DEVICE_BATCH_SIZE
python -m scripts.base_eval --max-per-task=50  # Reduced evaluation

# -----------------------------------------------------------------------------
# Stage 2: Midtraining - LIMITED ITERATIONS
echo ""
echo "Starting MIDTRAINING (limited to $MID_ITERATIONS iterations)..."
echo "This will take approximately $((MID_ITERATIONS / 50))-$(($MID_ITERATIONS / 25)) minutes"
echo ""

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Use num_iterations to LIMIT training
python -m scripts.mid_train \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --max_seq_len=$MAX_SEQ_LEN \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=$MID_ITERATIONS \
    --eval_every=50 \
    --eval_tokens=8192 \
    --run=$WANDB_RUN

python -m scripts.chat_eval -- -i mid --max-problems=50  # Reduced evaluation

# -----------------------------------------------------------------------------
# Stage 3: SFT - LIMITED ITERATIONS
echo ""
echo "Starting SFT (limited to $SFT_ITERATIONS iterations)..."
echo "This will take approximately $((SFT_ITERATIONS / 20))-$(($SFT_ITERATIONS / 10)) minutes"
echo ""

SFT_BATCH_SIZE=$((DEVICE_BATCH_SIZE / 2))
if [ $SFT_BATCH_SIZE -lt 2 ]; then
    SFT_BATCH_SIZE=2
fi

# Use num_iterations to LIMIT training
python -m scripts.chat_sft \
    --device_batch_size=$SFT_BATCH_SIZE \
    --num_iterations=$SFT_ITERATIONS \
    --eval_every=25 \
    --eval_steps=10 \
    --eval_metrics_max_problems=50 \
    --run=$WANDB_RUN

python -m scripts.chat_eval -- -i sft --max-problems=50

# -----------------------------------------------------------------------------
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "LIMITED TRAINING COMPLETE!"
echo "=========================================="
echo "Model: d${DEPTH}"
echo "Config: $CONFIG"
echo ""
echo "⚠️  REMINDER: This was LIMITED training for demo purposes"
echo "    For production, you would need:"
echo "    - 10-100x more iterations"
echo "    - Full dataset (not subset)"
echo "    - Days/weeks of training time"
echo ""
echo "To chat with your model:"
echo "  python -m scripts.chat_web"
echo "=========================================="
