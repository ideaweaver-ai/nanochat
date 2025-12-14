# Quick Fix for L4 GPU CUDA Error

## The Problem

PyTorch was installed with CUDA 12.8, but your L4 GPU system doesn't have `libnvshmem_host.so.3`.

## ✅ Immediate Fix (Run This First!)

```python
# In Colab, run this BEFORE anything else:

# 1. Uninstall broken PyTorch
!pip uninstall -y torch torchvision torchaudio

# 2. Install PyTorch with CUDA 11.8 (compatible with L4)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify it works
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## Then Continue with Setup

```python
# Now clone and setup
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat

# Install other dependencies (will skip PyTorch since it's already installed)
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!source .venv/bin/activate
!uv sync --extra gpu --no-install-project  # Skip PyTorch install
```

## L4 GPU Config (16GB VRAM)

Use these smaller settings for L4:

```python
# L4-specific config
DEPTH = 6              # Small model
MAX_SEQ_LEN = 512
DEVICE_BATCH_SIZE = 2  # Small batch
TOTAL_BATCH_SIZE = 4096
BASE_ITERATIONS = 500
```

## Complete Workflow

```python
# Cell 1: Fix PyTorch
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
print(f"CUDA: {torch.cuda.is_available()}")

# Cell 2: Clone and setup
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!source .venv/bin/activate
!uv sync --extra gpu

# Cell 3: Run training with L4 config
!python -m scripts.base_train \
    --depth=6 \
    --max_seq_len=512 \
    --device_batch_size=2 \
    --total_batch_size=4096 \
    --num_iterations=500 \
    --target_param_data_ratio=-1
```

That's it! The key is fixing PyTorch **before** running `uv sync`.
