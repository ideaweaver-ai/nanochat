# Fixing CUDA Error on L4 GPU in Colab

## The Error

```
ImportError: libnvshmem_host.so.3: cannot open shared object file: No such file or directory
```

This happens because PyTorch was installed with CUDA support that requires `nvshmem` (NVIDIA Shared Memory), but it's not available on your system.

## Quick Fix

### Option 1: Reinstall PyTorch (Recommended)

Run this in a Colab cell **before** running the training script:

```python
# Uninstall current PyTorch
!pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 11.8 (most compatible)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify it works
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Option 2: Use CPU-Only PyTorch (If GPU doesn't work)

```python
# Install CPU-only PyTorch
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Note**: This will be VERY slow, but it will work.

### Option 3: Install Missing Library

```bash
# Try installing nvshmem (may not work in Colab)
!apt-get update
!apt-get install -y libnvshmem2
```

---

## Updated Training Script for L4 GPU

The L4 GPU has different memory constraints. Here's an updated config:

### L4 GPU Config (16GB VRAM)

```python
# In Colab, before running training:
CONFIG = 'l4'  # New config for L4 GPU

# L4-specific settings:
DEPTH = 6      # Smaller model
MAX_SEQ_LEN = 512
DEVICE_BATCH_SIZE = 2  # Very small batch
TOTAL_BATCH_SIZE = 4096
BASE_ITERATIONS = 500
```

---

## Complete Fix Workflow

### Step 1: Fix PyTorch Installation

```python
# Cell 1: Fix PyTorch
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 2: Clone and Setup

```python
# Cell 2: Clone repo
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat

# Install dependencies (will use the fixed PyTorch)
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!uv sync --extra gpu
```

### Step 3: Run Training with L4 Config

```python
# Cell 3: Run with smaller config for L4
import os
os.environ['CONFIG'] = 'demo'
os.environ['NANOCHAT_BASE_DIR'] = '/content/.cache/nanochat'

# Override for L4 GPU (smaller batch size)
!python -m scripts.base_train \
    --depth=6 \
    --max_seq_len=512 \
    --device_batch_size=2 \
    --total_batch_size=4096 \
    --num_iterations=500 \
    --target_param_data_ratio=-1
```

---

## L4 GPU Specific Considerations

### Memory Limits
- **L4 GPU**: 16GB VRAM (vs 40GB for A100)
- **Solution**: Much smaller batches and model

### Recommended Settings for L4

```python
DEPTH = 6              # Very small model (~50M params)
MAX_SEQ_LEN = 512      # Shorter sequences
DEVICE_BATCH_SIZE = 2  # Tiny batches
TOTAL_BATCH_SIZE = 4096  # Small total batch
```

### If You Still Get OOM

1. Reduce `device_batch_size` to 1
2. Reduce `max_seq_len` to 256
3. Reduce `depth` to 4
4. Use gradient checkpointing (if available)

---

## Alternative: Use CPU (Very Slow)

If GPU doesn't work at all:

```python
# Install CPU-only PyTorch
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run with CPU
!python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=256 \
    --device_batch_size=1 \
    --num_iterations=100  # Very limited
```

**Warning**: This will be EXTREMELY slow (hours for just 100 iterations).

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `libnvshmem_host.so.3` | Reinstall PyTorch with CUDA 11.8 |
| `Out of memory` | Reduce batch size to 1, reduce depth to 4 |
| `CUDA not available` | Check GPU runtime is enabled |
| `Module not found` | Run `uv sync --extra gpu` again |

---

## Summary

1. ✅ **Fix PyTorch first**: Reinstall with CUDA 11.8
2. ✅ **Use smaller config**: L4 has less memory
3. ✅ **Start with demo**: Test with minimal iterations
4. ✅ **Monitor memory**: Use `!nvidia-smi` to check usage

The key is reinstalling PyTorch with the correct CUDA version before running anything else!
