# Fix PyTorch CUDA Issue in Colab

## The Problem

`uv sync --extra gpu` installs PyTorch with CUDA 12.8 (from `pyproject.toml`), which requires `libnvshmem_host.so.3` that's not available on L4 GPU systems.

## ✅ Solution: Install PyTorch AFTER uv sync

The updated script now fixes PyTorch **after** `uv sync` runs. But if you want to do it manually:

### Step 1: Clone and Setup

```python
# Cell 1: Clone repo
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat

# Install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!source .venv/bin/activate
```

### Step 2: Install Dependencies (will install wrong PyTorch)

```python
# Cell 2: Install dependencies (will install CUDA 12.8 PyTorch)
!uv sync --extra gpu
```

### Step 3: Fix PyTorch IMMEDIATELY After

```python
# Cell 3: Fix PyTorch (CRITICAL - do this right after uv sync)
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

### Step 4: Now Run Training

```python
# Cell 4: Run training (PyTorch is now fixed)
CONFIG=demo bash run_single_a100_REALISTIC.sh
```

---

## Alternative: Modify pyproject.toml

You can also modify `pyproject.toml` to use CUDA 11.8 instead of 12.8:

```toml
# Change this line in pyproject.toml:
# From:
{ index = "pytorch-cu128", extra = "gpu" }
# To:
{ index = "pytorch-cu118", extra = "gpu" }

# And change the index URL:
# From:
url = "https://download.pytorch.org/whl/cu128"
# To:
url = "https://download.pytorch.org/whl/cu118"
```

But the script fix is easier - it handles it automatically!

---

## Why This Happens

1. `pyproject.toml` specifies CUDA 12.8 for PyTorch
2. `uv sync` installs exactly what's specified
3. CUDA 12.8 requires `nvshmem` library
4. L4 GPU systems don't have this library
5. Solution: Override PyTorch installation after `uv sync`

---

## Updated Script Behavior

The updated `run_single_a100_REALISTIC.sh` now:
1. ✅ Runs `uv sync` (installs wrong PyTorch)
2. ✅ Immediately fixes PyTorch to CUDA 11.8
3. ✅ Verifies it works
4. ✅ Falls back to CUDA 12.1 if 11.8 doesn't work

So you can just run the script and it will handle everything!

---

## Quick Test

After fixing PyTorch, test it:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Should work without libnvshmem error!
```

If you see the GPU name, you're good to go! 🚀
