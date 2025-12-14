# DEFINITIVE Fix for Colab L4 GPU

## The Root Problem

Even CUDA 11.8 PyTorch is trying to load `libnvshmem_host.so.3` because the system has CUDA 12.8 libraries that PyTorch is detecting and trying to use.

## ✅ SOLUTION: Fix pyproject.toml FIRST

**This is the ONLY reliable way** - modify `pyproject.toml` before running `uv sync`:

### Step 1: Fix pyproject.toml

```python
# In Colab, after cloning:
!cd nanochat
!sed -i 's/pytorch-cu128/pytorch-cu118/g' pyproject.toml
!sed -i 's/cu128/cu118/g' pyproject.toml

# Verify the change
!grep -n "cu118\|cu128" pyproject.toml
```

### Step 2: Now Setup

```python
# Install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!source .venv/bin/activate

# Now uv sync will install CUDA 11.8 from the start
!uv sync --extra gpu
```

### Step 3: Verify PyTorch

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
# Should work without libnvshmem error!
```

---

## Alternative: Install Missing Library

If fixing pyproject.toml doesn't work, try installing the library:

```bash
# Try to install nvshmem (may not work in Colab)
apt-get update
apt-get install -y libnvshmem2

# Or try to find and install it
find /usr -name "*nvshmem*" 2>/dev/null
```

---

## Complete Workflow (Copy-Paste Ready)

```python
# Cell 1: Clone and Fix pyproject.toml
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat

# CRITICAL: Fix pyproject.toml FIRST
!sed -i 's/pytorch-cu128/pytorch-cu118/g' pyproject.toml
!sed -i 's/cu128/cu118/g' pyproject.toml
!grep "cu118" pyproject.toml  # Verify

# Cell 2: Setup
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!source .venv/bin/activate
!uv sync --extra gpu

# Cell 3: Verify
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Cell 4: Install Rust
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
import os
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ.get('PATH', '')

# Cell 5: Build tokenizer (should use correct PyTorch now)
!uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Cell 6: Verify PyTorch still works
import torch
print(f"✅ PyTorch still works: {torch.__version__}")

# Cell 7: Run training
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NANOCHAT_BASE_DIR'] = '/content/.cache/nanochat'

!python -m nanochat.report reset
!python -m nanochat.dataset -n 4
!python -m scripts.tok_train --max_chars=500000000
```

---

## Why This Works

1. **pyproject.toml** tells `uv` which PyTorch version to install
2. **CUDA 12.8** requires `nvshmem` (not available)
3. **CUDA 11.8** doesn't require `nvshmem` (should work)
4. **By fixing pyproject.toml first**, `uv sync` installs the right version from the start
5. **No more overwrites** - everything uses CUDA 11.8

---

## If CUDA 11.8 Still Doesn't Work

Try an even older version that definitely doesn't need nvshmem:

```python
# After fixing pyproject.toml, manually install older PyTorch
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## Summary

**The ONLY reliable fix:**
1. ✅ Fix `pyproject.toml` FIRST (before `uv sync`)
2. ✅ Then run `uv sync` (will install correct version)
3. ✅ Verify PyTorch works
4. ✅ Continue with training

**Don't try to fix PyTorch after installation** - fix the source (`pyproject.toml`) instead!
