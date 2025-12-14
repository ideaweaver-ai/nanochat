# Manual Fix for Colab - Step by Step

Since the script keeps getting overwritten, here's a **manual step-by-step** that you can run in Colab cells:

## Complete Manual Setup

### Cell 1: Clone and Initial Setup

```python
# Clone repo
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat

# Install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!source .venv/bin/activate
```

### Cell 2: Install Dependencies (Will Install Wrong PyTorch)

```python
# This will install PyTorch with CUDA 12.8 (wrong version)
!uv sync --extra gpu
```

### Cell 3: Fix PyTorch IMMEDIATELY

```python
# CRITICAL: Fix PyTorch right after uv sync
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 4: Install Rust

```python
# Install Rust
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
import os
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ.get('PATH', '')
```

### Cell 5: Build Tokenizer (Will Reinstall PyTorch)

```python
# This will reinstall PyTorch with wrong version
!uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Cell 6: Fix PyTorch AGAIN (After Maturin)

```python
# CRITICAL: Fix PyTorch again after maturin
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify again
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
```

### Cell 7: Now Run Training

```python
# Set environment variables
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NANOCHAT_BASE_DIR'] = '/content/.cache/nanochat'
os.environ['WANDB_RUN'] = 'dummy'

# Initialize report
!python -m nanochat.report reset

# Download data
!python -m nanochat.dataset -n 4
!python -m nanochat.dataset -n 10 &

# Train tokenizer
!python -m scripts.tok_train --max_chars=500000000
!python -m scripts.tok_eval
```

### Cell 8: Base Training

```python
# Wait for data download
import time
time.sleep(30)  # Wait a bit for download

# Base training
!python -m scripts.base_train \
    --depth=8 \
    --max_seq_len=512 \
    --device_batch_size=4 \
    --total_batch_size=8192 \
    --num_iterations=500 \
    --target_param_data_ratio=-1 \
    --eval_every=100 \
    --eval_tokens=8192 \
    --core_metric_every=-1 \
    --sample_every=200 \
    --run=dummy
```

---

## Why This Happens

1. `uv sync` installs PyTorch with CUDA 12.8
2. `maturin develop` also installs dependencies (including PyTorch)
3. Both operations overwrite your fixed PyTorch
4. **Solution**: Fix PyTorch after EACH operation

---

## Quick Fix Function

You can create a helper function:

```python
def fix_pytorch():
    """Fix PyTorch to CUDA 11.8"""
    import subprocess
    subprocess.run(['pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'], 
                   capture_output=True)
    subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', 
                    '--index-url', 'https://download.pytorch.org/whl/cu118'])
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")

# Use it after each operation:
!uv sync --extra gpu
fix_pytorch()

!uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fix_pytorch()
```

---

## Alternative: Modify pyproject.toml

You can also modify `pyproject.toml` to use CUDA 11.8 from the start:

```python
# In Colab, after cloning:
!sed -i 's/pytorch-cu128/pytorch-cu118/g' pyproject.toml
!sed -i 's/cu128/cu118/g' pyproject.toml

# Then run uv sync
!uv sync --extra gpu
```

This way, `uv sync` will install the correct PyTorch from the start!

---

## Recommended: Use the Manual Steps

The script keeps getting overwritten, so **use the manual steps above** - they're more reliable and you have full control.
