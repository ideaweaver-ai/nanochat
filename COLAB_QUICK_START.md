# Quick Start: NanoChat on Google Colab

## ⚠️ The Problem You Hit

You tried to run the script from `/content/` but it needs to be run from **inside the nanochat directory**.

## ✅ Correct Way to Run in Colab

### Step 1: Clone the Repository

```bash
# In a Colab cell:
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat
```

### Step 2: Run the Script

```bash
# Now run from inside nanochat directory:
CONFIG=demo bash run_single_a100_REALISTIC.sh
```

---

## Complete Colab Setup (Copy-Paste Ready)

### Cell 1: Clone and Setup

```python
# Clone repository
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat

# Check we're in the right place
!pwd
!ls -la pyproject.toml
```

### Cell 2: Run Training

```bash
# Set config and run
CONFIG=demo bash run_single_a100_REALISTIC.sh
```

**OR** use Python variables:

```python
import os
os.environ['CONFIG'] = 'demo'
os.environ['NANOCHAT_BASE_DIR'] = '/content/.cache/nanochat'
os.environ['OMP_NUM_THREADS'] = '1'

!bash run_single_a100_REALISTIC.sh
```

---

## What Went Wrong

1. ❌ **You ran from `/content/`** - script needs to be in `nanochat/` directory
2. ❌ **No `pyproject.toml`** - because you weren't in the repo
3. ❌ **Module not found** - because dependencies weren't installed

## What to Do

1. ✅ **Clone repo first**: `!git clone ... && cd nanochat`
2. ✅ **Then run script**: `bash run_single_a100_REALISTIC.sh`
3. ✅ **Script will auto-detect** if you're in wrong directory

---

## Fixed Script Features

The updated script now:
- ✅ **Checks if you're in the right directory**
- ✅ **Auto-installs uv and Rust** if missing
- ✅ **Better error messages** if something's wrong
- ✅ **Handles PATH issues** automatically

---

## Alternative: Use the Notebook

Instead of running the bash script, use the **Colab notebook**:

1. Upload `NanoChat_Colab_Training.ipynb` to Colab
2. Run all cells
3. Done!

The notebook handles everything automatically.

---

## Quick Troubleshooting

| Error | Solution |
|-------|----------|
| `No pyproject.toml found` | Run `cd nanochat` first |
| `Module not found` | Run `uv sync --extra gpu` |
| `maturin not found` | Script will auto-install Rust |
| `No module named scripts` | Make sure you're in nanochat directory |

---

## Summary

**Before:**
```bash
/content# CONFIG=demo bash run_single_a100_REALISTIC.sh  # ❌ Wrong directory
```

**After:**
```bash
/content# git clone https://github.com/ideaweaver-ai/nanochat.git
/content# cd nanochat
/content/nanochat# CONFIG=demo bash run_single_a100_REALISTIC.sh  # ✅ Correct!
```

The script now checks and gives helpful error messages if you're in the wrong place! 🚀
