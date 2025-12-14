# Running NanoChat on Google Colab

## ⚠️ Quick Answer

**No, `CONFIG=demo bash run_single_a100_REALISTIC.sh` will NOT work in Colab.**

Colab uses Jupyter notebooks, not bash shells. Use the **Colab notebook** instead!

---

## ✅ Solution: Use the Colab Notebook

I've created `NanoChat_Colab_Training.ipynb` - a ready-to-use Colab notebook.

### How to Use:

1. **Open Google Colab**: https://colab.research.google.com/
2. **Upload the notebook**: File → Upload notebook → Select `NanoChat_Colab_Training.ipynb`
3. **Change runtime**: Runtime → Change runtime type → GPU → A100
4. **Run all cells**: Runtime → Run all

That's it! The notebook handles everything.

---

## Alternative: Run Commands in Colab Cells

If you prefer to run commands manually, use `!` prefix in Colab cells:

### Cell 1: Setup
```python
# Clone repo
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat

# Install dependencies
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv
!uv sync --extra gpu
```

### Cell 2: Configuration
```python
# Set config
CONFIG = 'demo'  # or 'small' or 'medium'

# Set variables based on config
if CONFIG == 'demo':
    DEPTH = 8
    BASE_ITERATIONS = 500
    # ... etc
```

### Cell 3: Training
```python
# Run training commands with ! prefix
!python -m scripts.base_train \
    --depth=8 \
    --num_iterations=500 \
    --device_batch_size=4
```

---

## Key Differences: Bash Script vs Colab

| Aspect | Bash Script | Colab Notebook |
|--------|-------------|----------------|
| **Execution** | `bash script.sh` | Run cells sequentially |
| **Variables** | `$VAR` or `${VAR}` | Python variables |
| **Commands** | Direct | Use `!` prefix |
| **Environment** | Shell | Python + Shell mix |
| **File paths** | `~/.cache/` | `/content/.cache/` |

---

## Colab-Specific Considerations

### 1. File Paths
- Colab uses `/content/` as root
- Change: `$HOME/.cache/nanochat` → `/content/.cache/nanochat`

### 2. Time Limits
- **Free Colab**: 12 hours max
- **Pro Colab**: 24 hours max
- **Solution**: Use `demo` or `small` config

### 3. Environment Variables
```python
import os
os.environ['NANOCHAT_BASE_DIR'] = '/content/.cache/nanochat'
os.environ['OMP_NUM_THREADS'] = '1'
```

### 4. Background Processes
```python
import subprocess
process = subprocess.Popen(['python', '-m', 'nanochat.dataset', '-n', '10'])
# Later:
process.wait()
```

### 5. Path Setup
```python
import sys
sys.path.insert(0, '/content/nanochat/.venv/lib/python3.10/site-packages')
```

---

## Recommended Workflow

### Option 1: Use the Notebook (Easiest) ⭐
1. Upload `NanoChat_Colab_Training.ipynb`
2. Change runtime to A100
3. Run all cells
4. Done!

### Option 2: Manual Setup
1. Clone repo in Colab
2. Install dependencies
3. Run training commands in cells
4. Save to Drive

---

## Quick Start Commands for Colab

```python
# Cell 1: Setup
!git clone https://github.com/ideaweaver-ai/nanochat.git
%cd nanochat
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv && uv sync --extra gpu

# Cell 2: Config
CONFIG = 'demo'  # Change this
DEPTH = 8
BASE_ITERATIONS = 500

# Cell 3: Train
!python -m scripts.base_train \
    --depth={DEPTH} \
    --num_iterations={BASE_ITERATIONS} \
    --device_batch_size=4
```

---

## Troubleshooting

### "Command not found"
- Use `!` prefix for shell commands
- Use Python code for Python operations

### "Module not found"
- Activate venv: `sys.path.insert(0, '/path/to/venv')`
- Or use: `!source .venv/bin/activate && python script.py`

### "Out of memory"
- Reduce `device_batch_size`
- Reduce `max_seq_len`
- Use `demo` config

### "Session disconnected"
- Colab has time limits
- Save checkpoints to Drive
- Resume from checkpoint

---

## Summary

✅ **Use**: `NanoChat_Colab_Training.ipynb` notebook  
❌ **Don't use**: `bash run_single_a100_REALISTIC.sh` (won't work)

The notebook is ready to go - just upload and run! 🚀
