#!/bin/bash
# Fix dependencies - clean install after pyproject.toml fix

set -e

echo "=== Fixing Dependencies ==="
echo ""

# Make sure uv is in PATH
if ! command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:${PATH}"
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:${PATH}"
fi

echo "1. Removing old lock file (if exists)..."
rm -f uv.lock

echo ""
echo "2. Removing venv to start fresh..."
rm -rf .venv

echo ""
echo "3. Creating new venv..."
uv venv

echo ""
echo "4. Installing dependencies with GPU support..."
uv sync --extra gpu

echo ""
echo "5. Activating venv and verifying installation..."
source .venv/bin/activate

# Set library paths
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

echo ""
echo "6. Verifying PyTorch installation..."
python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print(f"✓ PyTorch {torch.__version__} installed")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("✗ CUDA not available")
    exit(1)
PYEOF

echo ""
echo "7. Verifying other dependencies..."
python -c "import transformers; print(f'✓ transformers {transformers.__version__}')"
python -c "import wandb; print(f'✓ wandb {wandb.__version__}')"
python -c "import psutil; print(f'✓ psutil {psutil.__version__}')"
python -c "import pyarrow; print(f'✓ pyarrow {pyarrow.__version__}')"

echo ""
echo "=== ✓ ALL DEPENDENCIES INSTALLED! ==="
echo "Now run: bash speedrun.sh"
