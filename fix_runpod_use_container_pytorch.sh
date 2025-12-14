#!/bin/bash
# Use RunPod container's built-in PyTorch instead of installing incompatible version

set -e

echo "=== Using RunPod Container's Built-in PyTorch ==="
echo ""

source .venv/bin/activate

# Set library paths
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"

echo "1. Your RunPod container image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1"
echo "   This image ALREADY has PyTorch 2.4.0 with CUDA 12.4.1!"
echo ""

echo "2. Checking if system PyTorch works..."
# Test system Python (not venv) to see if container's PyTorch works
/usr/bin/python3 << 'PYEOF'
import sys
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import torch
    print(f"System PyTorch: {torch.__version__}")
    print(f"CUDA compiled: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"✓ System PyTorch CAN see CUDA! {torch.cuda.device_count()} GPUs")
        sys.exit(0)
    else:
        print("✗ System PyTorch cannot see CUDA")
        sys.exit(1)
except ImportError:
    print("System Python doesn't have PyTorch")
    sys.exit(1)
PYEOF

SYSTEM_PYTORCH_WORKS=$?

if [ $SYSTEM_PYTORCH_WORKS -eq 0 ]; then
    echo ""
    echo "✓ Container's built-in PyTorch WORKS!"
    echo ""
    echo "3. The problem: Your venv has incompatible PyTorch 2.8.0"
    echo "   Solution: Remove venv PyTorch and use system PyTorch"
    echo ""
    
    echo "Removing incompatible PyTorch from venv..."
    uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    
    echo ""
    echo "4. Making venv use system PyTorch..."
    # Create a .pth file to add system site-packages
    SYSTEM_SITE_PACKAGES=$(/usr/bin/python3 -c "import site; print(site.getsitepackages()[0])")
    echo "$SYSTEM_SITE_PACKAGES" > .venv/lib/python3.11/site-packages/system_pytorch.pth
    
    echo "   Added system site-packages to venv"
    
    echo ""
    echo "5. Testing venv with system PyTorch..."
    python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"✓ SUCCESS! CUDA is available!")
    print(f"  GPU count: {count}")
    for i in range(count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test operation
    x = torch.randn(2, 2).cuda()
    print(f"  ✓ GPU tensor creation successful!")
    exit(0)
else:
    print("✗ CUDA still not available")
    exit(1)
PYEOF

    VENV_RESULT=$?
    
    if [ $VENV_RESULT -eq 0 ]; then
        echo ""
        echo "=== ✓ SUCCESS! Using container's built-in PyTorch ==="
        echo "Your venv now uses the container's PyTorch 2.4.0 with CUDA 12.4.1"
        echo ""
        echo "Run speedrun.sh - it should work now!"
    else
        echo ""
        echo "Trying alternative: Install compatible PyTorch in venv..."
        uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
            --index-url https://download.pytorch.org/whl/cu121 \
            --no-cache-dir
        
        python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA works! {torch.cuda.device_count()} GPUs")
else:
    print("✗ Still not working")
PYEOF
    fi
else
    echo ""
    echo "System PyTorch also doesn't work. Installing compatible version..."
    uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --no-cache-dir
    
    python << 'PYEOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA works! {torch.cuda.device_count()} GPUs")
else:
    print("✗ Still not working - may need RunPod support")
PYEOF
fi
