#!/bin/bash
# Quick diagnostic script to check CUDA/PyTorch setup

echo "=== CUDA/PyTorch Diagnostic ==="
echo ""

# Check nvidia-smi
echo "1. Checking nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    echo "  Detected $GPU_COUNT GPU(s)"
else
    echo "✗ nvidia-smi not found"
fi
echo ""

# Check CUDA in PATH
echo "2. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found"
    nvcc --version | grep "release"
else
    echo "⚠ nvcc not in PATH (this is OK if using conda/pip CUDA)"
fi
echo ""

# Check environment variables
echo "3. Checking CUDA environment variables..."
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo ""

# Check PyTorch
echo "4. Checking PyTorch installation..."
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else \"N/A\"}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  ✗ CUDA not available in PyTorch')
    print(f'  CUDA compiled version: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"N/A\"}')
" 2>&1
else
    echo "✗ Virtual environment not found. Run: source .venv/bin/activate"
fi
echo ""

# Check PyTorch installation method
echo "5. Checking PyTorch installation method..."
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -c "
import torch
import torch.utils.collect_env as collect_env
try:
    print(collect_env.get_pretty_env_info())
except:
    print('Could not get detailed environment info')
" 2>&1 | head -20
fi
echo ""

echo "=== Recommendations ==="
echo ""
if command -v nvidia-smi &> /dev/null && [ -f ".venv/bin/python" ]; then
    CUDA_AVAILABLE=$(.venv/bin/python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
    if [ "$CUDA_AVAILABLE" = "no" ]; then
        echo "⚠ PyTorch cannot access CUDA even though GPUs are detected."
        echo ""
        echo "Try these fixes:"
        echo "1. Reinstall PyTorch with CUDA support:"
        echo "   source .venv/bin/activate"
        echo "   uv pip uninstall torch torchvision torchaudio -y"
        echo "   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        echo ""
        echo "2. Or check CUDA version compatibility:"
        echo "   nvidia-smi  # Check driver version"
        echo "   python -c 'import torch; print(torch.version.cuda)'  # Check PyTorch CUDA version"
        echo ""
        echo "3. If CUDA_VISIBLE_DEVICES is set, try unsetting it:"
        echo "   unset CUDA_VISIBLE_DEVICES"
    else
        echo "✓ PyTorch CUDA is working correctly!"
    fi
fi
