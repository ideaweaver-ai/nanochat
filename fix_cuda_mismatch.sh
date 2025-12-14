#!/bin/bash
# Fix CUDA version mismatch between PyTorch and system CUDA

set -e

echo "=== Fixing CUDA Version Mismatch ==="
echo ""

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Activated virtual environment"
else
    echo "✗ Virtual environment not found. Please run: uv venv"
    exit 1
fi

# Detect system CUDA version
echo "Detecting system CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✓ System CUDA version: $CUDA_VERSION"
else
    echo "⚠ nvcc not found, checking nvidia-smi..."
    if command -v nvidia-smi &> /dev/null; then
        # Get driver version and infer CUDA version
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
        echo "  Driver version: $DRIVER_VERSION"
        # H100 with driver 570 supports CUDA 12.x
        CUDA_VERSION="12.4"
        echo "  Inferred CUDA version: $CUDA_VERSION"
    else
        echo "✗ Cannot detect CUDA version"
        exit 1
    fi
fi

# Determine compatible PyTorch CUDA version
# CUDA 12.4 is compatible with PyTorch built for CUDA 12.1 or 12.4
if [[ "$CUDA_VERSION" == "12."* ]]; then
    PYTORCH_CUDA="cu121"  # CUDA 12.1 builds work with 12.4
    echo "✓ Will install PyTorch for CUDA 12.1 (compatible with CUDA $CUDA_VERSION)"
else
    echo "⚠ Unsupported CUDA version: $CUDA_VERSION"
    exit 1
fi

# Check current PyTorch
echo ""
echo "Current PyTorch installation:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA compiled: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"N/A\"}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch not installed"

# Uninstall current PyTorch
echo ""
echo "Uninstalling current PyTorch..."
uv pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install compatible PyTorch
echo ""
echo "Installing PyTorch for CUDA $PYTORCH_CUDA..."
uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'✓ GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('✗ CUDA still not available')
    exit(1)
"

echo ""
echo "=== Fix Complete ==="
echo "PyTorch should now be able to access your GPUs."
