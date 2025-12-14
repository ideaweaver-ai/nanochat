#!/bin/bash

# Fix CUDA/PyTorch issues in Colab
# This script fixes the libnvshmem_host.so.3 error

echo "Fixing CUDA/PyTorch installation..."

# Check current CUDA version
echo "Checking CUDA version..."
nvcc --version 2>/dev/null || echo "nvcc not found"
nvidia-smi | grep "CUDA Version" || echo "Could not detect CUDA version"

# Uninstall current PyTorch
echo "Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 11.8 (most compatible)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: Install PyTorch with CUDA 12.1 (if 11.8 doesn't work)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "✅ PyTorch reinstalled. Testing import..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
