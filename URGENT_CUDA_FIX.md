# URGENT: Fix CUDA to Use Your $19/Hour GPUs

## The Problem

You're paying $19/hour for 8x H100 GPUs, but PyTorch can't access them because `cuInit()` fails with error 999.

## Root Cause

The CUDA driver API cannot initialize even though:
- ✅ nvidia-smi works (shows 8 GPUs)
- ✅ CUDA toolkit is installed
- ✅ Environment variables are set
- ❌ `cuInit()` fails with error 999

This is a **container GPU runtime configuration issue**.

## Immediate Fix Options

### Option 1: Check Container Startup (MOST LIKELY FIX)

The container must be started with:
```bash
docker run --gpus all --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ...
```

Or in docker-compose:
```yaml
services:
  your-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Option 2: Try Setting CUDA_VISIBLE_DEVICES Explicitly

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash speedrun.sh
```

### Option 3: Check Host nvidia-container-toolkit

On the HOST (not in container):
```bash
nvidia-container-toolkit --version
```

If not installed:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Option 4: Try Different Container Image

Some container images have better GPU support. If you're using a custom image, try:
- `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
- Or ensure your image includes proper NVIDIA runtime support

## Diagnostic Commands

Run these to diagnose:

```bash
# 1. Test cuInit directly
python << 'EOF'
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int
result = libcuda.cuInit(0)
print(f"cuInit result: {result} (0=success, 999=unknown error)")
EOF

# 2. Check container runtime
echo "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-not set}"
echo "NVIDIA_DRIVER_CAPABILITIES: ${NVIDIA_DRIVER_CAPABILITIES:-not set}"

# 3. Check if nvidia-container-runtime is being used
docker inspect <container_id> | grep -i runtime
```

## Expected Result

After fixing, you should see:
```
✓ cuInit() succeeded!
✓ Found 8 GPU(s)
✓ PyTorch CUDA is available
```

## Cost Impact

- **Current**: Paying $19/hour but using CPU (wasteful!)
- **After fix**: Using all 8 H100 GPUs (what you're paying for!)

**This is urgent - you're wasting money on CPU mode!**

---

**Next Step**: Run `bash fix_cuda_urgent.sh` to diagnose and attempt fixes.
