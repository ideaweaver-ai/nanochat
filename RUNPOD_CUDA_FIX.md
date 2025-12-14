# RunPod CUDA Fix - URGENT

## The Problem

You're paying $19/hour for RunPod GPUs, but `cuInit()` fails with error 999, meaning PyTorch can't access the GPUs.

## Root Cause

**RunPod pods need GPU enabled in the template settings.** Even though:
- ✅ nvidia-smi works
- ✅ Environment variables are set
- ✅ CUDA toolkit is installed

The pod may have been created **without GPU access enabled in the template**.

## Solution: Check RunPod Template Settings

### Step 1: Check Your Pod Template

1. Go to [RunPod Dashboard](https://www.runpod.io/console/pods)
2. Find your pod
3. Click on the pod to see details
4. Check if **GPU** is listed and enabled

### Step 2: Verify GPU Count

In the pod details, you should see:
- **GPU Count**: Should be > 0 (ideally 8 for H100s)
- **GPU Type**: Should show your GPU model (H100, A100, etc.)

### Step 3: If GPU is NOT Enabled

**You need to recreate the pod with GPU enabled:**

1. **Stop the current pod** (to save money!)
2. **Create a new pod** with:
   - GPU enabled in template
   - GPU count: 8 (or however many you need)
   - GPU type: H100 (or your GPU type)
3. **Start the new pod**

### Step 4: Alternative - Check RunPod API/CLI

If using RunPod API or CLI, ensure your pod creation includes:
```json
{
  "gpuCount": 8,
  "gpuTypeId": "your-gpu-type-id"
}
```

## Quick Test

Run this to verify GPU access:
```bash
bash fix_runpod_cuda.sh
```

## If Template is Correct But Still Fails

1. **Check RunPod Status**: Sometimes RunPod has GPU allocation issues
2. **Try Different GPU Type**: Some GPU types may have better compatibility
3. **Contact RunPod Support**: They can check if there's a pod-level issue

## Cost Impact

- **Current**: Paying $19/hour but using CPU (WASTING MONEY!)
- **After fix**: Using all 8 H100 GPUs (what you're paying for!)

**STOP THE POD NOW if GPU isn't enabled - you're wasting money!**

---

**Next Step**: Check your RunPod dashboard and verify GPU is enabled in the pod template.
