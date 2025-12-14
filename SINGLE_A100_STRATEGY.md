# Single A100 GPU Training Strategy for NanoChat

## Overview
This document outlines the strategy to adapt nanochat training from 8xH100 (640GB total) to a single A100 GPU (40GB or 80GB).

## Key Constraints

### Memory Constraints
- **8xH100**: 8 × 80GB = 640GB total VRAM
- **Single A100**: 40GB or 80GB VRAM
- **Reduction factor**: ~8-16x less memory

### Current Default Settings
- **Model depth**: 20 layers** (d20 = 561M parameters)
- **device_batch_size**: 32 per GPU
- **max_seq_len**: 2048 tokens
- **total_batch_size**: 524,288 tokens (across 8 GPUs)
- **Effective batch per GPU**: 32 × 2048 = 65,536 tokens/GPU
- **Total effective**: 8 × 65,536 = 524,288 tokens

## Strategy Components

### 1. Remove Distributed Training
- **Change**: Remove `torchrun` and run directly with `python -m scripts.base_train`
- **Impact**: Code automatically handles single GPU via `compute_init()` function
- **No code changes needed**: The code already supports single GPU mode

### 2. Reduce Model Size (Option A: Smaller Model)
**For 40GB A100:**
- **depth**: 12-14 layers (instead of 20)
- **Parameters**: ~200-300M (instead of 561M)
- **Rationale**: Fits comfortably in 40GB with reasonable batch size

**For 80GB A100:**
- **depth**: 16-18 layers
- **Parameters**: ~400-500M
- **Rationale**: Can fit larger model with careful batch size tuning

### 3. Reduce Batch Size
**For 40GB A100:**
- **device_batch_size**: 4-8 (instead of 32)
- **max_seq_len**: 1024-1536 (instead of 2048)
- **Rationale**: Major memory savings

**For 80GB A100:**
- **device_batch_size**: 8-16
- **max_seq_len**: 1536-2048
- **Rationale**: Can use larger batches

### 4. Increase Gradient Accumulation
- **Maintain total_batch_size**: Keep at 524,288 tokens (or reduce proportionally)
- **Calculation**: 
  - Single GPU: `grad_accum_steps = total_batch_size / (device_batch_size × max_seq_len)`
  - Example: `524288 / (8 × 1024) = 64` accumulation steps
- **Impact**: Training will be slower but maintains same effective batch size

### 5. Reduce Sequence Length (Optional)
- **Option**: Reduce `max_seq_len` from 2048 to 1024 or 1536
- **Memory savings**: ~2x for attention matrices
- **Trade-off**: Shorter context window

### 6. Enable Gradient Checkpointing (If Needed)
- **Status**: Not currently implemented in nanochat
- **Future**: Could add activation checkpointing for further memory savings

### 7. Adjust Data Requirements
- **Tokenizer training**: Same (2B chars)
- **Pretraining data**: 
  - Smaller model needs fewer tokens (Chinchilla: 20× params)
  - d12 model: ~4-6B tokens (vs 11.2B for d20)
  - Fewer data shards needed: ~80-120 shards (vs 240)

## Recommended Configurations

### Configuration 1: Conservative (40GB A100)
```bash
--depth=12 \
--max_seq_len=1024 \
--device_batch_size=4 \
--total_batch_size=131072  # Reduced from 524288
```
- **Model**: ~200M parameters
- **Training tokens**: ~4B tokens
- **Gradient accum**: 32 steps
- **Expected time**: ~8-12 hours

### Configuration 2: Balanced (40GB A100)
```bash
--depth=14 \
--max_seq_len=1536 \
--device_batch_size=4 \
--total_batch_size=262144
```
- **Model**: ~300M parameters
- **Training tokens**: ~6B tokens
- **Gradient accum**: 43 steps
- **Expected time**: ~12-16 hours

### Configuration 3: Aggressive (80GB A100)
```bash
--depth=16 \
--max_seq_len=2048 \
--device_batch_size=8 \
--total_batch_size=524288  # Same as original
```
- **Model**: ~400M parameters
- **Training tokens**: ~8B tokens
- **Gradient accum**: 32 steps
- **Expected time**: ~16-24 hours

### Configuration 4: Maximum (80GB A100)
```bash
--depth=18 \
--max_seq_len=2048 \
--device_batch_size=8 \
--total_batch_size=524288
```
- **Model**: ~500M parameters
- **Training tokens**: ~10B tokens
- **Gradient accum**: 32 steps
- **Expected time**: ~20-30 hours

## Memory Breakdown (Approximate)

For d20 model with batch_size=32, seq_len=2048:
- **Model weights**: ~2.2GB (bfloat16)
- **Optimizer states**: ~4.4GB (AdamW + Muon)
- **Activations**: ~15-20GB (batch × seq × hidden_dim)
- **Gradients**: ~2.2GB
- **KV cache**: ~5-10GB (if used)
- **Total**: ~30-40GB

For d12 model with batch_size=4, seq_len=1024:
- **Model weights**: ~0.8GB
- **Optimizer states**: ~1.6GB
- **Activations**: ~2-3GB
- **Gradients**: ~0.8GB
- **Total**: ~5-6GB (fits comfortably in 40GB)

## Training Time Estimates

### Original (8xH100, d20):
- **Time**: 4 hours
- **Throughput**: ~83K tokens/sec per GPU
- **Total**: ~664K tokens/sec

### Single A100 (d12, batch=4):
- **Time**: 8-12 hours
- **Throughput**: ~20-30K tokens/sec
- **Slowdown**: ~3-4x (due to single GPU + gradient accumulation)

## Implementation Steps

1. **Create single A100 script** (see `run_single_a100.sh`)
2. **Test memory usage** with small batch first
3. **Gradually increase** batch size until OOM
4. **Adjust model depth** if needed
5. **Monitor training** for stability

## Key Files to Modify

1. **Training script**: Create new `run_single_a100.sh`
2. **No code changes needed**: All scripts support single GPU via CLI args

## Testing Strategy

1. **Start small**: depth=8, batch=2, seq_len=512
2. **Verify it works**: Run 10 iterations
3. **Scale up**: Increase batch size until OOM
4. **Find sweet spot**: Balance between speed and memory
5. **Full training**: Run complete pipeline

## Troubleshooting

### Out of Memory (OOM)
- Reduce `device_batch_size` by half
- Reduce `max_seq_len` by 25%
- Reduce `depth` by 2 layers

### Training Too Slow
- Increase `device_batch_size` (if memory allows)
- Reduce `total_batch_size` (fewer gradient accum steps)
- Accept longer training time

### Poor Model Quality
- Ensure sufficient training tokens (20× params)
- Check learning rate schedule
- Verify data quality

## Conclusion

The nanochat codebase is already designed to work on single GPU. The main changes are:
1. Remove `torchrun` wrapper
2. Reduce hyperparameters (depth, batch_size, seq_len)
3. Increase gradient accumulation to maintain effective batch size
4. Adjust data requirements for smaller model

No code modifications are required - only hyperparameter tuning via CLI arguments.
