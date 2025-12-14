# Model Scaling to 5M/1M Parameters & RL Training Time Analysis

## Question 1: Can we scale down to 5M or 1M parameters?

**Short answer: Yes, it's absolutely possible!** The nanochat architecture is designed to scale by adjusting a single parameter: `depth`.

### How Model Size is Determined

The model architecture follows this formula:
```python
num_layers = depth
model_dim = depth * 64  # aspect ratio 64
num_heads = max(1, (model_dim + 127) // 128)  # head dim 128
vocab_size = 65536  # fixed
```

### Parameter Count Formula

The total parameters come from:
1. **Embedding layer**: `vocab_size × model_dim`
2. **LM head**: `model_dim × vocab_size`
3. **Transformer blocks** (depth layers):
   - **Attention per block**: `4 × model_dim²` (c_q, c_k, c_v, c_proj)
   - **MLP per block**: `8 × model_dim²` (c_fc: model_dim × 4×model_dim, c_proj: 4×model_dim × model_dim)
   - **Total per block**: `12 × model_dim²`
   - **All blocks**: `depth × 12 × model_dim²`

**Total parameters** = `2 × vocab_size × model_dim + depth × 12 × model_dim²`

### Parameter Counts for Different Depths

| Depth | Model Dim | Heads | Parameters | Approx |
|-------|-----------|-------|------------|--------|
| 20    | 1,280     | 10    | ~561M      | Current speedrun |
| 10    | 640       | 5     | ~70M       | ~8× smaller |
| 6     | 384       | 3     | ~25M       | ~22× smaller |
| **4** | **256**   | **2** | **~11M**   | **~51× smaller** |
| **3** | **192**   | **1** | **~6M**    | **~93× smaller** |
| **2** | **128**   | **1** | **~3.3M**  | **~170× smaller** |
| **1** | **64**    | **1** | **~1.6M**  | **~350× smaller** |

### Answer: Target Depths

- **For ~5M parameters**: Use `depth=3` (gives ~6M parameters, close enough)
- **For ~1M parameters**: Use `depth=1` (gives ~1.6M parameters)

**Note**: `depth=1` is extremely small and may have limited capabilities. `depth=3-4` is more practical for a 5M model.

### How to Train Smaller Models

Simply modify `speedrun.sh`:

```bash
# For ~5M parameters (depth=3)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=3 --run=$WANDB_RUN

# For ~1M parameters (depth=1) 
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=1 --run=$WANDB_RUN
```

### Important Considerations for Small Models

1. **Training Data Scaling**: 
   - Chinchilla scaling suggests: tokens = 20 × parameters
   - For 5M params: ~100M tokens needed
   - For 1M params: ~20M tokens needed
   - You'll need fewer data shards (calculate: tokens × 4.8 chars/token ÷ 250M chars/shard)

2. **Batch Size Adjustments**:
   - Smaller models use less VRAM
   - You can increase `device_batch_size` if needed
   - Or use fewer GPUs

3. **Performance Expectations**:
   - **5M model**: Will be very limited, similar to early GPT-2 small variants
   - **1M model**: Extremely limited, mostly useful for educational purposes
   - Both will make many mistakes and have poor reasoning

4. **Training Time**:
   - Much faster! Roughly proportional to parameter count
   - 5M model: ~30-45 minutes (instead of 4 hours)
   - 1M model: ~10-15 minutes (instead of 4 hours)

### Example: Modified speedrun.sh for 5M Parameters

```bash
# ... setup code stays the same ...

# Download fewer data shards (for 5M params: 100M tokens × 4.8 ÷ 250M ≈ 2 shards)
python -m nanochat.dataset -n 2
python -m nanochat.dataset -n 10 &  # Background download for pretraining
DATASET_DOWNLOAD_PID=$!

# Train tokenizer (same)
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# Wait for data
wait $DATASET_DOWNLOAD_PID

# Pretrain with depth=3 (5M params)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=3 --run=$WANDB_RUN

# ... rest stays the same ...
```

### Memory Requirements

Smaller models use much less VRAM:
- **d20 (561M)**: Needs ~80GB GPUs
- **d3 (5M)**: Could fit on ~8-16GB GPUs
- **d1 (1M)**: Could fit on ~4GB GPUs

You might even be able to train on consumer GPUs!

---

## Question 2: How much time does Reinforcement Learning add?

**Short answer: Approximately 30-60 minutes on 8×H100 GPUs**

### RL Training Details

From `scripts/chat_rl.py`, the RL phase:

1. **Training Data**: GSM8K dataset only
   - Train set: ~8,000 math problems
   - Test set: ~1,300 problems

2. **Training Configuration**:
   - **Examples per step**: 16 (across all 8 GPUs)
   - **Samples per example**: 16 (generates 16 completions per problem)
   - **Total sequences per step**: 16 × 16 = 256 sequences
   - **Max new tokens**: 256 per completion
   - **Number of epochs**: 1 (default)
   - **Number of steps**: `(len(train_task) // examples_per_step) × num_epochs`
     - For GSM8K: `(8000 // 16) × 1 = 500 steps`

3. **What happens each step**:
   - For each example (16 examples):
     - Generate 16 samples per example (256 total completions)
     - Evaluate rewards (0 or 1 for correct/incorrect)
     - Calculate advantages (reward - mean reward)
     - Compute policy gradient loss
     - Backpropagate and update
   - Evaluate on validation set every 60 steps

4. **Time per step**:
   - Generation: ~2-5 seconds (256 sequences × 256 tokens each)
   - Forward/backward: ~1-2 seconds
   - Evaluation (every 60 steps): ~30-60 seconds
   - **Total per step**: ~3-7 seconds average

5. **Total RL time**:
   - 500 steps × ~5 seconds = ~2,500 seconds = **~42 minutes**
   - Plus evaluation overhead: **~50-60 minutes total**

### Detailed Time Breakdown

| Phase | Time | Notes |
|-------|------|-------|
| Generation per step | 2-5s | 256 sequences, 256 tokens each |
| Forward/backward | 1-2s | Policy gradient computation |
| Evaluation (every 60 steps) | 30-60s | Pass@k on 400 validation examples |
| **Total per step** | **3-7s** | Average |
| **500 steps total** | **~42 min** | Pure training |
| **With evaluations** | **~50-60 min** | Including periodic evals |

### Factors Affecting RL Time

1. **Model size**: Larger models = slower generation
   - d20 (561M): ~50-60 minutes
   - d3 (5M): ~10-15 minutes
   - d1 (1M): ~5-10 minutes

2. **Number of samples**: Default is 16 per example
   - More samples = better learning but slower
   - Can reduce to 8 for faster training

3. **Max tokens**: Default is 256
   - Longer completions = slower generation
   - Math problems usually need <256 tokens

4. **Evaluation frequency**: Default every 60 steps
   - More frequent = better monitoring but slower
   - Can increase to 120 for faster training

### Modified RL for Faster Training

If you want to speed up RL:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl \
    -- --run=$WANDB_RUN \
    --num_samples=8 \        # Reduce from 16 to 8
    --max_new_tokens=128 \   # Reduce from 256 to 128
    --eval_every=120 \        # Evaluate less frequently
    --eval_examples=200       # Evaluate on fewer examples
```

This could reduce RL time to **~20-30 minutes**.

### Complete Pipeline with RL

| Phase | Time (d20) | Time (d3) | Time (d1) |
|-------|------------|-----------|-----------|
| Setup | 5 min | 5 min | 5 min |
| Tokenizer | 15-20 min | 15-20 min | 15-20 min |
| Base Pretraining | 2.5-3 hours | 30-45 min | 10-15 min |
| Midtraining | 30-45 min | 10-15 min | 5-10 min |
| SFT | 15-20 min | 5-10 min | 3-5 min |
| **RL (optional)** | **50-60 min** | **10-15 min** | **5-10 min** |
| Evaluation | 10-15 min | 5-10 min | 3-5 min |
| **Total (no RL)** | **~4 hours** | **~1.5 hours** | **~45 min** |
| **Total (with RL)** | **~5 hours** | **~2 hours** | **~1 hour** |

### RL Performance Improvement

Based on the README metrics:
- **GSM8K without RL**: ~0.025-0.045 (2.5-4.5%)
- **GSM8K with RL**: ~0.0758 (7.58%)
- **Improvement**: ~2-3× better on math problems

RL specifically helps with:
- Math reasoning (GSM8K)
- Structured problem-solving
- Following step-by-step instructions

It does NOT help much with:
- General conversation
- Multiple choice (ARC, MMLU)
- Coding (HumanEval)

### Recommendation

- **For d20 model**: Add RL if you have an extra hour and care about math performance
- **For smaller models (d3, d1)**: RL might not be worth it - the models are too small to benefit significantly
- **For production**: RL is optional but can help with specific tasks

---

## Summary

### Question 1: Scaling to 5M/1M Parameters
✅ **Yes, absolutely possible!**
- Use `depth=3` for ~6M params (close to 5M)
- Use `depth=1` for ~1.6M params (close to 1M)
- Training time scales roughly proportionally
- Much less VRAM needed (could use consumer GPUs)

### Question 2: RL Training Time
⏱️ **~50-60 minutes for d20 model**
- Scales with model size
- d3: ~10-15 minutes
- d1: ~5-10 minutes
- Can be optimized to ~20-30 minutes with reduced samples/tokens

### Complete Training Times

| Model Size | Params | No RL | With RL |
|------------|--------|-------|---------|
| d20        | 561M   | 4 hrs | 5 hrs  |
| d3         | ~6M    | 1.5 hrs | 2 hrs |
| d1         | ~1.6M  | 45 min | 1 hr  |
