# Realistic Single A100 Training Guide

## ⚠️ Reality Check

**You're absolutely right** - training on a single 40GB A100 with full datasets would take **DAYS or WEEKS**, not hours.

This guide provides **realistic options** for single GPU training.

---

## The Problem

### Full Training Requirements:
- **Stage 1 (Base)**: 11.2B tokens for d20 model = **~71,680 iterations** at batch_size=524K
  - On single GPU: **~50-100 hours** (2-4 days)
- **Stage 2 (Mid)**: 850K examples = **~1,700 iterations** at batch_size=500
  - On single GPU: **~10-20 hours**
- **Stage 3 (SFT)**: 23K examples = **~460 iterations** at batch_size=50
  - On single GPU: **~2-4 hours**

**Total: 60-120 hours (2.5-5 days) of continuous training**

---

## Solution: Limited Training for Learning

Use `run_single_a100_REALISTIC.sh` which limits iterations:

### Option 1: DEMO Mode (1 hour)
```bash
CONFIG=demo bash run_single_a100_REALISTIC.sh
```
- **Base**: 500 iterations (~30-60 min)
- **Mid**: 200 iterations (~15-30 min)
- **SFT**: 100 iterations (~10-20 min)
- **Purpose**: See the pipeline work, validate setup
- **Model quality**: Poor (just for learning)

### Option 2: SMALL Mode (3-6 hours)
```bash
CONFIG=small bash run_single_a100_REALISTIC.sh
```
- **Base**: 2,000 iterations (~2-4 hours)
- **Mid**: 500 iterations (~1 hour)
- **SFT**: 200 iterations (~30 min)
- **Purpose**: Better model, still limited
- **Model quality**: Basic (can chat, but limited)

### Option 3: MEDIUM Mode (9-13 hours)
```bash
CONFIG=medium bash run_single_a100_REALISTIC.sh
```
- **Base**: 5,000 iterations (~6-10 hours)
- **Mid**: 1,000 iterations (~2 hours)
- **SFT**: 500 iterations (~1 hour)
- **Purpose**: Reasonable quality for learning
- **Model quality**: Decent (can have conversations)

---

## Key Changes in Realistic Script

### 1. Limited Iterations
- Uses `--num_iterations` instead of Chinchilla scaling
- Prevents infinite training loops
- You control exactly how long to train

### 2. Reduced Dataset Sizes
- **Tokenizer**: 500M chars (vs 2B)
- **Base training**: 10-40 shards (vs 240)
- **Evaluation**: Reduced sample sizes

### 3. Faster Evaluation
- Less frequent evals
- Smaller eval sets
- Disabled expensive metrics (CORE)

---

## What You Get vs. Full Training

| Aspect | Limited Training | Full Training |
|--------|-----------------|---------------|
| **Time** | 1-13 hours | 60-120 hours |
| **Iterations** | 500-5,000 | 71,680+ |
| **Data** | Subset | Full dataset |
| **Model Quality** | Basic/Demo | Production |
| **Purpose** | Learning | Production |

---

## Recommendations

### For Learning the Pipeline:
✅ Use **DEMO mode** first (1 hour)
- Validates your setup works
- See all stages execute
- Catch any errors early

### For Better Model Quality:
✅ Use **SMALL or MEDIUM mode** (3-13 hours)
- Still limited, but better results
- Can actually chat with the model
- Good for understanding training

### For Production:
❌ **Don't use single A100** - you need:
- Multiple GPUs (8xH100)
- Or accept 2-5 days of training
- Or use pre-trained models

---

## Alternative: Use Pre-trained Checkpoints

Instead of training from scratch, consider:

1. **Download pre-trained nanochat models** (if available)
2. **Fine-tune only** (much faster)
3. **Skip base training**, start from midtraining

This would reduce training time to **hours instead of days**.

---

## Time Estimates (Realistic)

### DEMO Mode:
- Tokenizer: 5-10 min
- Base: 30-60 min
- Mid: 15-30 min
- SFT: 10-20 min
- **Total: ~1-2 hours**

### SMALL Mode:
- Tokenizer: 10-15 min
- Base: 2-4 hours
- Mid: 1 hour
- SFT: 30 min
- **Total: ~3-6 hours**

### MEDIUM Mode:
- Tokenizer: 15-20 min
- Base: 6-10 hours
- Mid: 2 hours
- SFT: 1 hour
- **Total: ~9-13 hours**

---

## Memory Usage

All modes should fit in 40GB A100:
- **DEMO**: ~10-15GB
- **SMALL**: ~15-20GB
- **MEDIUM**: ~20-25GB

If you get OOM:
- Reduce `device_batch_size` further
- Reduce `max_seq_len`
- Reduce `depth`

---

## Bottom Line

**For single A100 40GB:**
- ✅ Use limited iterations (realistic script)
- ✅ Accept lower model quality
- ✅ Use for learning, not production
- ❌ Don't try full training (takes days)

**The realistic script makes training feasible, but with trade-offs.**
