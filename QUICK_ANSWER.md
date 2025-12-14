# Quick Answer: Does run_single_a100.sh Handle All Stages?

## ✅ YES - The Script Handles All Three Stages

The `run_single_a100.sh` script executes all three training stages sequentially:

### Stage 1: Base Training ✅
**Lines 126-148 in run_single_a100.sh**
```bash
python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --run=$WANDB_RUN
```

**Dataset**: FineWeb-Edu 100BT (raw text, parquet shards)
- Downloads automatically via `python -m nanochat.dataset -n $DATA_SHARDS`
- ~80-180 shards depending on model size
- Each shard: ~250M characters

**Features**:
- ✅ Pretraining on raw text
- ✅ Chinchilla scaling (tokens = 20 × params)
- ✅ Muon optimizer (transformer layers)
- ✅ AdamW optimizer (embeddings + LM head)
- ✅ Gradient accumulation (automatic)
- ✅ LR warmup/warmdown
- ✅ Checkpointing support
- ✅ CORE metric evaluation

---

### Stage 2: Midtraining ✅
**Lines 150-163 in run_single_a100.sh**
```bash
python -m scripts.mid_train \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --max_seq_len=$MAX_SEQ_LEN \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --run=$WANDB_RUN
```

**Dataset**: TaskMixture (~850K examples)
- SmolTalk: 460K conversations
- MMLU: 100K multiple choice
- GSM8K: 8K math problems
- Identity: 2K synthetic conversations
- SimpleSpelling: 200K spelling tasks
- SpellingBee: 80K counting tasks

**Features**:
- ✅ Conversation format learning
- ✅ Tool use (Python calculator)
- ✅ Multiple choice reasoning
- ✅ Same optimizer setup (Muon + AdamW)
- ✅ Progress-based LR scheduling

---

### Stage 3: Supervised Fine-Tuning (SFT) ✅
**Lines 165-177 in run_single_a100.sh**
```bash
python -m scripts.chat_sft \
    --device_batch_size=$SFT_BATCH_SIZE \
    --run=$WANDB_RUN
```

**Dataset**: TaskMixture (~23K examples)
- ARC-Easy: 2.3K science questions
- ARC-Challenge: 1.1K hard science
- GSM8K: 8K math problems
- SmolTalk: 10K conversations (subset)
- Identity: 1K conversations
- SimpleSpelling: 300 examples
- SpellingBee: 300 examples

**Features**:
- ✅ Domain adaptation for chat
- ✅ Variable-length sequences
- ✅ Masked loss (only assistant tokens)
- ✅ Smaller, focused dataset

---

## Training Datasets Summary

| Stage | Primary Dataset | Size | Source |
|-------|----------------|------|--------|
| **1. Base** | FineWeb-Edu 100BT | ~4-11B tokens | HuggingFace (karpathy/fineweb-edu-100b-shuffle) |
| **2. Mid** | TaskMixture | ~850K examples | Multiple (SmolTalk, MMLU, GSM8K, etc.) |
| **3. SFT** | TaskMixture | ~23K examples | Curated subset of Stage 2 datasets |

---

## All Features Are Preserved

The script maintains all features from the original 8xH100 setup:
- ✅ All three training stages
- ✅ All optimizer features (Muon + AdamW)
- ✅ All evaluation steps
- ✅ All dataset downloads
- ✅ All checkpointing
- ✅ All reporting

**Only difference**: Runs on single GPU instead of 8 GPUs (slower, but functionally identical)

---

## To Run

Simply execute:
```bash
bash run_single_a100.sh
```

Or with specific config:
```bash
CONFIG=conservative bash run_single_a100.sh  # For 40GB A100
CONFIG=aggressive bash run_single_a100.sh   # For 80GB A100
```

The script handles everything automatically! 🚀
