# Training Datasets Used in NanoChat

This document details the datasets used in each training stage of nanochat.

## Overview

Yes, the `run_single_a100.sh` script **does handle all three stages**:
1. ✅ **Stage 1: Base Training** (lines 126-148)
2. ✅ **Stage 2: Midtraining** (lines 150-163)
3. ✅ **Stage 3: Supervised Fine-Tuning (SFT)** (lines 165-177)

All three stages are executed sequentially in the script, with proper evaluation after each stage.

---

## Stage 1: Base Training (Pretraining)

### Script: `scripts/base_train.py`
### Dataset: **FineWeb-Edu 100BT**

**Source**: HuggingFace dataset `karpathy/fineweb-edu-100b-shuffle`
- **URL**: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle
- **Format**: Parquet files (shards)
- **Total shards**: 1,822 shards available
- **Shard size**: ~250M characters per shard (~100MB compressed)
- **Total dataset**: ~455B characters (100B tokens after tokenization)

**What it contains**:
- Educational web text data
- High-quality, filtered content
- Preprocessed and shuffled

**How it's used**:
- **Tokenizer training**: First 8 shards (~2B characters)
- **Base pretraining**: Number of shards depends on model size
  - d20 model: 240 shards (~60B chars = ~11.2B tokens)
  - d12 model: 80 shards (~20B chars = ~4B tokens)
  - d14 model: 120 shards (~30B chars = ~6B tokens)
  - d16 model: 160 shards (~40B chars = ~8B tokens)

**Data loading**:
- Streams from parquet files on-the-fly
- Tokenized during data loading
- Distributed across GPUs (or sequential for single GPU)
- Each document prefixed with `<|bos|>` token

**Training objective**:
- Next token prediction (autoregressive language modeling)
- Causal attention mask
- Standard cross-entropy loss

**Chinchilla scaling**:
- Tokens = 20 × Parameters
- Ensures compute-optimal training

---

## Stage 2: Midtraining

### Script: `scripts/mid_train.py`
### Dataset: **TaskMixture** (Multiple datasets combined)

The midtraining stage uses a **mixture of 6 different datasets**:

#### 1. **SmolTalk** (460K conversations)
- **Source**: HuggingFace SmolTalk dataset
- **Type**: General conversational data
- **Purpose**: Teaches basic conversation format
- **Format**: Chat conversations with user/assistant turns

#### 2. **MMLU Auxiliary Train** (100K examples)
- **Source**: MMLU (Massive Multitask Language Understanding) dataset
- **Subset**: `auxiliary_train` 
- **Type**: Multiple choice questions
- **Topics**: ARC, MC_TEST, OBQA, RACE problems
- **Purpose**: Teaches multiple choice reasoning
- **Format**: Questions with multiple choice answers

#### 3. **GSM8K** (8K examples)
- **Source**: Grade School Math 8K dataset
- **Type**: Math word problems
- **Purpose**: Teaches mathematical reasoning and calculator tool use
- **Format**: Math problems with step-by-step solutions
- **Special**: Includes Python calculator tool usage examples

#### 4. **Identity Conversations** (1K × 2 = 2K examples)
- **Source**: Synthetic data from `dev/gen_synthetic_data.py`
- **Download**: https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
- **Type**: Synthetic personality conversations
- **Purpose**: Imparts a specific personality/identity to the model
- **Format**: Custom conversations defining model behavior
- **Note**: Used twice (2 epochs) for emphasis

#### 5. **SimpleSpelling** (200K examples)
- **Type**: Synthetic spelling tasks
- **Purpose**: Teaches spelling capabilities
- **Format**: "Spell the word 'apple'" → "a-p-p-l-e"

#### 6. **SpellingBee** (80K examples)
- **Type**: Synthetic letter counting tasks
- **Purpose**: Teaches counting letters in words
- **Format**: "How many 'r' are in 'strawberry'?" → "3"

**Total training examples**: 460K + 100K + 8K + 2K + 200K + 80K = **~850K rows**

**Validation dataset**:
- SmolTalk test: 24K rows
- MMLU test: 5.2K rows (subset)
- GSM8K test: 420 rows (subset)
- **Total**: ~39K validation examples

**Training objective**:
- Conversation format learning
- Tool use (Python calculator)
- Multiple choice reasoning
- Special token handling (`<|user_start|>`, `<|assistant_start|>`, etc.)

**Data mixing**:
- Deterministic shuffle ensures tasks are interleaved
- All tasks mixed throughout training (not sequential)

---

## Stage 3: Supervised Fine-Tuning (SFT)

### Script: `scripts/chat_sft.py`
### Dataset: **TaskMixture** (Smaller, curated mix)

The SFT stage uses a **smaller, more focused dataset mix**:

#### 1. **ARC-Easy** (2.3K examples)
- **Source**: AI2 Reasoning Challenge (Easy subset)
- **Type**: Science questions (multiple choice)
- **Purpose**: Domain-specific reasoning

#### 2. **ARC-Challenge** (1.1K examples)
- **Source**: AI2 Reasoning Challenge (Challenge subset)
- **Type**: Harder science questions
- **Purpose**: Advanced reasoning

#### 3. **GSM8K** (8K examples)
- **Source**: Grade School Math 8K (full train set)
- **Type**: Math word problems
- **Purpose**: Mathematical reasoning

#### 4. **SmolTalk** (10K examples, limited)
- **Source**: SmolTalk dataset (subset)
- **Type**: General conversations
- **Purpose**: Conversation quality
- **Note**: Limited to 10K (vs 460K in midtraining)

#### 5. **Identity Conversations** (1K examples)
- **Source**: Same as midtraining
- **Purpose**: Maintain personality

#### 6. **SimpleSpelling** (300 examples)
- **Type**: Spelling tasks
- **Purpose**: Maintain spelling capability
- **Note**: Much smaller than midtraining (300 vs 200K)

#### 7. **SpellingBee** (300 examples)
- **Type**: Letter counting tasks
- **Purpose**: Maintain counting capability
- **Note**: Much smaller than midtraining (300 vs 80K)

**Total training examples**: 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = **~23K rows**

**Validation dataset**:
- SmolTalk test: 24K rows (full test set)

**Training objective**:
- Domain adaptation for chat
- Variable-length sequences
- Masked loss (only train on assistant tokens)
- User tokens are masked out in loss calculation

**Key differences from midtraining**:
- **Much smaller dataset** (23K vs 850K)
- **More focused** on quality over quantity
- **Variable sequence lengths** (with padding)
- **Masked training** (only assistant responses contribute to loss)

---

## Dataset Summary Table

| Stage | Dataset | Examples | Purpose | Format |
|-------|---------|----------|---------|--------|
| **Stage 1: Base** | FineWeb-Edu | ~11.2B tokens (d20) | Language modeling | Raw text |
| **Stage 2: Mid** | SmolTalk | 460K | Conversations | Chat format |
| | MMLU Aux | 100K | Multiple choice | Q&A |
| | GSM8K | 8K | Math + tools | Math problems |
| | Identity | 2K | Personality | Synthetic |
| | SimpleSpelling | 200K | Spelling | Spelling tasks |
| | SpellingBee | 80K | Counting | Letter counting |
| **Stage 3: SFT** | ARC-Easy | 2.3K | Science Q&A | Multiple choice |
| | ARC-Challenge | 1.1K | Hard science | Multiple choice |
| | GSM8K | 8K | Math | Math problems |
| | SmolTalk | 10K | Conversations | Chat format |
| | Identity | 1K | Personality | Synthetic |
| | SimpleSpelling | 300 | Spelling | Spelling tasks |
| | SpellingBee | 300 | Counting | Letter counting |

---

## Data Download Details

### Stage 1 (FineWeb):
- **Automatic download**: Yes, via `python -m nanochat.dataset -n <num_shards>`
- **Location**: `~/.cache/nanochat/base_data/`
- **Format**: `shard_00000.parquet`, `shard_00001.parquet`, etc.
- **Size**: ~100MB per shard (compressed)

### Stage 2 & 3 (Task datasets):
- **Automatic download**: Yes, via HuggingFace `datasets` library
- **Cached location**: `~/.cache/huggingface/datasets/`
- **Identity conversations**: Downloaded from S3 URL
- **Synthetic tasks**: Generated on-the-fly (SimpleSpelling, SpellingBee)

---

## Notes

1. **All datasets are automatically downloaded** - no manual setup needed
2. **Tokenizer is trained first** on FineWeb data before any model training
3. **Data sharding** for FineWeb allows efficient streaming
4. **Task mixtures** ensure diverse training throughout
5. **Validation sets** are separate from training sets
6. **Single GPU**: All datasets work the same way, just slower

---

## Verification

To verify the script handles all stages, check `run_single_a100.sh`:

- **Lines 126-148**: Stage 1 (Base Training) ✅
- **Lines 150-163**: Stage 2 (Midtraining) ✅  
- **Lines 165-177**: Stage 3 (SFT) ✅

All stages use the same optimizer setup (Muon + AdamW) and include evaluation steps.
