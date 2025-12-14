# Speedrun.sh Training Breakdown: Step-by-Step Details

This document provides a detailed breakdown of what exactly happens during the 4-hour `speedrun.sh` training script for the $100 tier nanochat model (d20 - 20 layers, 561M parameters).

## Overview

The speedrun script trains a complete ChatGPT-like model from scratch in approximately 4 hours on an 8XH100 GPU node. The total cost is ~$100 ($24/hr × 4 hours). The model trained is:
- **Architecture**: GPT-style Transformer with 20 layers (depth=20)
- **Parameters**: 561 million parameters
- **Training tokens**: ~11.2 billion tokens (following Chinchilla scaling: 20× parameters)
- **Training data**: ~54 billion characters (~240 data shards, each ~250M chars)

---

## Step-by-Step Training Pipeline

### **PHASE 1: Environment Setup** (~5 minutes)

#### Step 1.1: Python Environment Setup
- **Action**: Sets up Python virtual environment using `uv` package manager
- **Commands**:
  - Installs `uv` if not present
  - Creates `.venv` virtual environment
  - Installs dependencies with `uv sync --extra gpu`
  - Activates the virtual environment
- **Purpose**: Ensures all required Python packages are available

#### Step 1.2: Report System Initialization
- **Action**: Initializes the markdown report system
- **Command**: `python -m nanochat.report reset`
- **Purpose**: Creates report directory and writes header with system info and timestamp

#### Step 1.3: Rust/Cargo Installation
- **Action**: Installs Rust compiler for building custom tokenizer
- **Command**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`
- **Purpose**: Required for building the custom Rust BPE tokenizer

---

### **PHASE 2: Tokenizer Training** (~15-20 minutes)

#### Step 2.1: Build Rust BPE Tokenizer
- **Action**: Compiles the custom Rust-based BPE tokenizer
- **Command**: `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`
- **Purpose**: Builds the fast Rust implementation of Byte-Pair Encoding (BPE) tokenizer

#### Step 2.2: Download Initial Training Data
- **Action**: Downloads first 8 data shards (~2 billion characters) for tokenizer training
- **Command**: `python -m nanochat.dataset -n 8`
- **Details**:
  - Each shard is ~250M characters (~100MB compressed)
  - Total: ~800MB of text data
  - Data source: FineWeb dataset from HuggingFace

#### Step 2.3: Background Data Download
- **Action**: Starts downloading additional data shards in background
- **Command**: `python -m nanochat.dataset -n 240 &`
- **Details**:
  - Downloads 240 shards total (~24GB of data)
  - Runs in background while tokenizer trains
  - Needed for pretraining phase (calculated: 561M params × 20 × 4.8 chars/token ÷ 250M = ~216 shards, rounded to 240)

#### Step 2.4: Train BPE Tokenizer
- **Action**: Trains the tokenizer on ~2 billion characters
- **Command**: `python -m scripts.tok_train --max_chars=2000000000`
- **Details**:
  - **Vocabulary size**: 65,536 tokens (2^16)
  - **Training data**: 2 billion characters
  - **Algorithm**: Byte-Pair Encoding (BPE) in GPT-4 style
  - **Output**: Saves tokenizer to `~/.cache/nanochat/tokenizer/`
  - **Time**: ~10-15 minutes

#### Step 2.5: Evaluate Tokenizer
- **Action**: Evaluates tokenizer compression ratio
- **Command**: `python -m scripts.tok_eval`
- **Purpose**: Measures how efficiently the tokenizer compresses text (bits per byte)

---

### **PHASE 3: Base Model Pretraining** (~2.5-3 hours)

This is the **longest and most compute-intensive phase**. The model learns general language understanding from raw text.

#### Step 3.1: Wait for Data Download
- **Action**: Waits for background data download to complete
- **Command**: `wait $DATASET_DOWNLOAD_PID`
- **Purpose**: Ensures all 240 data shards are available before training starts

#### Step 3.2: Pretrain Base Model
- **Action**: Trains the GPT Transformer model from scratch
- **Command**: `torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN`
- **Model Architecture**:
  - **Depth**: 20 layers
  - **Model dimension**: 1,280 (depth × 64)
  - **Attention heads**: 10 (model_dim ÷ 128)
  - **KV heads**: 10 (GQA ratio 1:1, so no grouping)
  - **Max sequence length**: 2,048 tokens
  - **Total parameters**: ~561 million

- **Training Configuration**:
  - **Device batch size**: 32 sequences per GPU
  - **Total batch size**: 524,288 tokens per step
  - **Gradient accumulation**: Automatically calculated to reach total batch size
  - **Training tokens**: ~11.2 billion (561M params × 20, following Chinchilla scaling)
  - **Training iterations**: Calculated from tokens ÷ batch size
  - **Optimizers**:
    - **Muon optimizer**: For linear/transformer layers (LR: 0.02)
    - **AdamW optimizer**: For embeddings and output head (LR: 0.2 for embeddings, 0.004 for unembedding)
  - **Learning rate schedule**:
    - Warmup: 0% (no warmup)
    - Warmdown: 20% of training (linear decay to 0)
    - Final LR fraction: 0.0
  - **Gradient clipping**: 1.0
  - **Precision**: bfloat16

- **Training Process**:
  - **Evaluation every 250 steps**: Validates on held-out data, measures bits-per-byte (bpb)
  - **CORE metric every 2000 steps**: Evaluates on CORE benchmark (comprehensive evaluation)
  - **Sampling every 2000 steps**: Generates text samples to monitor quality
  - **Checkpoint saving**: Only at the end (or every `save_every` steps if configured)

- **What the model learns**:
  - General language patterns
  - Next-token prediction
  - Basic reasoning and factual knowledge
  - Text generation capabilities

- **Time**: ~2.5-3 hours (majority of the 4-hour runtime)

#### Step 3.3: Evaluate Base Model Loss
- **Action**: Evaluates model on larger validation set and generates samples
- **Command**: `torchrun --standalone --nproc_per_node=8 -m scripts.base_loss`
- **Details**:
  - Evaluates bits-per-byte on train/val splits (20 × 524,288 tokens each)
  - Generates text samples from various prompts
  - Measures model's raw language modeling capability

#### Step 3.4: Evaluate Base Model CORE Metric
- **Action**: Comprehensive evaluation on CORE benchmark
- **Command**: `torchrun --standalone --nproc_per_node=8 -m scripts.base_eval`
- **Details**:
  - Evaluates on CORE benchmark (comprehensive evaluation suite)
  - Measures base model's capabilities across multiple tasks
  - Downloads ~162MB evaluation bundle if needed
  - **Output**: CORE metric score (typically ~0.22 for d20 model)

---

### **PHASE 4: Midtraining** (~30-45 minutes)

This phase teaches the model to handle conversations, special tokens, and structured tasks.

#### Step 4.1: Download Identity Conversations
- **Action**: Downloads synthetic identity/personality data
- **Command**: `curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl`
- **Details**:
  - ~2.3MB of synthetic conversations
  - Imparts personality to the model
  - ~1,000 conversation examples

#### Step 4.2: Midtrain Model
- **Action**: Trains model on conversation format and structured tasks
- **Command**: `torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN`
- **Training Data Mixture** (total ~848K rows):
  - **SmolTalk**: 460K rows of general conversations
  - **MMLU (auxiliary_train)**: 100K rows of multiple choice problems (ARC, MC_TEST, OBQA, RACE)
  - **GSM8K**: 8K rows teaching simple math and calculator tool use
  - **Identity conversations**: 1K rows × 2 epochs = 2K rows
  - **Simple Spelling**: 200K rows (e.g., "spell the word 'apple'")
  - **Spelling Bee**: 80K rows (e.g., "how many 'r' are in 'strawberry'?")

- **Training Configuration**:
  - **Device batch size**: 32
  - **Total batch size**: 524,288 tokens
  - **Learning rates**: Same as pretraining but with `init_lr_frac=1.0` (full LR)
  - **Training**: Runs for 1 epoch over the dataset (~848K examples)
  - **LR schedule**: Constant for first 80%, then linear decay to 0

- **What the model learns**:
  - Conversation format (user/assistant turns)
  - Special tokens (`<|user|>`, `<|assistant|>`, `<|assistant_end|>`, etc.)
  - Multiple choice question answering
  - Tool use (Python code execution for calculator)
  - Spelling tasks
  - Model personality/identity

- **Time**: ~30-45 minutes

#### Step 4.3: Evaluate Midtrained Model
- **Action**: Evaluates model on chat tasks
- **Command**: `torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid`
- **Details**:
  - Evaluates on various benchmarks:
    - **ARC-Challenge/Easy**: Science questions
    - **GSM8K**: Math problems
    - **HumanEval**: Python coding
    - **MMLU**: Broad knowledge multiple choice
    - **ChatCORE**: Conversational evaluation
  - Measures improvement over base model

---

### **PHASE 5: Supervised Fine-Tuning (SFT)** (~15-20 minutes)

This phase fine-tunes the model on high-quality conversational data.

#### Step 5.1: Train SFT
- **Action**: Fine-tunes model on curated conversation dataset
- **Command**: `torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN`
- **Training Data Mixture** (total ~23K rows):
  - **ARC-Easy**: 2.3K rows
  - **ARC-Challenge**: 1.1K rows
  - **GSM8K**: 8K rows
  - **SmolTalk**: 10K rows (subset)
  - **Identity conversations**: 1K rows
  - **Simple Spelling**: 300 rows
  - **Spelling Bee**: 300 rows

- **Training Configuration**:
  - **Device batch size**: 4 (smaller to avoid OOM with variable-length sequences)
  - **Target examples per step**: 32
  - **Gradient accumulation**: Automatically calculated
  - **Number of epochs**: 1
  - **Learning rates**: Much lower (`init_lr_frac=0.02` = 2% of base LR)
    - Unembedding LR: 0.004 × 0.02 = 0.00008
    - Embedding LR: 0.2 × 0.02 = 0.004
    - Matrix LR: 0.02 × 0.02 = 0.0004
  - **LR schedule**: Linear decay from 1.0 to 0.0 over training
  - **Evaluation**: Every 100 steps (validation loss), every 200 steps (task metrics)

- **What the model learns**:
  - Better conversation quality
  - Improved task-specific performance
  - Refined response formatting
  - Better handling of edge cases

- **Time**: ~15-20 minutes

#### Step 5.2: Evaluate SFT Model
- **Action**: Final evaluation on all tasks
- **Command**: `torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft`
- **Details**:
  - Comprehensive evaluation on all benchmarks
  - Measures final model performance
  - Should show improvement over midtrained model

---

### **PHASE 6: Report Generation** (~1 minute)

#### Step 6.1: Generate Final Report
- **Action**: Compiles all training metrics into a markdown report
- **Command**: `python -m nanochat.report generate`
- **Output**: `report.md` file with:
  - System information
  - Tokenizer statistics
  - Base model training metrics
  - Midtraining metrics
  - SFT metrics
  - Evaluation results table
  - Total training time

---

## Optional Phase: Reinforcement Learning (Commented Out)

The script includes (but doesn't run by default) a reinforcement learning phase:

```bash
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

This would further improve the model using RLHF (Reinforcement Learning from Human Feedback), but is optional and not included in the 4-hour speedrun.

---

## Training Statistics Summary

### Model Specifications
- **Architecture**: GPT Transformer
- **Layers**: 20
- **Parameters**: 561 million
- **Model dimension**: 1,280
- **Attention heads**: 10
- **Vocabulary size**: 65,536 tokens
- **Max context**: 2,048 tokens

### Training Data
- **Pretraining**: ~11.2 billion tokens (~54 billion characters)
- **Midtraining**: ~848K conversation examples
- **SFT**: ~23K high-quality examples

### Training Time Breakdown (Approximate)
- Environment setup: ~5 minutes
- Tokenizer training: ~15-20 minutes
- Base pretraining: ~2.5-3 hours (majority)
- Midtraining: ~30-45 minutes
- SFT: ~15-20 minutes
- Evaluation & reporting: ~10-15 minutes
- **Total**: ~4 hours

### Compute Resources
- **Hardware**: 8× H100 GPUs (80GB each)
- **Cost**: ~$24/hour × 4 hours = ~$96-100
- **Total FLOPs**: ~4e19 FLOPs (40 exaFLOPs)

### Expected Performance (d20 model)
Based on typical results:
- **CORE metric**: ~0.22 (base model)
- **ARC-Challenge**: ~0.28-0.29
- **ARC-Easy**: ~0.35-0.39
- **GSM8K**: ~0.02-0.05 (without RL)
- **HumanEval**: ~0.07-0.09
- **MMLU**: ~0.31-0.32
- **ChatCORE**: ~0.07-0.09

---

## Key Training Techniques

1. **Chinchilla Scaling**: Trains on 20× parameters in tokens (optimal data-to-parameter ratio)
2. **Muon Optimizer**: Fast optimizer for transformer layers
3. **AdamW Optimizer**: For embeddings and output head
4. **Gradient Accumulation**: Enables large effective batch sizes
5. **Mixed Precision**: bfloat16 for faster training
6. **Distributed Training**: 8 GPUs in parallel using PyTorch DDP
7. **Progressive Training**: Base → Mid → SFT (increasingly specialized)
8. **Task Mixtures**: Combines multiple datasets for diverse learning

---

## Output Artifacts

After training completes, you'll have:

1. **Tokenizer**: `~/.cache/nanochat/tokenizer/`
2. **Base model checkpoint**: `~/.cache/nanochat/base_checkpoints/d20/`
3. **Midtrained checkpoint**: `~/.cache/nanochat/mid_checkpoints/d20/`
4. **SFT checkpoint**: `~/.cache/nanochat/chatsft_checkpoints/d20/`
5. **Report**: `report.md` in project directory
6. **Training logs**: If using wandb, logged to wandb.ai

---

## Next Steps After Training

Once training completes, you can:

1. **Chat via CLI**:
   ```bash
   python -m scripts.chat_cli -p "Why is the sky blue?"
   ```

2. **Chat via Web UI**:
   ```bash
   python -m scripts.chat_web
   ```
   Then visit the URL shown (e.g., `http://<your-ip>:8000/`)

3. **View the report**:
   ```bash
   cat report.md
   ```

---

## Notes

- The model is intentionally small (561M params) to fit the $100 budget
- Performance is comparable to GPT-2 (2019), but far below modern LLMs
- The model will make mistakes and hallucinate - this is expected for this size
- All training is done from scratch - no pretrained weights used
- The entire pipeline is reproducible and hackable
