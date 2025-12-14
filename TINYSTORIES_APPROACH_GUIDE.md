# Building a TinyStories-Style Model with nanochat

## Can We Build It? **YES!** ✅

You can absolutely build a TinyStories-style model with nanochat. Here's how and what to expect.

---

## What is TinyStories?

TinyStories is a synthetic dataset designed for training very small language models:
- **Vocabulary**: ~1,500 simple words (vs 65,536 for nanochat)
- **Content**: Short stories for 3-4 year olds
- **Complexity**: Simple grammar, basic concepts
- **Result**: Models with <10M parameters can generate coherent stories

**Key Insight**: By restricting vocabulary and domain, small models can work!

---

## How to Build It with nanochat

### Step 1: Create a Simplified Dataset

You need to create or find a dataset with:
- **Limited vocabulary**: ~1,000-2,000 words
- **Simple domain**: Children's stories, simple conversations, basic instructions
- **Short texts**: Each example should be 50-200 words

**Options:**

1. **Use TinyStories dataset** (if available):
   ```python
   # Download TinyStories dataset
   # Format: One story per line
   ```

2. **Generate synthetic data** (like TinyStories paper):
   ```python
   # Use GPT-4/Claude to generate simple stories
   # With vocabulary restrictions
   # Following simple templates
   ```

3. **Filter existing data**:
   ```python
   # Take a subset of your data
   # Filter to only simple sentences
   # Remove complex vocabulary
   ```

### Step 2: Train a Small Vocabulary Tokenizer

The nanochat tokenizer supports custom vocabulary size! Modify the training:

```bash
# Instead of default 65,536 vocab size, use 1,000-2,000
python -m scripts.tok_train \
    --vocab_size=2000 \
    --max_chars=1000000000  # Train on your simplified dataset
```

**Key changes:**
- `--vocab_size=2000` (vs default 65536)
- Train on your simplified dataset (not general web text)
- This will learn subword patterns specific to your simple vocabulary

### Step 3: Modify Training Script

Create a modified `speedrun_tinystories.sh`:

```bash
#!/bin/bash

# TinyStories-style training with nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup (same as before)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Initialize report
python -m nanochat.report reset

# Step 1: Prepare simplified dataset
# (You'll need to create this - see dataset preparation below)

# Step 2: Train small vocabulary tokenizer
python -m scripts.tok_train \
    --vocab_size=2000 \
    --max_chars=500000000  # 500M chars of simplified text

# Step 3: Train small model (depth=3 for ~6M params)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=3 \
    --run=tinystories

# Step 4: Fine-tune on conversation format (optional)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
    --run=tinystories

# Generate report
python -m nanochat.report generate
```

### Step 4: Dataset Preparation

You need to modify `nanochat/dataset.py` or create a custom dataset loader:

```python
# Example: Simple dataset iterator for TinyStories
def tinystories_iterator():
    """
    Yields simple stories with limited vocabulary
    """
    # Load your simplified dataset
    with open('tinystories_dataset.txt', 'r') as f:
        for line in f:
            story = line.strip()
            # Filter: only keep stories with <200 words
            if len(story.split()) < 200:
                yield story
```

**Or modify the existing dataset loader** to filter/preprocess:
- Remove complex words
- Simplify sentences
- Keep only short texts

---

## What You'll Get

### Expected Performance

With a TinyStories approach (5M params, 2K vocab):

**What it CAN do:**
- ✅ Generate coherent simple stories
- ✅ Follow basic story patterns
- ✅ Use correct grammar (mostly)
- ✅ Maintain simple character consistency
- ✅ Generate text in the simplified domain

**What it CANNOT do:**
- ❌ Handle complex vocabulary
- ❌ Understand nuanced concepts
- ❌ Reason about complex topics
- ❌ Work outside the simplified domain
- ❌ General-purpose conversation

### Comparison

| Aspect | Standard nanochat (561M, 65K vocab) | TinyStories-style (5M, 2K vocab) |
|--------|-------------------------------------|----------------------------------|
| **Vocabulary** | 65,536 tokens | 2,000 tokens |
| **Domain** | General web text | Simple stories only |
| **Coherence** | Limited (kindergartener) | Good (for simple stories) |
| **General use** | Poor but works | Doesn't work |
| **Specific domain** | Poor | Good |
| **Training time** | 4 hours | ~1 hour |
| **Memory** | 80GB GPUs | 8-16GB GPUs |

---

## Drawbacks and Limitations

### 1. **Extremely Limited Vocabulary**

**Problem**: 2,000 tokens vs 65,536 tokens
- Cannot express complex concepts
- Many words must be broken into subwords
- Limited expressiveness

**Impact**: 
- Model can only work in very simple domains
- Cannot handle technical terms
- Cannot understand nuanced language

**Example**:
- ✅ "The cat sat on the mat" - Works
- ❌ "The feline positioned itself on the textile surface" - Fails
- ❌ "The quantum entanglement experiment" - Fails

### 2. **Domain Restriction**

**Problem**: Model only works in the training domain
- Trained on simple stories → only generates simple stories
- Cannot generalize to other domains
- Cannot handle different text types

**Impact**:
- Not useful for general-purpose tasks
- Cannot answer questions
- Cannot write code
- Cannot do math
- Only works for the specific domain you trained on

**Example**:
- ✅ "Once upon a time, there was a cat..." - Works
- ❌ "Write a Python function to sort a list" - Fails
- ❌ "What is the capital of France?" - Fails

### 3. **Limited Reasoning Capacity**

**Problem**: Even with simplified vocabulary, 5M parameters is still very small
- Cannot handle multi-step reasoning
- Cannot maintain long context
- Cannot understand complex relationships

**Impact**:
- Stories are simple and repetitive
- Cannot follow complex plot lines
- Limited character development
- Basic grammar only

### 4. **Data Requirements**

**Problem**: Need high-quality simplified dataset
- Must create/find appropriate data
- Need to ensure vocabulary restrictions
- Quality matters more than quantity

**Impact**:
- More work to prepare data
- Harder to find suitable datasets
- May need to generate synthetic data

### 5. **Not Transferable**

**Problem**: Model trained on simple stories cannot be used for other tasks
- Cannot fine-tune for different domains easily
- Vocabulary mismatch
- Domain mismatch

**Impact**:
- One model = one narrow use case
- Cannot reuse for other applications
- Limited practical value

### 6. **Still Limited Even in Domain**

**Problem**: Even in the simplified domain, model has limitations
- May still make grammatical errors
- Stories can be repetitive
- Limited creativity
- May hallucinate

**Impact**:
- Not production-ready even for simple tasks
- Requires careful prompting
- May need post-processing

---

## When Is This Approach Worth It?

### ✅ Good Use Cases:

1. **Educational Purposes**
   - Teaching how small models work
   - Demonstrating vocabulary impact
   - Learning about tokenization

2. **Research**
   - Studying model scaling
   - Understanding capacity limits
   - Testing hypotheses

3. **Very Specific Applications**
   - Simple story generation for children
   - Basic template filling
   - Pattern matching in narrow domain

4. **Resource Constraints**
   - Limited compute budget
   - Need to run on small devices
   - Quick prototyping

### ❌ Not Good For:

1. **General-Purpose Applications**
   - Chatbots
   - Question answering
   - Code generation
   - Any broad use case

2. **Production Systems**
   - Too limited
   - Too error-prone
   - Not reliable

3. **Complex Tasks**
   - Reasoning
   - Multi-step problems
   - Technical domains

---

## Comparison: Standard vs TinyStories Approach

### Standard nanochat (561M, 65K vocab)

**Pros:**
- Can handle general text
- Works for multiple domains
- More flexible
- Better for learning general patterns

**Cons:**
- Still limited (kindergartener level)
- Makes many mistakes
- Requires more compute
- Longer training time

### TinyStories-style (5M, 2K vocab)

**Pros:**
- Works well in narrow domain
- Faster training
- Less compute needed
- Can run on smaller GPUs
- Good for specific use cases

**Cons:**
- Extremely limited vocabulary
- Only works in one domain
- Not general-purpose
- Requires specialized dataset
- Less transferable

---

## Practical Implementation Steps

### 1. Create Simplified Dataset

```python
# tinystories_dataset.py
def create_simplified_dataset():
    """
    Create a dataset with limited vocabulary
    """
    # Option A: Use existing TinyStories dataset
    # Option B: Generate with GPT-4/Claude
    # Option C: Filter existing dataset
    
    simple_words = [
        "the", "cat", "dog", "ran", "sat", "mat", 
        "big", "small", "happy", "sad", "went", "came",
        # ... ~1,500-2,000 simple words
    ]
    
    # Generate or filter stories using only these words
    stories = []
    for _ in range(100000):  # Generate 100K stories
        story = generate_simple_story(simple_words)
        stories.append(story)
    
    return stories
```

### 2. Modify Tokenizer Training

```bash
# Train tokenizer on simplified dataset
python -m scripts.tok_train \
    --vocab_size=2000 \
    --max_chars=500000000
```

### 3. Train Small Model

```bash
# Use depth=3 for ~6M parameters
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=3 \
    --run=tinystories
```

### 4. Evaluate

```python
# Test on simple story generation
prompts = [
    "Once upon a time, there was a",
    "The cat sat on the",
    "A little girl went to the",
]

for prompt in prompts:
    completion = model.generate(prompt)
    print(f"{prompt} {completion}")
```

---

## Expected Results

### What Works:
- ✅ Simple story generation
- ✅ Basic grammar
- ✅ Coherent short texts
- ✅ Pattern following

### What Doesn't:
- ❌ Complex vocabulary
- ❌ General knowledge
- ❌ Reasoning
- ❌ Other domains

### Performance Metrics (Estimated):

| Metric | Standard (561M) | TinyStories (5M) |
|--------|----------------|------------------|
| **General coherence** | 22% (CORE) | N/A (domain-specific) |
| **Simple story quality** | Poor | Good |
| **Vocabulary coverage** | 65K | 2K |
| **Domain flexibility** | Limited | None |
| **Training time** | 4 hours | 1 hour |

---

## The Bottom Line

**Can you build it?** ✅ **Yes, absolutely!**

**Should you build it?** It depends:

- ✅ **Yes, if:**
  - You want to learn about small models
  - You have a very specific narrow use case
  - You're doing research
  - You have limited compute

- ❌ **No, if:**
  - You need general-purpose capabilities
  - You want production-ready system
  - You need flexibility
  - You want to handle multiple domains

**The trade-off is clear:**
- TinyStories approach = Works well in narrow domain, but completely useless outside it
- Standard approach = Works poorly everywhere, but at least works everywhere

**My recommendation:**
- For learning/research: ✅ Go for it!
- For practical use: ❌ Stick with larger models (100M+) or accept the limitations of the standard approach

The TinyStories approach is a clever way to make small models work, but it comes with severe limitations that make it unsuitable for most practical applications.

