# Decoding the Magic Behind Andrej Karpathy's NanoChat

*A Deep Dive into Building a ChatGPT Clone from Scratch*

---

## Introduction

So I've been messing around with [nanochat](https://github.com/karpathy/nanochat) for the past few weeks. Andrej Karpathy released this thing claiming it's "The best ChatGPT that $100 can buy" and I was skeptical at first - like, really? A full ChatGPT clone for $100?

Turns out, it's actually pretty legit. The codebase is surprisingly small (like 8K lines), and it does everything - tokenization, training, fine-tuning, even a web UI. I've been trying to get it running on a single GPU (because I don't have 8 H100s lying around), and let me tell you, it's been... educational.

I've hit a bunch of roadblocks, learned a ton about how these models actually work, and figured out some stuff that might be useful to share. So here's my take on what makes nanochat tick, what I learned from trying to run it, and why it's actually pretty cool even if the model quality isn't GPT-4 level.

---

## What is NanoChat?

Okay, so what is this thing? Basically, nanochat is a complete ChatGPT clone implementation that's way smaller than you'd expect. We're talking like 8K lines of code across 45 files - that's it. No massive framework, no thousands of config options, just the essentials.

It does the whole pipeline - tokenization with a custom Rust BPE (which is pretty cool), pretraining on raw text, fine-tuning for chat, evaluation, and even a web UI so you can actually talk to it. The whole point is that you can understand the entire codebase. You can read it, modify it, break it, fix it. It's designed to train on 8 H100s for around $100-1000, which is... still expensive, but way cheaper than training GPT-4.

---

## The Architecture: A Modern Transformer

### Core Design Principles

So the architecture is basically a Transformer, but with some modern tweaks. Instead of learning where words are in a sentence, it uses rotations to figure out position. This works better for longer text.

It also normalizes some values before doing attention (QK normalization). I didn't know this was a thing, but it helps the model train better. The model also uses separate embedding matrices for input and output instead of sharing one. This helps smaller models work better.

For the activation function, it uses ReLU² (that's `relu(x)²`) instead of GELU. It's simpler and works just as well. The attention uses Group-Query Attention (GQA), which shares some parts across multiple heads. This makes it faster without hurting quality.

Oh, and all the layers don't have bias terms. This makes the model slightly smaller and more efficient. It's these little details that add up.

### Model Scaling

The cool thing is that model size is controlled by just one number: `depth`, which is how many layers the model has. Everything else figures itself out. Here's the code that does it:

```python
# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads  # default is 1:1 GQA (Group Query Attention) ratio
```

So the model size is `depth × 64`. The number of heads is figured out from that to keep things at 128, and the vocab size is always 65,536.

If you set `depth=20`, you get about 561 million parameters (they call this the d20 model). If you go up to `depth=32`, you get around 1.9 billion parameters. It's a simple way to make the model bigger or smaller depending on what you can afford.

---

## The Training Pipeline: Three Stages

Training happens in three stages, each doing something different. It's not just "train on data and you're done" - first it learns language, then it learns how to chat, then it gets better at chatting. Let me explain what each stage does.

### Stage 1: Base Training (Pretraining)

This is where you train the model on raw text to learn language. They use **FineWeb-Edu 100BT**, which is educational web text from HuggingFace. It's huge - like 455 billion characters total, split into about 1,822 files with roughly 250 million characters each.

**Dataset**: [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle)

The goal is simple: predict the next word, just like any other language model. But there are some cool details. They use something called Chinchilla scaling, which means they train on 20 times the number of parameters in tokens. So for the d20 model with 561 million parameters, that's 11.2 billion tokens.

They also use two different optimizers. Muon for the transformer layers (it's a momentum optimizer with some math tricks), and AdamW for the embeddings and output layer. I'll explain why later, but it's a smart split.

The code is made for training across 8 GPUs, but it automatically works on a single GPU too. When I tried running this on a single A100, it took forever (like 3-4 days instead of 4 hours), but the code just automatically increased the gradient accumulation to keep the same batch size. Pretty neat that it just works.

### Stage 2: Midtraining

This is where the magic happens. The base model can generate text, but it doesn't know how to have a conversation. Midtraining teaches it how to actually chat.

The training data is a mix of about 850K examples:

- **SmolTalk** (460K conversations): [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) - General conversational data
- **MMLU** (100K examples): [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) - Multiple choice questions
- **GSM8K** (8K examples): [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) - Math word problems (uses Python calculator)
- **Identity** (2K examples): [Download link](https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl) - Synthetic personality conversations
- **Spelling tasks** (280K examples): SimpleSpelling (200K) and SpellingBee (80K) - Synthetic tasks for spelling and letter counting

This stage teaches the model how to format conversations with user/assistant turns, how to use tools (like that Python calculator for math), how to handle multiple choice questions, and all those special tokens like `<|user_start|>` and `<|assistant_start|>`.

The tool use is pretty clever. When the model needs to do math, it wraps Python code in special tokens. The engine detects these, runs the code, and puts the results back. It's a simple pattern but it works.

### Stage 3: Supervised Fine-Tuning (SFT)

This is the final polish. By now the model knows how to chat, so SFT is about making conversations better. The dataset is much smaller - only about 23K examples, but they're more curated. It's a mix of:

- **ARC** (3.4K examples): [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) - Science questions (ARC-Easy: 2.3K, ARC-Challenge: 1.1K)
- **GSM8K** (8K examples): [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) - Math problems
- **SmolTalk** (10K examples): [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) - General conversations (subset)
- **Identity** (1K examples): [Download link](https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl) - Personality maintenance
- **Spelling tasks** (600 examples): SimpleSpelling (300) and SpellingBee (300) - Synthetic spelling tasks

The key difference is they use masked loss. Only the assistant's words count for training - user words are ignored. This makes sense because you want the model to learn how to respond, not how to predict what the user will say.

The smaller dataset is intentional. By this point, the model already knows how to chat - SFT is just about making it better. Quality over quantity.

---

## The Optimizers: Muon + AdamW

Okay, this is pretty cool. Nanochat uses TWO different optimizers, which I thought was weird at first but makes sense once you understand why:

### Muon Optimizer (for Transformer Layers)

Muon is a momentum optimizer with some math tricks. It does a standard update, but then does some processing to make it more stable. I'm not going to pretend I fully understand the math, but the idea is that it helps the model train better.

Here's the core code from `nanochat/muon.py` that shows how it works:

```python
@torch.no_grad()
def step(self):
    for group in self.param_groups:
        params = group["params"]
        for p in params:
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            # Standard SGD-momentum update
            buf.lerp_(g, 1 - group["momentum"])
            g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
            # Orthogonalize the update via Newton-Schulz iteration
            g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
            # Apply the update
            p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
```

It's more efficient for large operations and runs in bfloat16, which saves memory. The transformer layers (attention and MLP) use this optimizer.

### AdamW Optimizer (for Embeddings + LM Head)

The embeddings and output layer use standard AdamW instead. This makes sense because embedding layers update differently (most words don't appear in every batch), and the output layer needs different handling. AdamW works better for these.

So the split is: transformer layers get Muon, embeddings and the output layer get AdamW. It's a smart approach - using the right optimizer for the right parts. I wouldn't have thought of this, but it works.

---

## The Tokenizer: Custom Rust BPE

The tokenizer is interesting - they use a custom Rust implementation for training, then switch to tiktoken for inference. Why Rust? Because it's way faster for training, which matters when you're processing huge datasets. Same algorithm, just faster.

It's GPT-4 style, which means byte-level tokenization, BPE merging, and special tokens for conversation format and tool use. Here's the special tokens definition from `nanochat/tokenizer.py`:

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # Beginning of sequence (document delimiter)
    "<|user_start|>",       # User messages
    "<|user_end|>",
    "<|assistant_start|>",  # Assistant messages
    "<|assistant_end|>",
    "<|python_start|>",     # Python tool calls
    "<|python_end|>",
    "<|output_start|>",     # Tool output
    "<|output_end|>",
]
```

And here's how they train with Rust BPE and then create a tiktoken encoding for inference:

```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1) Train using rustbpe (fast Rust implementation)
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
    
    # 2) Construct tiktoken encoding for efficient inference
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return cls(enc, "<|bos|>")
```

It's a lot of special tokens, but they make the conversation format clear.

---

## The Inference Engine: Efficient Generation

The inference engine has some nice optimizations. It uses a KV cache, which stores some values so you don't have to recompute them for previous words. The cache grows as needed, which is efficient.

For batch generation, it does a single pass, then clones the cache for multiple samples. This lets you generate multiple responses at the same time.

The tool use is straightforward - it detects those `<|python_start|>` tokens, runs the Python code, and puts the results back. And it has streaming support, giving you tokens one at a time so you can build real-time chat interfaces.

---

## Running on Single GPU: My Experience

### The Challenge

The original `speedrun.sh` is designed for **8xH100 GPUs** (640GB total VRAM). That's a massive setup - each H100 has 80GB of memory, and with 8 of them working together, you can train a 561M parameter model (d20) with a batch size of 32 and sequence length of 2048 in just 4 hours.

I wanted to run it on a **single A100 GPU** (40GB or 80GB). That's a huge difference - instead of 640GB total VRAM, I'd have just 40-80GB. Instead of 8 GPUs working in parallel, I'd have one GPU doing everything sequentially. The math is simple: 8 GPUs can process 8 batches at once, so a single GPU needs to do 8x the work, which means 8x the time (or more, since there's overhead).

But here's the thing - I don't have access to 8 H100s. Most people don't. So I wanted to see if I could actually make this work on a single GPU, even if it meant waiting days instead of hours. The question was: would the code even work, or would it crash immediately?

### What I Learned

The first thing I found is that the code already supports single GPU! The scripts automatically detect if you're using multiple GPUs. If you use `torchrun`, it runs in multi-GPU mode. If you just run the Python script directly without `torchrun`, it automatically switches to single GPU mode and increases gradient accumulation to keep the same batch size. So you can just remove `torchrun` and run `python -m scripts.base_train --depth=20` and it works.

But there are memory constraints. For a single 40GB A100, the d20 model with 561M parameters is too large - you need to drop down to d12-d14. You also need to reduce the batch size from 32 to maybe 4-8, and shorten sequences from 2048 to 1024-1536. It still works, just smaller.

The time reality check is... well, it's not great. On 8 H100s it takes 4 hours and costs about $96. On a single H100, you're looking at 1.5 days and $81-148. 

Here's the catch: An 80GB A100 can train the full d20 model (561M params), which takes 3.5-4.5 days ($81-218). A 40GB A100 can only fit a smaller d12-d14 model, which takes 1-2 days ($27-82). So the 80GB is actually more powerful, but it's doing more work (bigger model), so it takes longer. The 40GB is faster because it's training a smaller model.

Single GPU is 20-25x slower than 8 GPUs, but hey, it works!

### The PyTorch CUDA Issue

One frustrating issue I encountered: The `pyproject.toml` specifies CUDA 12.8, which requires `libnvshmem_host.so.3` that's not available on all systems (like Colab's L4 GPU).

**Solution**: Modify `pyproject.toml` to use CUDA 11.8 instead:
```bash
sed -i 's/pytorch-cu128/pytorch-cu118/g' pyproject.toml
sed -i 's/cu128/cu118/g' pyproject.toml
```

This ensures `uv sync` installs a compatible PyTorch version from the start.

---

## What NanoChat Can (and Can't) Do

So what can this thing actually do? It's pretty good at general conversation - natural dialogue, following instructions, keeping context. For reasoning tasks, it gets 28-39% on ARC science questions, 31% on MMLU multiple choice, and 2.5-7.6% on GSM8K math (which is basic, but it can use the Python calculator). It can run simple Python expressions through that calculator.

But it's not good at coding. The HumanEval score is only 6.7-8.5%, which is very low. It's not trained on coding datasets, so while it can generate some Python, the quality is poor. It also struggles with complex reasoning, long content, and facts - it makes stuff up a lot.

The reality is that NanoChat is like "talking to a kindergartener" - it's amusing, makes mistakes, and makes stuff up, but it's yours - you can change it however you want. That's the trade-off.

---

## Key Insights and Learnings

The biggest thing I learned is that simplicity is a feature. NanoChat proves you don't need a huge framework to build a modern LLM. The entire codebase is only about 8K lines, yet it includes a custom tokenizer, distributed training, multiple optimizers, tool use, a web UI, and evaluation. Sometimes less is more - a simple, readable codebase is way more valuable than a "powerful" but complex framework.

The three-stage pipeline (base → mid → SFT) is elegant. Base learns language, mid teaches conversation format and tools, and SFT makes it better. Each stage builds on the previous one, and keeping them separate makes the code easier to understand and change.

The modern optimizations actually matter. RoPE instead of learned positional embeddings, QK normalization, the Muon optimizer, GQA for inference - these aren't just academic. They make the model more efficient and easier to train.

Tool use is simpler than I thought. The Python calculator integration is straightforward - special tokens mark code blocks, the engine detects and runs them, and results get put back into the conversation. This pattern could easily be extended to other tools like web search or database queries.

And single GPU is possible, just slow. You don't need 8 GPUs to train nanochat. A single A100 works fine if you make the model smaller, cut the batch size, increase gradient accumulation, and have patience (3-4 days instead of 4 hours).

---

## The Codebase Structure

What I love about nanochat is how everything is organized:

```
nanochat/
├── gpt.py              # The Transformer model (clean, readable)
├── tokenizer.py        # BPE tokenizer wrapper
├── engine.py           # Efficient inference engine
├── dataloader.py       # Distributed data loading
├── muon.py             # Muon optimizer
├── adamw.py            # Distributed AdamW
└── ...

scripts/
├── base_train.py       # Pretraining
├── mid_train.py       # Midtraining
├── chat_sft.py        # Supervised fine-tuning
├── chat_eval.py       # Evaluation
└── chat_web.py        # Web UI

tasks/
├── arc.py             # Science questions
├── gsm8k.py           # Math problems
├── humaneval.py       # Coding benchmark
└── ...
```

Each file has a clear purpose, and the code is well-commented. You can actually **read and understand** the entire codebase in a reasonable amount of time.

---

## Performance Benchmarks

From the README, here are the actual results for a d20 model:

| Metric | MID | SFT |
|--------|-----|-----|
| ARC-Challenge | 28.75% | 28.07% |
| ARC-Easy | 35.61% | 38.76% |
| GSM8K | 2.50% | 4.55% |
| HumanEval | 6.71% | 8.54% |
| MMLU | 31.11% | 31.51% |

**Context**: 
- GPT-2 (2019): ~20% on CORE metric
- NanoChat d20: ~22% on CORE metric (slightly better!)
- Modern models (GPT-4, etc.): Much higher

**The Point**: For a $100 model, these results are impressive. It's not GPT-4, but it's a fully functional ChatGPT clone that you can train yourself.

---

## What Makes NanoChat Special

Most LLM repos do one thing - pretraining OR fine-tuning OR evaluation. NanoChat does all of it. Tokenizer training, pretraining, fine-tuning, evaluation, even a web UI. You can go from raw text to a working ChatGPT clone in one script. That's pretty wild.

The dependencies are pretty minimal - just PyTorch, HuggingFace datasets, FastAPI for the web UI, tiktoken, and wandb (optional). No huge frameworks, no weird abstractions. Just what you need.

This is honestly the best codebase I've seen for learning. It has all the modern techniques (RoPE, QK norm, Muon optimizer), but the code is clean and readable. You can actually understand what's happening.

And it's super hackable. Want to add a new task? Just drop a file in `tasks/`. Want to mess with the optimizer? Edit `muon.py` or `adamw.py`. Everything is right there, no hidden magic.

---

## Challenges I Faced

### 1. CUDA Version Mismatches

Ugh, this was annoying. The `pyproject.toml` file specifies CUDA 12.8, which needs some library (`libnvshmem_host.so.3`) that doesn't exist on Colab's L4 GPUs. I kept getting import errors and it took me way too long to figure out I needed to change the CUDA version in `pyproject.toml` BEFORE running `uv sync`. Classic case of fixing the symptom instead of the cause.

### 2. Memory Constraints

My single 40GB A100 couldn't handle the default settings. I had to drop the model size down to d12 instead of d20, cut the batch size way down to 4 instead of 32, and shorten sequences to 1024 instead of 2048. It still works, just slower and smaller.

### 3. Training Time

Yeah, so full training on a single GPU takes days. Like, multiple days. I ended up using limited iterations just to test things, which works but the model quality suffers. You get what you pay for, I guess.

### 4. Dataset Sizes

The datasets are huge. Base training alone is like 240 shards (~24GB). I used smaller subsets for testing, but if you want the real deal, you need the full datasets.

---

## What I Would Do Differently

If I could find pre-trained base models, I'd skip straight to midtraining/SFT. Base training takes forever and honestly, I don't need to train from scratch just to learn how it works.

Even 2-4 GPUs would make a huge difference. Single GPU works, but man is it slow. If you have access to multiple GPUs, use them.

I tried to run everything at once and got overwhelmed. Should've gotten base training working first, then moved on. Baby steps.

I should've been watching `nvidia-smi` more. If you're not using most of your GPU memory, you can probably increase batch size. If you're hitting OOM errors, decrease it. Simple, but easy to forget.

---

## The Bigger Picture

Look, I'm not saying nanochat is going to change AI or anything. But it does show that you don't need millions of dollars to train an LLM. With a single GPU (or cloud access), a few hundred bucks, and a lot of patience, you can actually do this.

But honestly, the real value is learning. By actually reading the code, running it, breaking it, and fixing it, you learn how these models actually work (not just theory), what the training pipeline looks like in practice, why different optimizers matter, how tool use is actually implemented, and how evaluation works. That stuff is way more valuable than just reading papers. At least for me, anyway.

---

## Conclusion

So yeah, nanochat is pretty cool. It's not going to beat GPT-4, but that's not really the point. The point is:

1. **It actually works** - You can train a ChatGPT clone that functions
2. **You can understand it** - The code is readable, not some massive framework
3. **It's (relatively) affordable** - $100-1000 is still a lot, but way better than millions
4. **You learn a ton** - I've learned more from running this than reading papers

I've spent way too many hours on this, hit a bunch of frustrating bugs (looking at you, CUDA version conflicts), but honestly? It's been worth it. There's something really satisfying about understanding how these models work by actually running the code yourself.

If you're curious about LLMs, definitely check it out. Clone it, break it, fix it, modify it. That's the whole point - it's meant to be hacked on.

---

## Resources

- **Repository**: [https://github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
- **Live Demo**: [nanochat.karpathy.ai](https://nanochat.karpathy.ai/)
- **Discussions**: Check the GitHub Discussions for guides and tips

---

Anyway, that's my experience with nanochat so far. If you've tried it, let me know what you think. Or if you're thinking about trying it and have questions, feel free to ask. I'm definitely not an expert, but I've made enough mistakes that I might be able to help you avoid some of them.
