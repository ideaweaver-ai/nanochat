"""
Setup script for Qwen tokenizer.

This script:
1. Loads Qwen tokenizer from HuggingFace
2. Adds nanochat's special tokens
3. Computes token_bytes.pt (needed for bits-per-byte evaluation)
4. Saves tokenizer configuration
"""

import os
import torch
from nanochat.qwen_tokenizer import QwenTokenizer
from nanochat.common import get_base_dir

def main():
    print("=" * 70)
    print("Setting up Qwen tokenizer for nanochat")
    print("=" * 70)
    
    # Load Qwen tokenizer
    print("\n1. Loading Qwen tokenizer from HuggingFace...")
    tokenizer = QwenTokenizer.from_pretrained("Qwen/Qwen2.5-7B", add_special_tokens=True)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Vocab size: {vocab_size:,}")
    
    # Get base directory
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Save tokenizer
    print("\n2. Saving tokenizer...")
    tokenizer.save(tokenizer_dir)
    
    # Compute token_bytes.pt (needed for bits-per-byte evaluation)
    print("\n3. Computing token_bytes.pt...")
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    
    token_bytes = []
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            if token_str in special_set:
                token_bytes.append(0)  # Special tokens are not counted
            else:
                id_bytes = len(token_str.encode("utf-8"))
                token_bytes.append(id_bytes)
        except Exception as e:
            # Handle any decoding errors
            token_bytes.append(0)
            if token_id < 10:  # Only print first few errors
                print(f"   Warning: Could not decode token {token_id}: {e}")
    
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    torch.save(token_bytes, token_bytes_path)
    print(f"   Saved token_bytes to {token_bytes_path}")
    
    # Print statistics
    token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    print(f"\n4. Token bytes statistics:")
    print(f"   Min: {int(token_bytes_nonzero.min().item())}")
    print(f"   Max: {int(token_bytes_nonzero.max().item())}")
    print(f"   Mean: {token_bytes_nonzero.mean().item():.2f}")
    print(f"   Std: {token_bytes_nonzero.std().item():.2f}")
    
    # Test tokenizer
    print("\n5. Testing tokenizer...")
    test_text = "Hello world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text, f"Decode mismatch: '{decoded}' != '{test_text}'"
    print(f"   Test passed: '{test_text}' -> {len(encoded)} tokens -> '{decoded}'")
    
    # Test special tokens
    print("\n6. Testing special tokens...")
    bos_id = tokenizer.get_bos_token_id()
    print(f"   BOS token ID: {bos_id}")
    for special in ["<|bos|>", "<|user_start|>", "<|assistant_start|>"]:
        try:
            token_id = tokenizer.encode_special(special)
            print(f"   {special}: {token_id}")
        except Exception as e:
            print(f"   {special}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print("Qwen tokenizer setup complete!")
    print("=" * 70)
    print(f"\nTokenizer saved to: {tokenizer_dir}")
    print(f"Vocab size: {vocab_size:,}")
    print("\nNext steps:")
    print("1. Modify nanochat/tokenizer.py to use QwenTokenizer")
    print("2. Skip tokenizer training in speedrun.sh")
    print("3. Run training with new vocab_size")

if __name__ == "__main__":
    main()
