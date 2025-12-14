"""
Qwen3 Tokenizer wrapper for nanochat compatibility.

This module provides a QwenTokenizer class that wraps Qwen's HuggingFace tokenizer
to match nanochat's tokenizer API, allowing seamless replacement.
"""

import os
import copy
from functools import lru_cache
from transformers import AutoTokenizer

# nanochat special tokens (must match nanochat/tokenizer.py)
NANOCHAT_SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


class QwenTokenizer:
    """
    Wrapper around Qwen tokenizer to match nanochat's tokenizer API.
    
    This class provides the same interface as RustBPETokenizer but uses
    Qwen's pre-trained tokenizer from HuggingFace.
    """
    
    def __init__(self, hf_tokenizer, add_special_tokens=True):
        """
        Initialize Qwen tokenizer wrapper.
        
        Args:
            hf_tokenizer: HuggingFace AutoTokenizer instance
            add_special_tokens: Whether to add nanochat's special tokens
        """
        self.tokenizer = hf_tokenizer
        
        if add_special_tokens:
            # Add nanochat's special tokens to Qwen tokenizer
            # Check which tokens already exist
            existing_tokens = set(self.tokenizer.get_added_vocab().keys())
            tokens_to_add = [tok for tok in NANOCHAT_SPECIAL_TOKENS if tok not in existing_tokens]
            
            if tokens_to_add:
                self.tokenizer.add_special_tokens({
                    "additional_special_tokens": tokens_to_add
                })
                print(f"Added {len(tokens_to_add)} special tokens to Qwen tokenizer")
        
        # Map BOS token: Qwen uses <|endoftext|>, nanochat uses <|bos|>
        # We'll use <|bos|> if it exists, otherwise fall back to <|endoftext|>
        if "<|bos|>" in self.tokenizer.get_vocab():
            self._bos_token = "<|bos|>"
        elif "<|endoftext|>" in self.tokenizer.get_vocab():
            self._bos_token = "<|endoftext|>"
        else:
            # Use the first special token as BOS
            special_tokens = self.tokenizer.get_added_vocab()
            if special_tokens:
                self._bos_token = list(special_tokens.keys())[0]
            else:
                raise ValueError("No BOS token found in tokenizer")
    
    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen2.5-7B", add_special_tokens=True):
        """
        Load Qwen tokenizer from HuggingFace.
        
        Args:
            model_name: HuggingFace model name (default: Qwen/Qwen2.5-7B)
            add_special_tokens: Whether to add nanochat's special tokens
            
        Returns:
            QwenTokenizer instance
        """
        print(f"Loading Qwen tokenizer from {model_name}...")
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return cls(hf_tokenizer, add_special_tokens=add_special_tokens)
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_special_tokens(self):
        """Get set of special tokens."""
        added_vocab = self.tokenizer.get_added_vocab()
        return set(added_vocab.keys())
    
    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """
        Encode a single special token.
        
        Args:
            text: Special token string (e.g., "<|bos|>")
            
        Returns:
            Token ID
        """
        # Use convert_tokens_to_ids for special tokens
        token_id = self.tokenizer.convert_tokens_to_ids(text)
        if token_id == self.tokenizer.unk_token_id:
            # Try encoding as regular text
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                return encoded[0]
            raise ValueError(f"Special token '{text}' not found in tokenizer")
        return token_id
    
    def get_bos_token_id(self):
        """Get the BOS (beginning of sequence) token ID."""
        return self.encode_special(self._bos_token)
    
    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text (string or list of strings)
            prepend: Token or token ID to prepend (optional)
            append: Token or token ID to append (optional)
            num_threads: Number of threads (ignored, kept for API compatibility)
            
        Returns:
            List of token IDs (or list of lists if text is a list)
        """
        if isinstance(text, str):
            # Encode single string
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Handle prepend/append
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
                ids.insert(0, prepend_id)
            if append is not None:
                append_id = append if isinstance(append, int) else self.encode_special(append)
                ids.append(append_id)
            
            return ids
        elif isinstance(text, list):
            # Encode batch of strings
            ids_batch = self.tokenizer.batch_encode_plus(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None
            )["input_ids"]
            
            # Handle prepend/append
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
                for ids in ids_batch:
                    ids.insert(0, prepend_id)
            if append is not None:
                append_id = append if isinstance(append, int) else self.encode_special(append)
                for ids in ids_batch:
                    ids.append(append_id)
            
            return ids_batch
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
    
    def __call__(self, *args, **kwargs):
        """Make tokenizer callable."""
        return self.encode(*args, **kwargs)
    
    def decode(self, ids, skip_special_tokens=False):
        """
        Decode token IDs to text.
        
        Args:
            ids: Token IDs (list or list of lists)
            skip_special_tokens: Whether to skip special tokens (default: False for nanochat compatibility)
            
        Returns:
            Decoded text (string or list of strings)
        """
        if isinstance(ids[0], list):
            # Batch decode
            return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)
        else:
            # Single decode
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def id_to_token(self, id):
        """Convert token ID to token string."""
        return self.tokenizer.convert_ids_to_tokens([id])[0]
    
    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (nanochat-style).
        
        This method matches the API of RustBPETokenizer.render_conversation().
        
        Args:
            conversation: Dictionary with "messages" key
            max_tokens: Maximum tokens to return
            
        Returns:
            ids: List of token IDs
            mask: List of mask values (1 for assistant tokens, 0 for others)
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))
        
        # Handle system message (merge with first user message)
        messages = conversation["messages"]
        if messages and messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"
        
        # Fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")
        
        # Now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            # Sanity checking
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
            
            content = message["content"]
            
            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # Simple string
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)
        
        # Truncate to max_tokens
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
    
    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. Render conversation priming the Assistant for completion.
        
        Args:
            conversation: Dictionary with "messages" key
            
        Returns:
            List of token IDs
        """
        # Pop the last message (assistant's message)
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()
        
        # Tokenize the conversation
        ids, mask = self.render_conversation(conversation)
        
        # Append assistant start token to prime for completion
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids
    
    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """
        Visualize tokenization (for debugging).
        
        Args:
            ids: List of token IDs
            mask: List of mask values
            with_token_id: Whether to show token IDs
            
        Returns:
            Colored string representation
        """
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)
    
    def save(self, tokenizer_dir):
        """
        Save tokenizer to disk (for compatibility).
        
        Args:
            tokenizer_dir: Directory to save to
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_dir)
        print(f"Saved Qwen tokenizer to {tokenizer_dir}")
