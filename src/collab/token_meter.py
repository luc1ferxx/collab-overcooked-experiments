# /workspace/src/collab/token_meter.py
import math
import os

def _tiktoken_count(text: str, model_hint: str) -> int:
    try:
        import tiktoken
        enc = None
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except Exception:
            # Fall back to cl100k_base for OpenAI-like models
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return -1

def _hf_count(text: str, model_hint: str) -> int:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_hint, use_fast=True)
        return len(tok.encode(text or "", add_special_tokens=True))
    except Exception:
        return -1

def _rough_count(text: str) -> int:
    # Reasonable fallback: ~4 chars/token heuristic
    n_chars = len(text or "")
    return max(1, math.ceil(n_chars / 4.0))

def count_tokens(text: str, model_hint: str = "") -> int:
    """
    Try: tiktoken -> HF tokenizer -> rough heuristic.
    model_hint can be "gpt-3.5-turbo-0125", "llama-3-8b-instruct", etc.
    """
    # Try OpenAI-style first
    n = _tiktoken_count(text, model_hint)
    if n > 0:
        return n

    # If using a HF model under vLLM, the HF tokenizer may be present:
    n = _hf_count(text, model_hint)
    if n > 0:
        return n

    # Fallback
    return _rough_count(text)
