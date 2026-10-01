"""
Context-Budget Check (Stage 5b from the build plan).

If the decision's context/state object exceeds ~400-500 tokens,
skip Tier 0 entirely and route straight to Path B.

Why? Laya's context window is only 512 tokens (English) / 1024 (multilingual).
Trying to chunk long inputs into sequential calls would add latency,
defeating the purpose of using Laya in the first place.
"""

import tiktoken

from pelid.config import CONTEXT_BUDGET_TOKENS

# TODO: Implement in Stage 5
# Use tiktoken to count tokens in the relevant parts of the request


try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tokenizer = None


def count_tokens(text: str) -> int:
    """Count tokens in input text."""
    if not text:
        return 0
    if _tokenizer:
        return len(_tokenizer.encode(text))
    # Approximation: ~4 chars per token
    return max(1, len(text) // 4)


def is_within_budget(text: str) -> bool:
    """
    Check if the input text is short enough for Laya to handle.

    Args:
        text: The text content to check (extracted from the request).

    Returns:
        True if within budget (can go to Tier 0 / Laya).
        False if too long (must go to Path B).
    """
    if not text:
        return True
    return count_tokens(text) <= CONTEXT_BUDGET_TOKENS
