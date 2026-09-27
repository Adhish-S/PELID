"""
Configuration — loads settings from .env file or environment variables.

All thresholds and URLs are defined here so they're easy to find and change.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file (if it exists) into environment
load_dotenv()


# --- Path B (Frontier LLM) settings ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
UPSTREAM_BASE_URL: str = os.getenv("UPSTREAM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
UPSTREAM_MODEL: str = os.getenv("UPSTREAM_MODEL", "gemini-2.0-flash")

# --- Decision thresholds ---
# If Laya's confidence is >= this value, use the local answer (Path A)
# If below, forward to the frontier LLM (Path B)
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# If the input has more tokens than this, skip Tier 0 entirely
# Laya's context window is only 512 tokens (English) / 1024 (multilingual)
CONTEXT_BUDGET_TOKENS: int = int(os.getenv("CONTEXT_BUDGET_TOKENS", "450"))

# --- Server settings ---
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8080"))

# --- Destructive action keywords ---
# Any detected intent matching these ALWAYS goes to Path B, regardless of confidence
DESTRUCTIVE_KEYWORDS: list[str] = [
    "delete",
    "remove",
    "cancel",
    "refund",
    "terminate",
    "close_account",
    "revoke",
    "purge",
]
