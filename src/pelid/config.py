"""
Configuration — loads settings from .env file or environment variables.

All thresholds and URLs are defined here so they're easy to find and change.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Load variables from project root .env file into environment
load_dotenv(PROJECT_ROOT / ".env", override=True)


# --- Path B (Frontier LLM) settings ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
UPSTREAM_BASE_URL: str = os.getenv("UPSTREAM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
UPSTREAM_MODEL: str = os.getenv("UPSTREAM_MODEL", "gemini-3.5-flash-lite")

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
# Any detected intent or query matching these ALWAYS goes to Path B, regardless of confidence
DESTRUCTIVE_KEYWORDS: list[str] = [
    "delete",
    "remove",
    "cancel",
    "refund",
    "terminate",
    "close_account",
    "revoke",
    "purge",
    "billing_dispute",
    "shipping_address_change",
    "auto-renewal",
    "auto_renewal",
    "auto renewal",
    "auto-debit",
    "auto_debit",
    "auto-pay",
    "autopay",
    "unsubscribe",
    "discontinue",
]

# --- Security & Production Hardening ---
# Optional Bearer token to protect this gateway from unauthorized access
PELID_PROXY_API_KEY: str = os.getenv("PELID_PROXY_API_KEY", "")

# Block or escalate adversarial prompt injections and jailbreaks
ENABLE_INJECTION_SHIELD: bool = os.getenv("ENABLE_INJECTION_SHIELD", "true").lower() == "true"

# Maximum raw payload size in bytes (prevents memory exhaustion DoS)
MAX_PAYLOAD_BYTES: int = int(os.getenv("MAX_PAYLOAD_BYTES", "65536"))

# Rate limiting (requests per minute per client IP, 0 = disabled)
RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

# Anonymize PII (credit cards, phone numbers, emails) before sending to Path B
ENABLE_PII_REDACTION: bool = os.getenv("ENABLE_PII_REDACTION", "true").lower() == "true"

# Shadow Mode: Mirror traffic and audit savings without blocking or answering locally
SHADOW_MODE: bool = os.getenv("SHADOW_MODE", "false").lower() == "true"

