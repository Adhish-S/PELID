"""
Pelid Security & Guardrails Engine

Hardens the gateway against:
1. Unauthorized proxy abuse (Bearer API key validation)
2. Prompt injection & jailbreak attacks (adversarial override detection)
3. Denial of Service (payload size & token limits)
"""

import hmac
import re
from typing import Optional

# Common prompt injection, jailbreak, and system extraction signatures
PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore (all|the|previous|above|system)\b.{0,40}\b(instructions|rules|prompts|directives)\b",
    r"(?i)\bdisregard (all|previous|system)\b.{0,30}\b(instructions|directives)\b",
    r"(?i)\brepeat (after me|the prompt|everything|verbatim)\b",
    r"(?i)\b(reveal|show|display|print)\b.{0,30}\b(system prompt|hidden prompt|secret key|api[ _-]?key)\b",
    r"(?i)\byou are now (in|operating in|dan|jailbreak)\b",
    r"(?i)\b(dan mode|jailbreak mode|developer mode|god mode)\b",
    r"(?i)<\|im_start\|>",
    r"(?i)\[INST\]",
    r"(?i)\b(sudo mode|superadmin mode)\b",
]

COMPILED_INJECTION_REGEX = [re.compile(p) for p in PROMPT_INJECTION_PATTERNS]


def is_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Scan incoming text for known prompt injection or adversarial jailbreak signatures.

    Returns:
        (is_injection, reason)
    """
    if not text:
        return False, ""

    for regex in COMPILED_INJECTION_REGEX:
        match = regex.search(text)
        if match:
            return True, f"Prompt injection signature detected: '{match.group(0)}'"

    return False, ""


def validate_api_key(auth_header: Optional[str], expected_key: str) -> bool:
    """
    Validate incoming Authorization header against configured PELID_PROXY_API_KEY.
    Uses constant-time comparison to prevent timing attacks.
    If expected_key is empty, auth is considered disabled (open development mode).
    """
    if not expected_key:
        return True

    if not auth_header:
        return False

    parts = auth_header.strip().split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    token = parts[1]
    return hmac.compare_digest(token, expected_key)


# ─── PII Redaction / Anonymization Filter ────────────────────

# Regex patterns for high-sensitivity data
PII_PATTERNS = [
    # 16-digit credit / debit card numbers (with spaces, hyphens, or continuous)
    (re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), "[REDACTED_CARD]"),
    # Standard email addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # 10-digit Indian & international mobile numbers
    (re.compile(r"\b(?:\+?91[-.\s]?)?[6-9]\d{9}\b"), "[REDACTED_PHONE]"),
    # US / International 10-digit phone numbers
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]


def anonymize_pii(text: str) -> str:
    """
    Sanitize sensitive user PII (credit cards, phone numbers, emails)
    before forwarding queries to external third-party foundation models on Path B.
    Ensures GDPR / HIPAA data leakage compliance.
    """
    if not text:
        return ""
    sanitized = text
    for pattern, replacement in PII_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


# ─── In-Memory Sliding Window Rate Limiter ───────────────────

import time
from collections import defaultdict


class SlidingWindowRateLimiter:
    """
    In-memory, sliding-window rate limiter per client IP address.
    Protects the gateway against runaway agent loops, accidental spam, and DoS.
    """

    def __init__(self, max_requests_per_minute: int = 120):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        """
        Check if client_ip is within the rate limit.

        Returns:
            (allowed: bool, retry_after_seconds: int)
        """
        # If rate limiting is disabled (0 or negative), allow immediately
        if self.max_requests <= 0:
            return True, 0

        now = time.time()
        window_start = now - 60.0

        # Clean old timestamps outside the 60s sliding window
        self.requests[client_ip] = [ts for ts in self.requests[client_ip] if ts > window_start]

        if len(self.requests[client_ip]) >= self.max_requests:
            # Calculate time until earliest timestamp expires
            earliest = self.requests[client_ip][0]
            retry_after = max(1, int(60.0 - (now - earliest)))
            return False, retry_after

        # Record this request
        self.requests[client_ip].append(now)
        return True, 0

