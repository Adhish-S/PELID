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
