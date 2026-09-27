"""
Language Router (Stage 4 from the build plan).

Detects the language/script of the input and routes to the
appropriate Laya checkpoint:
  - Clean English → English checkpoint (ModernBERT-large, 421M params)
  - Mixed-script / Romanized Hindi / Hinglish / Manglish
    → Multilingual checkpoint (mmBERT-base, 322M params, 100+ languages)
"""

# TODO: Implement in Stage 4
# Use script detection + langdetect to classify input language


def detect_language(text: str) -> str:
    """
    Detect the language/script of the input text.

    Args:
        text: The input text to classify.

    Returns:
        "en" for clean English, "ml" for multilingual/mixed-script.
    """
    # TODO: Implement language detection
    # For now, default to English
    return "en"
