"""
Pelid — Tri-Language & Script Router (Stage 4)

Routes incoming requests to the appropriate model checkpoint and response tone:
- 'en': Pure English text -> ModernBERT English Head (1024-dim)
- 'hi': Hindi, Hinglish, or Devanagari script -> mmBERT Multilingual Head (768-dim) + Hinglish response
- 'ml': Malayalam, Manglish, or Malayalam script -> mmBERT Multilingual Head (768-dim) + Manglish response
"""

import re
import laya

# Dedicated Romanized Malayalam (Manglish) markers
MALAYALAM_TOKENS = {
    "ente", "ennte", "eppo", "eppol", "eppozha", "kittum", "kittiyilla", "thaa",
    "tharumo", "vannu", "vannitilla", "vannath", "pottiya", "potti", "aavnnilla", "aakunn",
    "aakumpol", "aano", "poyi", "undu", "undo", "venam", "venda", "evide", "entha",
    "enthanu", "engane", "enghana", "nokk", "sheri", "machane", "aakkanam", "aanu", "aan",
    "ith", "cheruthann", "eedeya", "ulle", "kidakkukayanu", "kooduthal", "valare",
    "mosham", "nanni", "ariyilla", "njan", "pinne", "kaanaam", "oru", "divasam",
    "thorakkum", "aayilla", "varnnilla", "pattuo", "pattumo", "thavana", "eda",
    "cheyyo", "cheyyam", "njangal", "njangalude", "ningal", "ningalude", "kazhinjilla",
    "mathi", "khedikkunnu", "parayam"
}

# Dedicated Romanized Hindi (Hinglish) markers
HINDI_TOKENS = {
    "mera", "meri", "mere", "kab", "kahaa", "kahan", "kyu", "kyun", "hai", "hain",
    "nahi", "kardo", "chahiye", "bhai", "gaya", "aaya", "paise", "baat", "karo",
    "kuch", "bhi", "ho", "tha", "aaj", "batao", "aur", "pehle", "bhej", "galat",
    "dabba", "diya", "bol", "rahe", "kaise", "karu", "suno", "re", "dikhao", "aree",
    "kitna", "lagega", "hoga", "shuru", "bass", "bohot", "bakwas", "wapas", "khatam",
    "tuuta", "toota", "kharab", "madad", "hum", "aapka", "aapki", "aapko", "karte",
    "karenge", "sharminda", "maafi", "hua", "dekh"
}


def detect_language(text: str) -> str:
    """
    Detect whether text is English, Hinglish, or Manglish.

    Returns:
        "en" for pure English.
        "hi" for Hindi / Hinglish.
        "ml" for Malayalam / Manglish.
    """
    if not text or not text.strip():
        return "en"

    # 1. Native Indic Script Check
    script = laya.detect_script(text)
    if script == "malayalam":
        return "ml"
    elif script in ("devanagari", "bengali", "gujarati", "gurmukhi"):
        return "hi"
    elif script != "latin":
        return "hi"

    # 2. Token Lexicon Matching
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    ml_matches = len(words.intersection(MALAYALAM_TOKENS))
    hi_matches = len(words.intersection(HINDI_TOKENS))

    if ml_matches > hi_matches and ml_matches > 0:
        return "ml"
    elif hi_matches > ml_matches and hi_matches > 0:
        return "hi"
    elif ml_matches > 0 and ml_matches == hi_matches:
        # Tie-breaker: if distinctive Malayalam word present, favor ml
        return "ml"

    # 3. Default to English
    return "en"
