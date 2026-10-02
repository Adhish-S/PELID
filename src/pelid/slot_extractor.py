"""
Pelid Dynamic Slot & Entity Extractor

Extracts structured parameter slots (e.g. order_id, email, tracking_number, food_item, part_number)
from conversational user queries across multiple languages.

Architecture:
1. Fast Micro-Heuristic Slot Matchers (<1ms) for common structured tokens (IDs, codes, emails, phones).
2. Open-Vocabulary Zero-Shot Entity Extractor interface (GLiNER ONNX architecture ready).
"""

import re
from typing import Any, Dict, List, Optional


# Pre-compiled high-precision slot patterns
SLOT_PATTERNS = {
    "order_id": [
        # Explicit markers with digits: #9921, ORD-9921, order id 9921, ID is 9921
        re.compile(r"(?:#|ord-?|order\s*(?:id|no|#|-id)?\s*|-id\s*|id\s+is\s+|id\s*)(#?[A-Z0-9]*[0-9][A-Z0-9]*)\b", re.IGNORECASE),
        # Pure numeric sequences between 4 and 10 digits
        re.compile(r"\b([0-9]{4,10})\b"),
    ],
    "tracking_number": [
        re.compile(r"\b(AWB[-0-9A-Z]{6,16}|BLR[0-9]{4,8}|TRK[-0-9A-Z]{6,16})\b", re.IGNORECASE),
    ],
    "email": [
        re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b"),
    ],
    "phone": [
        re.compile(r"\b(?:\+?91[-.\s]?)?([6-9]\d{9})\b"),
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b"),
    ],
}


class SlotExtractor:
    """
    Extensible entity and slot extractor.
    Combines sub-millisecond pattern matching with open-vocabulary zero-shot models.
    """

    def __init__(self, use_gliner: bool = False):
        self.use_gliner = use_gliner
        self._gliner_model = None

    def extract_slots(self, text: str, required_slots: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract required slots from the input text.

        Args:
            text: Raw user message.
            required_slots: List of slot names to extract (e.g. ['order_id', 'item']).
                            If None, extracts all detectable slots.

        Returns:
            Dict of extracted slots, e.g. {"order_id": "9921"}
        """
        if not text:
            return {}

        extracted: Dict[str, Any] = {}
        slots_to_check = required_slots if required_slots else list(SLOT_PATTERNS.keys())

        # 1. Fast Pattern Matching (<0.5ms)
        for slot in slots_to_check:
            if slot in SLOT_PATTERNS:
                for regex in SLOT_PATTERNS[slot]:
                    match = regex.search(text)
                    if match:
                        extracted[slot] = match.group(1) if match.groups() else match.group(0)
                        break

        # 2. Open-Vocabulary / Zero-Shot Slot Fallback (GLiNER Architecture)
        # If specific domain slots (like 'food_item', 'car_model', 'product_name') are needed
        missing_slots = [s for s in slots_to_check if s not in extracted]
        if missing_slots and self.use_gliner:
            gliner_results = self._extract_with_gliner(text, missing_slots)
            extracted.update(gliner_results)

        return extracted

    def _extract_with_gliner(self, text: str, entity_types: List[str]) -> Dict[str, str]:
        """
        Zero-shot entity extraction via GLiNER for open entities without regex.
        Falls back cleanly if gliner library/weights are not present.
        """
        try:
            if self._gliner_model is None:
                # Lazy-load gliner if installed
                from gliner import GLiNER
                self._gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.5")

            entities = self._gliner_model.predict_entities(text, entity_types, threshold=0.5)
            return {e["label"]: e["text"] for e in entities}
        except Exception:
            # Graceful zero-overhead fallback
            return {}


# Global singleton instance
_default_extractor = SlotExtractor()


def extract_slots(text: str, required_slots: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function to extract slots using default extractor."""
    return _default_extractor.extract_slots(text, required_slots)
