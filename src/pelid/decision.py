"""
Decision Engine (Stage 6-7 from the build plan).

Runs the Laya model to make a decision on a classified request.
Returns the decision label and a confidence score.

Based on the confidence:
  - >= 0.70 → Path A (return local answer, bypass LLM, FREE)
  - < 0.70  → Path B (forward to frontier LLM)

Safety override: destructive actions (delete, cancel, refund, etc.)
ALWAYS go to Path B regardless of confidence score.
"""

from pelid.config import CONFIDENCE_THRESHOLD, DESTRUCTIVE_KEYWORDS

# TODO: Implement in Stage 6-7
# Load the fine-tuned Laya ONNX model
# Run inference and return decision + confidence


class DecisionResult:
    """The result of a Laya decision."""

    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

    @property
    def is_confident(self) -> bool:
        """Is the model confident enough to bypass the LLM?"""
        return self.confidence >= CONFIDENCE_THRESHOLD

    @property
    def is_destructive(self) -> bool:
        """Does this decision involve a destructive/irreversible action?"""
        return any(keyword in self.label.lower() for keyword in DESTRUCTIVE_KEYWORDS)

    @property
    def should_use_path_a(self) -> bool:
        """Should we use the free local answer (Path A)?"""
        # Destructive actions ALWAYS go to Path B, even if confident
        if self.is_destructive:
            return False
        return self.is_confident


async def run_decision(text: str, language: str) -> DecisionResult:
    """
    Run Laya inference on the input text.

    Args:
        text: The input text to classify.
        language: "en" or "ml" — determines which checkpoint to use.

    Returns:
        A DecisionResult with the label and confidence score.
    """
    # TODO: Implement ONNX inference
    # For now, return a dummy result that always routes to Path B (safe default)
    return DecisionResult(label="unknown", confidence=0.0)
