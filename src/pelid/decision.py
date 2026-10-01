"""
Pelid — Dual Decision Engine (Stage 6-7)

Runs Laya calibrated non-autoregressive decision heads on input queries.
Routes dynamically:
- 'en': ModernBERT backbone (1024-dim) + pelid_head_english.onnx
- 'ml': mmBERT backbone (768-dim) + pelid_head_multilingual.onnx

Routing Rules:
- Confidence >= 0.70 & Non-Destructive -> Path A (Local resolution, $0, ~30ms)
- Confidence < 0.70 OR Destructive -> Path B (Frontier LLM fallback)
"""

import json
from pathlib import Path
import numpy as np
import torch
import laya

from pelid.config import CONFIDENCE_THRESHOLD, DESTRUCTIVE_KEYWORDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

_agent_en = None
_agent_ml = None
_metadata_en = None
_metadata_ml = None
_onnx_session_en = None
_onnx_session_ml = None


def get_agent(language: str = "en") -> laya.Agent:
    """Lazy-load the appropriate frozen backbone (ModernBERT for en, mmBERT for ml/hi)."""
    global _agent_en, _agent_ml
    if language in ("ml", "hi"):
        if _agent_ml is None:
            _agent_ml = laya.load("convaiinnovations/laya", subfolder="multilingual")
        return _agent_ml
    else:
        if _agent_en is None:
            _agent_en = laya.load("convaiinnovations/laya")
        return _agent_en


def get_metadata(language: str = "en") -> dict:
    """Load metadata (labels and temperature) for the specified language head."""
    global _metadata_en, _metadata_ml
    if language in ("ml", "hi"):
        if _metadata_ml is None:
            meta_path = MODELS_DIR / "pelid_head_multilingual.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    _metadata_ml = json.load(f)
            else:
                _metadata_ml = {"labels": [], "temperature": 1.0}
        return _metadata_ml
    else:
        if _metadata_en is None:
            meta_path = MODELS_DIR / "pelid_head_english.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    _metadata_en = json.load(f)
            else:
                _metadata_en = {"labels": [], "temperature": 1.0}
        return _metadata_en


def get_onnx_session(language: str = "en"):
    """Load ONNX Runtime inference session for 5ms decision head inference."""
    global _onnx_session_en, _onnx_session_ml
    if language in ("ml", "hi"):
        if _onnx_session_ml is None:
            onnx_path = MODELS_DIR / "pelid_head_multilingual.onnx"
            if onnx_path.exists():
                try:
                    import onnxruntime as ort
                    _onnx_session_ml = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                except Exception:
                    _onnx_session_ml = None
        return _onnx_session_ml
    else:
        if _onnx_session_en is None:
            onnx_path = MODELS_DIR / "pelid_head_english.onnx"
            if onnx_path.exists():
                try:
                    import onnxruntime as ort
                    _onnx_session_en = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                except Exception:
                    _onnx_session_en = None
        return _onnx_session_en


class DecisionResult:
    """The result of a Laya decision."""

    def __init__(
        self,
        label: str,
        confidence: float,
        all_probabilities: dict[str, float] | None = None,
        embedding: Optional[np.ndarray] = None,
    ):
        self.label = label
        self.confidence = confidence
        self.all_probabilities = all_probabilities or {}
        self.embedding = embedding

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


async def run_decision(text: str, language: str = "en") -> DecisionResult:
    """
    Run Laya inference on the input text using the appropriate language head.

    Args:
        text: The input text to classify.
        language: "en" (ModernBERT 1024-dim) or "ml" (mmBERT 768-dim).

    Returns:
        A DecisionResult with the label, calibrated confidence score, and routing flag.
    """
    if not text or not text.strip():
        return DecisionResult(label="other_unclear", confidence=0.0)

    # 1. Select language-appropriate backbone and head
    agent = get_agent(language)
    meta = get_metadata(language)
    labels = meta.get("labels", [])
    temperature = meta.get("temperature", 1.0)

    # 2. Extract pooled embedding (1024-dim for en, 768-dim for ml)
    with torch.no_grad():
        enc = agent.tok([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        out = agent.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = out[0]

        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float32)

    # 3. Run Head Inference via ONNX session
    session = get_onnx_session(language)
    if session:
        logits = session.run(["logits"], {"embeddings": embedding})[0]
    else:
        # Fallback to PyTorch weights if ONNX runtime is unavailable
        pt_filename = "pelid_head_multilingual.pt" if language in ("ml", "hi") else "pelid_head_english.pt"
        pt_path = MODELS_DIR / pt_filename
        in_dim = 768 if language in ("ml", "hi") else 1024
        from scripts.train_head import PelidDecisionHead
        head = PelidDecisionHead(in_features=in_dim, num_classes=len(labels))
        head.load_state_dict(torch.load(pt_path, map_location="cpu"))
        head.eval()
        with torch.no_grad():
            logits = head(torch.from_numpy(embedding)).numpy()

    # 4. Apply Calibrated Temperature Scaling
    scaled_logits = logits / max(0.1, temperature)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]

    best_idx = int(np.argmax(probs))
    best_label = labels[best_idx] if best_idx < len(labels) else "other_unclear"
    confidence = float(probs[best_idx])
    prob_dict = {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}

    return DecisionResult(
        label=best_label,
        confidence=confidence,
        all_probabilities=prob_dict,
        embedding=embedding,
    )
