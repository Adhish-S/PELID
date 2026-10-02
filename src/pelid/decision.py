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
import re
from pathlib import Path
from typing import Optional
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
        query_text: str = "",
    ):
        self.label = label
        self.confidence = confidence
        self.all_probabilities = all_probabilities or {}
        self.embedding = embedding
        self.query_text = query_text

    @property
    def is_confident(self) -> bool:
        """Is the model confident enough to bypass the LLM?"""
        return self.confidence >= CONFIDENCE_THRESHOLD

    @property
    def is_destructive(self) -> bool:
        """Does this decision involve a destructive/irreversible action?"""
        label_match = any(keyword in self.label.lower() for keyword in DESTRUCTIVE_KEYWORDS)
        query_match = any(keyword in self.query_text.lower() for keyword in DESTRUCTIVE_KEYWORDS) if self.query_text else False
        return label_match or query_match

    @property
    def should_use_path_a(self) -> bool:
        """Should we use the free local answer (Path A)?"""
        # Destructive actions ALWAYS go to Path B, even if confident
        if self.is_destructive:
            return False
        return self.is_confident


def fast_path_pre_router(text: str, language: str = "en") -> Optional[DecisionResult]:
    """
    Sub-millisecond micro-heuristic pre-router for high-precision boundaries.
    Prevents rookie errors like escalating 'HI...' or misrouting compound defect queries.
    """
    q = text.strip()
    q_lower = q.lower()

    # 1. Greetings & Chit-chat (<1ms, Path A, $0.00)
    # Catches 'HI...', 'Hi!', 'Hello', 'Hey', 'Good morning', 'Namaskaram', 'Namaste', etc.
    greeting_patterns = [
        r"^(?:hi+|hello+|hey+|hai+|hlo+|halo)\b[\s.!?]*$",
        r"^(?:good\s+(?:morning|afternoon|evening)|namaskaram|namaste)\b[\s.!?]*$",
        r"^(?:hi+|hello+|hey+|hai+)\s+(?:team|all|there|sir|support|bro|assist|pelid)[\s.!?]*$",
    ]
    for pat in greeting_patterns:
        if re.search(pat, q_lower):
            return DecisionResult(
                label="general_chitchat_or_faq",
                confidence=1.0,
                query_text=text,
            )

    # 2. Destructive Guardrails (Cancellations & Refunds) - Overrides delivery words
    cancel_patterns = [
        r"\b(?:cancel\s*(?:my\s*)?order|order\s*cancel|radd\s*kardo|order\s*radd|cancel\s*aakkanam|cancel\s*cheyyu|stop\s*dispatch)\b",
        r"\b(?:cancel\s*(?:my\s*)?subscription|stop\s*auto\s*(?:pay|renewal|debit)|cancel\s*membership)\b",
        r"\b(?:delete\s*(?:my\s*)?account|close\s*(?:my\s*)?account|erase\s*(?:all\s*)?data)\b",
    ]
    for pat in cancel_patterns:
        if re.search(pat, q_lower):
            lbl = "cancel_subscription" if "sub" in pat or "membership" in q_lower or "auto" in q_lower else ("delete_account" if "account" in q_lower else "cancel_order")
            return DecisionResult(
                label=lbl,
                confidence=0.999,
                query_text=text,
            )

    if re.search(r"\b(?:refund|money\s*back|paisa\s*wapas|cash\s*(?:thirike|refund))\b", q_lower):
        return DecisionResult(
            label="refund_request",
            confidence=0.999,
            query_text=text,
        )

    # 3. Defect & Damage Dominance Rule
    # Even if query mentions 'package arrived this morning' or 'order #9921',
    # the presence of defect/damage terms makes the core intent wrong_or_defective_item!
    defect_terms = [
        "cracked", "crack", "pottiyath", "pottiya", "pottipoyi", "potti",
        "toota", "phoota", "damaged", "broken", "shattered", "defective",
        "keeri", "keeripoyi", "odinja", "odinjirikkunnu", "sathanam cracked",
        "screen cracked", "screen damaged", "display broken", "seal khula",
        "faulty", "tampered"
    ]
    if any(term in q_lower for term in defect_terms):
        return DecisionResult(
            label="wrong_or_defective_item",
            confidence=0.998,
            query_text=text,
        )

    # 4. Explicit GST / Invoice Download (without dispute/refund)
    if re.search(r"\b(?:gst\s*invoice|tax\s*invoice|download\s*bill|bill\s*copy|invoice\s*copy)\b", q_lower):
        if not any(w in q_lower for w in ["refund", "wrong", "extra", "dispute", "cancel"]):
            return DecisionResult(
                label="invoice_or_receipt",
                confidence=0.995,
                query_text=text,
            )

    # 5. Explicit Login / OTP / Password
    if re.search(r"\b(?:otp\s*(?:nahi|not|varnnilla|vannilla|delay)|password\s*reset|forgot\s*password|account\s*locked)\b", q_lower):
        return DecisionResult(
            label="login_or_password_issue",
            confidence=0.995,
            query_text=text,
        )

    # 6. Explicit Human Agent Escalation
    if re.search(r"\b(?:connect\s*(?:me\s*)?to\s*(?:a\s*)?(?:human|agent|person|executive)|speak\s*(?:with|to)\s*(?:a\s*)?human|manager\s*se\s*baat|agent-ine\s*connect)\b", q_lower):
        return DecisionResult(
            label="complaint_or_escalate_to_human",
            confidence=0.995,
            query_text=text,
        )

    return None


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

    # 0. Deterministic Fast-Path Pre-Router (<1ms)
    fast_result = fast_path_pre_router(text, language)
    if fast_result is not None:
        return fast_result

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
        query_text=text,
    )
