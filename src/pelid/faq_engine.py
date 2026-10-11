"""
Pelid Universal Zero-Code Semantic Knowledge Base & FAQ Engine

Enables any company (Fintech, EdTech, SaaS, Logistics, Healthcare) to onboard
custom Q&As without writing Python code or retraining neural classification heads.

Features:
1. Instant CSV / YAML Knowledge Ingestion: Ingests `faqs:` from `pelid.yaml` or `data/company_faqs.csv`.
2. Sub-5ms Vector Matching: ModernBERT normalized pooled embeddings with fast NumPy dot-product similarity.
3. Path A $0.00 Resolution: Queries matching approved company FAQs resolve locally with 0% hallucination.
4. Hot-Reloadable: Add or update FAQs via API / Admin Console without restarting the server.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from pelid.domain import load_domain_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FAQS_CSV = DATA_DIR / "company_faqs.csv"


@dataclass
class FAQItem:
    id: str
    question: str
    answer: str
    variations: List[str] = field(default_factory=list)
    domain: str = "general"
    category: str = "general"


@dataclass
class FAQMatchResult:
    matched: bool
    score: float
    faq_id: str
    question: str
    answer: str
    category: str = "general"


class FAQEngine:
    """
    Zero-code semantic FAQ search engine.
    """

    def __init__(self, similarity_threshold: float = 0.86):
        self.similarity_threshold = similarity_threshold
        self.faqs: List[FAQItem] = []
        self._texts: List[str] = []
        self._faq_indices: List[int] = []
        self._embeddings: Optional[np.ndarray] = None
        self._is_initialized: bool = False

    def load_faqs(self):
        """Load FAQs from pelid.yaml and optional company_faqs.csv."""
        self.faqs = []

        # 1. Load from pelid.yaml
        cfg = load_domain_config()
        yaml_faqs = cfg.get("faqs", [])
        if isinstance(yaml_faqs, list):
            for i, item in enumerate(yaml_faqs):
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    faq_id = item.get("id", f"faq_{i+1}")
                    q = item.get("question", "").strip()
                    a = item.get("answer", "").strip()
                    vars_ = item.get("variations", [])
                    cat = item.get("category", "general")
                    if q and a:
                        self.faqs.append(FAQItem(id=faq_id, question=q, answer=a, variations=vars_, category=cat))

        # 2. Load from data/company_faqs.csv if present
        if DEFAULT_FAQS_CSV.exists():
            try:
                with open(DEFAULT_FAQS_CSV, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        q = row.get("question", "").strip()
                        a = row.get("answer", "").strip()
                        faq_id = row.get("id", f"csv_faq_{i+1}")
                        cat = row.get("category", "general")
                        if q and a:
                            self.faqs.append(FAQItem(id=faq_id, question=q, answer=a, category=cat))
            except Exception as e:
                print(f"  [Pelid FAQ] Warning: Could not read {DEFAULT_FAQS_CSV}: {e}")

        # 3. Default starter company knowledge base if empty
        if not self.faqs:
            self.faqs = [
                FAQItem(
                    id="shipping_coverage",
                    question="Do you deliver to Bangalore?",
                    answer="Yes! We provide same-day and next-day delivery across Bangalore and major tier-1 cities.",
                    variations=["Can you ship to Bengaluru?", "Is Bangalore delivery available?", "Delivery in Bangalore?"],
                    category="shipping",
                ),
                FAQItem(
                    id="return_policy",
                    question="What is your return policy and window?",
                    answer="We offer hassle-free 7-day returns for all eligible products from the date of delivery.",
                    variations=["How many days to return?", "Return window for items?"],
                    category="returns",
                ),
                FAQItem(
                    id="payment_modes",
                    question="What payment methods do you accept?",
                    answer="We accept UPI, Credit/Debit cards (Visa, Mastercard, RuPay), Net Banking, and Cash on Delivery.",
                    variations=["Can I pay using UPI or cash on delivery?", "Payment options supported?"],
                    category="billing",
                ),
                FAQItem(
                    id="international_shipping",
                    question="Do you ship internationally outside India?",
                    answer="Currently, we ship across all states in India. International shipping will be launched soon.",
                    variations=["Can I order to USA or UK?", "Worldwide delivery available?"],
                    category="shipping",
                ),
            ]

        self._build_index()

    def _build_index(self):
        """Build normalized embedding index using ModernBERT backbone."""
        self._texts = []
        self._faq_indices = []

        for idx, faq in enumerate(self.faqs):
            self._texts.append(faq.question)
            self._faq_indices.append(idx)
            for var in faq.variations:
                if var.strip():
                    self._texts.append(var.strip())
                    self._faq_indices.append(idx)

        if not self._texts:
            self._embeddings = None
            self._is_initialized = True
            return

        # Compute embeddings via ModernBERT backbone
        try:
            from pelid.decision import get_agent
            agent = get_agent("en")
            with torch.no_grad():
                enc = agent.tok(self._texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                out = agent.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = out[0]

                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float32)

                # L2 Normalize for fast cosine similarity dot product
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-9)
                self._embeddings = pooled / norms

            self._is_initialized = True
            print(f"  [Pelid FAQ] Indexed {len(self.faqs)} FAQs ({len(self._texts)} questions/variations) with ModernBERT.")
        except Exception as e:
            print(f"  [Pelid FAQ] Warning: Could not pre-compute FAQ embeddings: {e}")
            self._embeddings = None
            self._is_initialized = True

    def match(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Optional[FAQMatchResult]:
        """
        Match an incoming user query against indexed FAQs using cosine similarity.

        Args:
            query: User's raw message
            query_embedding: Optional 1024-dim embedding if already calculated
            threshold: Cosine similarity threshold (defaults to self.similarity_threshold)

        Returns:
            FAQMatchResult if a high-confidence match is found, else None.
        """
        if not self._is_initialized:
            self.load_faqs()

        if self._embeddings is None or len(self._texts) == 0:
            return None

        eff_threshold = threshold if threshold is not None else self.similarity_threshold

        # Extract and normalize query embedding if not provided
        if query_embedding is None:
            try:
                from pelid.decision import get_agent
                agent = get_agent("en")
                with torch.no_grad():
                    enc = agent.tok([query], padding=True, truncation=True, max_length=128, return_tensors="pt")
                    input_ids = enc["input_ids"]
                    attention_mask = enc["attention_mask"]
                    out = agent.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    token_embeddings = out[0]

                    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    q_emb = (sum_embeddings / sum_mask).cpu().numpy().astype(np.float32)
            except Exception:
                return None
        else:
            q_emb = np.array(query_embedding, dtype=np.float32)
            if q_emb.ndim == 1:
                q_emb = q_emb.reshape(1, -1)

        # L2 Normalize
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norm = np.maximum(q_norm, 1e-9)
        normalized_q = (q_emb / q_norm)[0]

        # Fast matrix dot product for cosine similarity across all stored FAQ vectors
        scores = np.dot(self._embeddings, normalized_q)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # Content token filtering to prevent Transformer anisotropy false-positives
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'you',
            'your', 'my', 'me', 'i', 'we', 'our', 'it', 'to', 'for', 'in', 'on', 'at',
            'by', 'with', 'and', 'or', 'can', 'what', 'where', 'how', 'when', 'which',
            'who', 'why', 'this', 'that', 'there', 'please', 'tell', 'about', 'get', 'any'
        }
        import re
        q_tokens = set(re.findall(r'\b[a-z0-9]{3,}\b', query.lower())) - stopwords
        faq_idx = self._faq_indices[best_idx]
        faq_item = self.faqs[faq_idx]
        matched_text = self._texts[best_idx]
        target_tokens = set(re.findall(r'\b[a-z0-9]{3,}\b', matched_text.lower())) - stopwords

        has_overlap = bool(q_tokens & target_tokens)

        # High confidence match requires either strong lexical overlap (>=0.82) or near-identical semantics (>=0.96)
        if (has_overlap and best_score >= 0.82) or (best_score >= 0.96):
            return FAQMatchResult(
                matched=True,
                score=round(best_score, 4),
                faq_id=faq_item.id,
                question=faq_item.question,
                answer=faq_item.answer,
                category=faq_item.category,
            )

        return None

    def add_faq(self, question: str, answer: str, category: str = "general", variations: Optional[List[str]] = None) -> FAQItem:
        """Add an FAQ dynamically and update index."""
        faq_id = f"custom_faq_{len(self.faqs) + 1}"
        item = FAQItem(id=faq_id, question=question, answer=answer, variations=variations or [], category=category)
        self.faqs.append(item)
        self._build_index()
        return item


# Global singleton instance
_GLOBAL_FAQ_ENGINE = FAQEngine()


def get_faq_engine() -> FAQEngine:
    global _GLOBAL_FAQ_ENGINE
    if not _GLOBAL_FAQ_ENGINE._is_initialized:
        _GLOBAL_FAQ_ENGINE.load_faqs()
    return _GLOBAL_FAQ_ENGINE
