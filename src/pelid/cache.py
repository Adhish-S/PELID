"""
Pelid Sub-Millisecond Semantic & Exact Cache Engine (Stage 10)

Provides a dual-level cache for ultra-high-frequency queries:
1. Level 1: Normalized Exact Hash Cache (<0.05ms lookup)
2. Level 2: Semantic Cosine Similarity Cache (<0.8ms lookup on pre-computed embeddings)

Reduces CPU load to near zero and delivers sub-1ms responses for repeated
FAQs, greeting chitchat, and tracking inquiries.
"""

import hashlib
import re
import time
from typing import Any, Optional
import numpy as np

# Cache limits and thresholds
MAX_CACHE_ENTRIES = 2000
DEFAULT_SIMILARITY_THRESHOLD = 0.96


def normalize_query(text: str) -> str:
    """Normalize query by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s#]", "", text)
    return re.sub(r"\s+", " ", text)


class CacheEntry:
    """A single cached entry with exact string, embedding, and response payload."""

    def __init__(
        self,
        query: str,
        normalized: str,
        language: str,
        embedding: Optional[np.ndarray],
        response_content: str,
        intent: str,
        confidence: float,
        tokens_saved: int,
    ):
        self.query = query
        self.normalized = normalized
        self.language = language
        self.embedding = embedding
        self.response_content = response_content
        self.intent = intent
        self.confidence = confidence
        self.tokens_saved = tokens_saved
        self.hit_count = 0
        self.last_accessed = time.time()


class SemanticCache:
    """In-memory dual-level LRU cache with exact and cosine similarity matching."""

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES):
        self.max_entries = max_entries
        self.exact_map: dict[str, CacheEntry] = {}
        self.entries: list[CacheEntry] = []
        self.total_lookups = 0
        self.exact_hits = 0
        self.semantic_hits = 0

    def _make_key(self, normalized: str, language: str) -> str:
        raw = f"{language}:{normalized}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_exact(self, query: str, language: str) -> Optional[dict[str, Any]]:
        """Level 1: Lightning-fast exact hash lookup (<0.05ms)."""
        self.total_lookups += 1
        norm = normalize_query(query)
        key = self._make_key(norm, language)

        entry = self.exact_map.get(key)
        if entry:
            self.exact_hits += 1
            entry.hit_count += 1
            entry.last_accessed = time.time()
            return {
                "content": entry.response_content,
                "intent": entry.intent,
                "confidence": entry.confidence,
                "tokens_saved": entry.tokens_saved,
                "cache_type": "exact",
                "hit_count": entry.hit_count,
            }
        return None

    def get_semantic(
        self,
        embedding: np.ndarray,
        language: str,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> Optional[dict[str, Any]]:
        """Level 2: Cosine similarity search over cached embeddings in the same language (<0.8ms)."""
        if not self.entries or embedding is None:
            return None

        # Filter candidate entries by matching language and having embeddings
        candidates = [e for e in self.entries if e.language == language and e.embedding is not None]
        if not candidates:
            return None

        # Stack candidate vectors (N, D)
        candidate_matrix = np.vstack([c.embedding for c in candidates])

        # Normalize query vector
        query_norm = np.linalg.norm(embedding)
        if query_norm == 0:
            return None
        q_unit = embedding / query_norm

        # Compute cosine similarities in parallel with BLAS / NumPy
        matrix_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        matrix_norms[matrix_norms == 0] = 1e-9
        unit_matrix = candidate_matrix / matrix_norms

        similarities = np.dot(unit_matrix, q_unit.T).flatten()
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= threshold:
            self.semantic_hits += 1
            match = candidates[best_idx]
            match.hit_count += 1
            match.last_accessed = time.time()
            return {
                "content": match.response_content,
                "intent": match.intent,
                "confidence": match.confidence,
                "tokens_saved": match.tokens_saved,
                "cache_type": "semantic",
                "similarity": round(best_sim, 3),
                "matched_query": match.query,
                "hit_count": match.hit_count,
            }

        return None

    def put(
        self,
        query: str,
        language: str,
        embedding: Optional[np.ndarray],
        response_content: str,
        intent: str,
        confidence: float,
        tokens_saved: int,
    ):
        """Insert or update a resolved item in the cache."""
        norm = normalize_query(query)
        if not norm or len(norm) < 3:
            return

        key = self._make_key(norm, language)

        # Evict oldest entry if at capacity
        if len(self.entries) >= self.max_entries:
            oldest = min(self.entries, key=lambda e: e.last_accessed)
            oldest_key = self._make_key(oldest.normalized, oldest.language)
            self.exact_map.pop(oldest_key, None)
            self.entries.remove(oldest)

        # Flatten 2D embedding to 1D if needed
        emb_flat = embedding.flatten() if embedding is not None else None

        entry = CacheEntry(
            query=query,
            normalized=norm,
            language=language,
            embedding=emb_flat,
            response_content=response_content,
            intent=intent,
            confidence=confidence,
            tokens_saved=tokens_saved,
        )

        self.exact_map[key] = entry
        self.entries.append(entry)

    def clear(self):
        """Clear the cache."""
        self.exact_map.clear()
        self.entries.clear()
        self.total_lookups = 0
        self.exact_hits = 0
        self.semantic_hits = 0

    def stats(self) -> dict[str, Any]:
        """Return cache health and hit rate statistics."""
        total_hits = self.exact_hits + self.semantic_hits
        hit_rate = (total_hits / self.total_lookups * 100.0) if self.total_lookups > 0 else 0.0
        return {
            "entries_count": len(self.entries),
            "total_lookups": self.total_lookups,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "total_hits": total_hits,
            "hit_rate_pct": round(hit_rate, 1),
        }


# Global singleton cache instance
_GLOBAL_CACHE = SemanticCache()


def get_cache() -> SemanticCache:
    """Access the global Pelid semantic cache instance."""
    return _GLOBAL_CACHE
