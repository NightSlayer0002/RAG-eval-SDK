"""Lazy, bounded embedding backends.

The original project loaded a transformer at import time. That made even
schema validation allocate model memory and made the package impossible to use
without PyTorch. This module keeps the legacy ``embed_text`` API while making
model loading lazy and configurable.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from threading import RLock
from typing import Protocol, Sequence

import numpy as np


class SimilarityBackend(Protocol):
    """Minimal interface used by the evaluation pipeline."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a normalized ``(len(texts), dimensions)`` float matrix."""


class HashingSimilarityBackend:
    """Dependency-free lexical baseline.

    This backend is intentionally a baseline, not a semantic verifier. It is
    useful for tests, constrained deployments, and measuring the incremental
    value of a learned embedding model.
    """

    def __init__(self, dimensions: int = 2048) -> None:
        if dimensions < 128:
            raise ValueError("dimensions must be at least 128")
        self.dimensions = dimensions
        self.name = f"hashing-word-bigram-{dimensions}"

    @staticmethod
    def _terms(text: str) -> list[str]:
        words = re.findall(r"[\w%.-]+", text.casefold(), flags=re.UNICODE)
        return words + [f"{a}::{b}" for a, b in zip(words, words[1:])]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in self._terms(text):
                digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
                raw = int.from_bytes(digest, "little")
                column = raw % self.dimensions
                # Use disjoint hash bits for index and sign. With power-of-two
                # dimensions, the low bit is part of the index and cannot also
                # provide an independent signed-hashing projection.
                sign = 1.0 if (raw >> 32) & 1 else -1.0
                matrix[row, column] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceTransformerBackend:
    """Lazy sentence-transformer backend with a bounded in-process cache."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        cache_entries: int = 4096,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "RAG_EVAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = device or os.getenv("RAG_EVAL_DEVICE") or None
        self.cache_entries = max(0, cache_entries)
        self.batch_size = max(1, batch_size)
        self.name = self.model_name
        self._model = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = RLock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover - environment specific
                    raise RuntimeError(
                        "Sentence-transformers is not installed. Install "
                        "rag-eval-sdk[local] or select HashingSimilarityBackend."
                    ) from exc
                kwargs = {"device": self.device} if self.device else {}
                self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized = [str(text or "") for text in texts]
        if not normalized:
            return np.empty((0, 0), dtype=np.float32)

        if self.cache_entries == 0:
            vectors = self._load().encode(
                normalized,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        missing: list[str] = []
        with self._lock:
            for text in normalized:
                if text not in self._cache and text not in missing:
                    missing.append(text)

        if missing:
            vectors = self._load().encode(
                missing,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            with self._lock:
                for text, vector in zip(missing, vectors):
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                while len(self._cache) > self.cache_entries:
                    self._cache.popitem(last=False)

        with self._lock:
            return np.stack([self._cache[text] for text in normalized]).astype(np.float32)


_DEFAULT_BACKEND: SimilarityBackend | None = None
_DEFAULT_LOCK = RLock()


def get_default_backend() -> SimilarityBackend:
    """Return the configured backend without loading its model yet."""

    global _DEFAULT_BACKEND
    with _DEFAULT_LOCK:
        if _DEFAULT_BACKEND is None:
            backend_name = os.getenv("RAG_EVAL_SIMILARITY_BACKEND", "hashing")
            if backend_name.casefold() in {"hash", "hashing", "lexical"}:
                _DEFAULT_BACKEND = HashingSimilarityBackend()
            else:
                _DEFAULT_BACKEND = SentenceTransformerBackend()
        return _DEFAULT_BACKEND


def set_default_backend(backend: SimilarityBackend | None) -> None:
    """Inject a backend, primarily for controlled experiments and tests."""

    global _DEFAULT_BACKEND
    with _DEFAULT_LOCK:
        _DEFAULT_BACKEND = backend


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if not denominator:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


def similarity_matrix(
    queries: Sequence[str],
    documents: Sequence[str],
    backend: SimilarityBackend | None = None,
) -> np.ndarray:
    if not queries or not documents:
        return np.zeros((len(queries), len(documents)), dtype=np.float32)
    encoder = backend or get_default_backend()
    encoded = encoder.encode([*queries, *documents])
    query_vectors = encoded[: len(queries)]
    document_vectors = encoded[len(queries) :]
    return np.matmul(query_vectors, document_vectors.T)


def embed_text(text: str) -> np.ndarray:
    """Backward-compatible single-text embedding helper."""

    return get_default_backend().encode([text])[0]
