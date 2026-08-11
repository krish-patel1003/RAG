"""Pluggable embedding models.

The embedding model is a *long-term commitment* (see the blog's "model-lock"
section): every stored vector must be produced by the same model used at query
time.  To make that contract enforceable we tag every embedder with a stable
``name`` and ``dim``; those values are persisted in the registry and asserted at
query time.

Two implementations ship here:

* :class:`GeminiEmbedder` -- production embedder (``gemini-embedding-001``).
* :class:`HashingEmbedder` -- deterministic, dependency-free hashing embedder so
  the pipeline and its tests run fully offline.  It is a bag-of-words hashed
  projection: not competitive for real recall, but stable and fast, and BM25
  carries the lexical load in offline mode.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol, Sequence

import numpy as np

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised row vectors."""
        ...


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic hashing embedder (offline / CI fallback)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, count in counts.items():
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                # sublinear tf weighting, à la tf-idf
                out[i, idx] += sign * (1.0 + math.log(count))
        return _l2_normalise(out)


class GeminiEmbedder:
    """Gemini embedding model.  Batches requests and L2-normalises output."""

    def __init__(self) -> None:
        from google import genai  # lazy import; heavy + optional

        self.dim = settings.embedding_dimensions
        self.name = f"{settings.embedding_model}-{self.dim}"
        self._model = settings.embedding_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from google.genai import types

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: List[List[float]] = []
        batch = settings.embed_batch_size
        for start in range(0, len(texts), batch):
            chunk = list(texts[start : start + batch])
            resp = self._client.models.embed_content(
                model=self._model,
                contents=chunk,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dim
                ),
            )
            vectors.extend(e.values for e in resp.embeddings)
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))


def get_embedder() -> Embedder:
    kind = settings.resolved_embedder()
    if kind == "gemini":
        return GeminiEmbedder()
    if kind == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder: {kind}")
