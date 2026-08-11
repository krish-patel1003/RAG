"""Cross-encoder reranking.

Bi-encoder retrieval (query and chunk embedded independently) is fast but coarse.
A cross-encoder scores the (query, chunk) pair *jointly*, which is far more
accurate for the final ordering -- the blog calls this out as essential for
"meaningful accuracy".  We load ``cross-encoder/ms-marco-MiniLM-L-6-v2`` via
sentence-transformers.

If the model can't be loaded (offline, no weights cached) we fall back to a
:class:`NoopReranker` that keeps the fusion order, so the pipeline still runs.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .config import settings


class NoopReranker:
    name = "noop"
    available = False

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        # Preserve incoming order with a descending gradient.
        return [float(len(texts) - i) for i in range(len(texts))]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # heavy, lazy

        self.name = model_name
        self.available = True
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        pairs: List[Tuple[str, str]] = [(query, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs)]


_RERANKER = None


def get_reranker():
    """Return a singleton reranker, degrading to Noop if the model is unavailable."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    if not settings.use_reranker:
        _RERANKER = NoopReranker()
        return _RERANKER
    try:
        _RERANKER = CrossEncoderReranker(settings.reranker_model)
    except Exception as exc:  # noqa: BLE001 -- any load failure -> graceful noop
        print(f"[reranker] falling back to noop: {type(exc).__name__}: {exc}")
        _RERANKER = NoopReranker()
    return _RERANKER
