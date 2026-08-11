"""Hybrid retrieval: dense vectors + BM25, fused with RRF, then cross-encoder rerank.

Pipeline for one query:

1. embed the query with the *same* model used at indexing (model-lock guard);
2. dense ANN search (vector store) -> candidate set A;
3. BM25 lexical search over the active corpus -> candidate set B;
4. Reciprocal Rank Fusion merges A and B into one ranked list;
5. a cross-encoder reranks the top fused candidates;
6. return the top-k, each carrying every intermediate score for attribution.

Every stage records a span on the trace, and each surviving candidate emits a
``chunk_retrieved`` event -- this is what makes a bad answer debuggable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .bm25 import BM25Index
from .config import settings
from .embeddings import Embedder, get_embedder
from .reranker import get_reranker
from .store import Store, get_store
from .tracing import Trace
from .types import RetrievedChunk


class ModelMismatchError(RuntimeError):
    """Raised when stored vectors were produced by a different embedding model."""


def _rrf_fuse(rankings: List[List[str]], k: int) -> Dict[str, float]:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank) across ranked lists."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    def __init__(self, store: Optional[Store] = None, embedder: Optional[Embedder] = None) -> None:
        self.store = store or get_store()
        self.embedder = embedder or get_embedder()
        self.reranker = get_reranker()
        self._bm25: Optional[BM25Index] = None
        self._bm25_fingerprint: Optional[tuple] = None

    # BM25 index is rebuilt lazily and cached by (index_version, corpus size).
    def _get_bm25(self, index_version: str) -> BM25Index:
        corpus = self.store.active_corpus(index_version)
        fingerprint = (index_version, len(corpus))
        if self._bm25 is None or self._bm25_fingerprint != fingerprint:
            self._bm25 = BM25Index().build(corpus)
            self._bm25_fingerprint = fingerprint
        return self._bm25

    def retrieve(self, query: str, trace: Trace, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        index_version = self.store.get_current_index_version()
        trace.set("index_version", index_version)

        # 1. embed query
        with trace.span("embedding.query") as sp:
            qvec = self.embedder.embed([query])[0]
            sp.set("model", self.embedder.name)
            sp.set("dim", int(qvec.shape[0]))

        # 2. dense search
        with trace.span("retrieval.vector_search") as sp:
            hits = self.store.vector_search(qvec, settings.candidate_k, index_version)
            sp.set("top_k", settings.candidate_k)
            sp.set("num_results", len(hits))
            sp.set("index_version", index_version)
            # model-lock guard: stored model must match the query model
            for h in hits:
                if h.embedding_model != self.embedder.name:
                    raise ModelMismatchError(
                        f"chunk {h.chunk_vector_id} embedded with "
                        f"'{h.embedding_model}' but querying with '{self.embedder.name}'"
                    )
        by_id: Dict[str, RetrievedChunk] = {}
        vector_ranking: List[str] = []
        for h in hits:
            by_id[h.chunk_vector_id] = RetrievedChunk(
                chunk_vector_id=h.chunk_vector_id, doc_id=h.doc_id, text=h.text,
                section=h.section, ordinal=h.ordinal, vector_score=h.score,
                embedding_model=h.embedding_model, index_version=h.index_version,
            )
            vector_ranking.append(h.chunk_vector_id)

        # 3. BM25 lexical search
        bm25_ranking: List[str] = []
        if settings.use_bm25:
            with trace.span("retrieval.bm25") as sp:
                bm25 = self._get_bm25(index_version)
                bm_hits = bm25.search(query, settings.candidate_k)
                sp.set("num_results", len(bm_hits))
                needed = [cid for cid, _ in bm_hits if cid not in by_id]
                fetched = self.store.get_chunks(needed) if needed else {}
                for cid, score in bm_hits:
                    bm25_ranking.append(cid)
                    if cid in by_id:
                        by_id[cid].bm25_score = score
                    elif cid in fetched:
                        r = fetched[cid]
                        by_id[cid] = RetrievedChunk(
                            chunk_vector_id=cid, doc_id=r["doc_id"], text=r["text"],
                            section=r["section"], ordinal=r["ordinal"], bm25_score=score,
                            embedding_model=r["embedding_model"], index_version=r["index_version"],
                        )

        # 4. RRF fusion
        with trace.span("retrieval.fuse") as sp:
            rankings = [vector_ranking]
            if bm25_ranking:
                rankings.append(bm25_ranking)
            fused = _rrf_fuse(rankings, settings.rrf_k)
            for cid, score in fused.items():
                if cid in by_id:
                    by_id[cid].fused_score = score
            candidates = sorted(by_id.values(), key=lambda c: c.fused_score or 0.0, reverse=True)
            sp.set("method", "reciprocal_rank_fusion")
            sp.set("rrf_k", settings.rrf_k)
            sp.set("num_candidates", len(candidates))

        # 5. cross-encoder rerank (top fused candidates only)
        rerank_pool = candidates[: max(top_k * 4, 20)]
        with trace.span("retrieval.rerank") as sp:
            scores = self.reranker.score(query, [c.text for c in rerank_pool])
            for c, s in zip(rerank_pool, scores):
                c.rerank_score = s
            reranked = sorted(rerank_pool, key=lambda c: c.rerank_score, reverse=True)
            sp.set("model", self.reranker.name)
            sp.set("reranker_available", getattr(self.reranker, "available", False))
            sp.set("num_input", len(rerank_pool))

        # 6. final top-k + attribution events
        final = reranked[:top_k]
        for rank, c in enumerate(final):
            c.final_score = c.rerank_score if self.reranker.available else (c.fused_score or 0.0)
        with trace.span("retrieval.select") as sp:
            sp.set("top_k", top_k)
            for rank, c in enumerate(final):
                sp.event("chunk_retrieved", {"rank": rank, **c.to_event()})
        trace.set("num_chunks", len(final))
        return final
