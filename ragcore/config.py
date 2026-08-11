"""Central configuration for the production RAG system.

Everything is env-driven so the same code runs in three modes:

* ``RAG_BACKEND=pgvector`` (default) -- production / scale path backed by
  Postgres + pgvector HNSW.  Used by the Docker demo.
* ``RAG_BACKEND=sqlite``            -- zero-dependency path used by the unit
  tests and for laptop experiments; no external services required.
* ``RAG_EMBEDDER=hashing``          -- deterministic offline embedder so CI can
  run the full pipeline without network access or an API key.

The defaults are chosen so ``docker compose up`` "just works" while the test
suite can flip a couple of env vars and run fully offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass
class Settings:
    # -- storage backend -------------------------------------------------
    backend: str = os.getenv("RAG_BACKEND", "pgvector")  # 'pgvector' | 'sqlite'
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://rag:rag@db:5432/rag"
    )
    sqlite_path: str = os.getenv("RAG_SQLITE_PATH", "data/rag.db")

    # -- embeddings ------------------------------------------------------
    # 'gemini' uses the Gemini API; 'hashing' is a deterministic offline
    # fallback used by tests.  If 'gemini' is selected but no key is present
    # the factory downgrades to 'hashing' and logs a warning.
    embedder: str = os.getenv("RAG_EMBEDDER", "gemini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    gemini_judge_model: str = os.getenv("GEMINI_JUDGE_MODEL", "gemini-flash-lite-latest")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dimensions: int = _int("EMBEDDING_DIMENSIONS", 768)
    embed_batch_size: int = _int("EMBED_BATCH_SIZE", 64)

    # -- chunking --------------------------------------------------------
    chunk_size: int = _int("CHUNK_SIZE", 900)          # characters (~220 tokens)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")  # recursive|semantic|structure
    semantic_threshold: float = _float("SEMANTIC_THRESHOLD", 0.55)

    # -- retrieval -------------------------------------------------------
    top_k: int = _int("RAG_TOP_K", 5)                  # chunks sent to the LLM
    candidate_k: int = _int("RAG_CANDIDATE_K", 40)     # per-retriever fan-out
    rrf_k: int = _int("RAG_RRF_K", 60)                 # RRF smoothing constant
    use_bm25: bool = os.getenv("RAG_USE_BM25", "1") == "1"
    use_reranker: bool = os.getenv("RAG_USE_RERANKER", "1") == "1"
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # -- observability ---------------------------------------------------
    judge_sample_rate: float = _float("RAG_JUDGE_SAMPLE_RATE", 1.0)
    trace_db_path: str = os.getenv("RAG_TRACE_DB", "data/traces.db")
    rationale_sample_rate: float = _float("RAG_RATIONALE_SAMPLE_RATE", 0.0)

    # -- ingestion / research tool --------------------------------------
    searxng_url: str = os.getenv("SEARXNG_URL", "http://localhost:8080")
    # Science/paper aggregation across many engines can take ~30s; allow headroom.
    search_timeout: int = _int("RAG_SEARCH_TIMEOUT", 45)
    research_max_results: int = _int("RAG_RESEARCH_MAX_RESULTS", 10)

    # -- scale knobs (pgvector HNSW) ------------------------------------
    hnsw_m: int = _int("PG_HNSW_M", 16)
    hnsw_ef_construction: int = _int("PG_HNSW_EF_CONSTRUCTION", 200)
    hnsw_ef_search: int = _int("PG_HNSW_EF_SEARCH", 100)

    def resolved_embedder(self) -> str:
        """Downgrade gemini->hashing when no API key is configured."""
        if self.embedder == "gemini" and not self.gemini_api_key:
            return "hashing"
        return self.embedder


settings = Settings()
