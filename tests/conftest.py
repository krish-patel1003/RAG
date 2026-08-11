"""Test configuration: force the offline, zero-dependency path.

Tests run with the SQLite backend and the deterministic hashing embedder, so the
whole pipeline is exercised without a database, network, or API key. Each test
gets an isolated temp DB via the ``env`` fixture.
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "sqlite")
    monkeypatch.setenv("RAG_EMBEDDER", "hashing")
    monkeypatch.setenv("RAG_USE_RERANKER", "0")
    monkeypatch.setenv("RAG_SQLITE_PATH", str(tmp_path / "rag.db"))
    monkeypatch.setenv("RAG_TRACE_DB", str(tmp_path / "traces.db"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Reload config + modules that snapshot settings at import time.
    import ragcore.config as cfg
    importlib.reload(cfg)
    for name in [
        "ragcore.embeddings", "ragcore.store", "ragcore.store.sqlite_store",
        "ragcore.bm25", "ragcore.chunking", "ragcore.reranker", "ragcore.retriever",
        "ragcore.generator", "ragcore.evaluation", "ragcore.tracing",
        "ragcore.indexer", "ragcore.connectors", "ragcore.connectors.filesystem",
        "ragcore.connectors.web", "ragcore.ingest", "ragcore.pipeline", "ragcore",
    ]:
        mod = importlib.import_module(name)
        importlib.reload(mod)
    yield
