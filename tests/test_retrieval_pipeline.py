"""Retrieval, hybrid fusion, model-lock guard, and the end-to-end pipeline."""

import numpy as np
import pytest

from ragcore.bm25 import BM25Index


def test_bm25_ranks_lexical_match_first():
    idx = BM25Index().build([
        ("a", "the battery lasts ten hours on a full charge"),
        ("b", "international shipping takes seven to fourteen days"),
        ("c", "the widget has a two year warranty"),
    ])
    hits = idx.search("how long does the battery last", top_k=3)
    assert hits and hits[0][0] == "a"


def test_pipeline_retrieves_relevant_document(rag_env):
    from ragcore import Indexer, RAGPipeline

    ix = Indexer()
    ix.index_corpus([
        ("warranty", "The ACME widget has a two-year warranty. Returns within 30 days."),
        ("battery", "The device battery lasts about ten hours on a full charge."),
        ("shipping", "Orders ship in two business days. International shipping takes 7-14 days."),
    ])
    res = RAGPipeline().query("How long does the battery last?", judge=False)
    assert res["sources"], "expected at least one retrieved chunk"
    assert res["sources"][0]["doc_id"] == "battery"
    assert res["trace_id"]
    assert res["mode"] == "extractive"  # no key in test env


def test_model_lock_guard_detects_drift(rag_env):
    from ragcore import Indexer
    from ragcore.retriever import ModelMismatchError, Retriever
    from ragcore.tracing import Trace

    ix = Indexer()
    ix.index_document("d1", "Content embedded by the hashing model.")

    r = Retriever()
    # Simulate an embedding-model swap without re-indexing.
    r.embedder.name = "some-other-model-999"
    with pytest.raises(ModelMismatchError):
        r.retrieve("anything", Trace())


def test_valid_from_hides_staged_chunks(rag_env):
    from datetime import timedelta

    from ragcore import Indexer
    from ragcore.store import utcnow

    ix = Indexer()
    # Stage a document 5 minutes in the future -> must not be visible now.
    future = utcnow() + timedelta(minutes=5)
    ix.index_document("staged", "Future content not yet live.", valid_from=future)
    version = ix.store.get_current_index_version()
    qvec = ix.embedder.embed(["future content"])[0]
    hits = ix.store.vector_search(qvec, k=5, index_version=version)
    assert all(h.doc_id != "staged" for h in hits)
