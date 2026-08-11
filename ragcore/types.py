"""Shared dataclasses passed between pipeline stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


def sanitize_text(text: str) -> str:
    """Strip bytes that break storage/embedding.

    Postgres ``TEXT`` columns cannot contain NUL (``0x00``), which crawled pages
    and extracted PDFs (e.g. arXiv papers) sometimes include. We also drop the
    other C0 control characters except tab/newline/carriage-return, which carry
    no semantic value and only cause trouble downstream.
    """
    if not text:
        return text
    if "\x00" in text:
        text = text.replace("\x00", "")
    # Remove remaining non-printable C0 controls, keeping \t \n \r.
    return "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or ord(ch) >= 0x20
    )


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    """A unit of retrievable text plus the metadata that makes it debuggable."""

    doc_id: str
    ordinal: int                       # position within the document
    text: str
    section: str = ""                  # nearest heading (structure-aware split)
    char_start: int = 0
    char_end: int = 0
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.text)


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, carrying every score for attribution."""

    chunk_vector_id: str
    doc_id: str
    text: str
    section: str = ""
    ordinal: int = 0
    vector_score: Optional[float] = None   # cosine similarity
    bm25_score: Optional[float] = None     # lexical score
    fused_score: Optional[float] = None    # reciprocal-rank-fusion score
    rerank_score: Optional[float] = None   # cross-encoder score
    final_score: float = 0.0
    embedding_model: str = ""
    index_version: str = ""

    def to_event(self) -> dict:
        """Structured record logged as a trace event (chunk-level attribution)."""
        return {
            "chunk_vector_id": self.chunk_vector_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "ordinal": self.ordinal,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "fused_score": self.fused_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "embedding_model": self.embedding_model,
            "index_version": self.index_version,
            "preview": self.text[:160],
        }
