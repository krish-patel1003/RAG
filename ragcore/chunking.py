"""Chunking strategies.

The blog's thesis: fixed-size character chunking "cuts sentences in half,
separates questions from answers, and splits code across function boundaries."
We implement the three strategies that hold up in production:

* ``recursive``  -- split on paragraphs, then sentences, then characters as a
  fallback, packing pieces up to a target size with overlap.  Good default.
* ``structure``  -- Markdown-heading aware: every chunk carries its parent
  section heading, and boundaries never cross a heading.
* ``semantic``   -- embed consecutive sentences and start a new chunk where the
  cosine similarity between adjacent sentences drops below a threshold (a real
  topic shift) rather than at an arbitrary character offset.

All strategies emit :class:`~ragcore.types.Chunk` objects with the metadata the
rest of the system relies on (doc_id, ordinal, section, char range, hash).
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

from .config import settings
from .types import Chunk

# Paragraph / sentence splitters kept deliberately simple and dependency-free.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _pack(
    pieces: Sequence[str],
    size: int,
    overlap: int,
    separator: str = " ",
) -> List[str]:
    """Greedily pack pieces into <= size windows with character overlap."""
    windows: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(separator) + len(piece) <= size:
            current = f"{current}{separator}{piece}"
        else:
            windows.append(current)
            # carry a tail of the previous window as overlap
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}{separator}{piece}".strip() if tail else piece
        # a single oversize piece: hard-split on characters
        while len(current) > size:
            windows.append(current[:size])
            current = current[size - overlap :] if overlap < size else current[size:]
    if current.strip():
        windows.append(current)
    return windows


def recursive_split(text: str, size: int, overlap: int) -> List[str]:
    """Paragraph -> sentence -> character fallback."""
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    units: List[str] = []
    for para in paragraphs:
        if len(para) <= size:
            units.append(para)
        else:
            units.extend(split_sentences(para) or [para])
    return _pack(units, size, overlap)


def structure_split(text: str, size: int, overlap: int) -> List[tuple[str, str]]:
    """Markdown-heading-aware split.  Returns (section_heading, chunk_text)."""
    lines = text.splitlines()
    sections: List[tuple[str, str]] = []
    heading = ""
    buff: List[str] = []

    def flush() -> None:
        body = "\n".join(buff).strip()
        if body:
            for piece in recursive_split(body, size, overlap):
                sections.append((heading, piece))

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            buff = []
            heading = m.group(2).strip()
        else:
            buff.append(line)
    flush()
    if not sections:  # no headings at all
        return [("", p) for p in recursive_split(text, size, overlap)]
    return sections


def semantic_split(
    text: str,
    embedder,
    threshold: float,
    max_size: int,
) -> List[str]:
    """Insert a boundary where adjacent-sentence similarity drops below threshold."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []
    vecs = embedder.embed(sentences)  # already L2-normalised
    chunks: List[str] = []
    current: List[str] = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))
        prospective = " ".join(current + [sentences[i]])
        if sim < threshold or len(prospective) > max_size:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    doc_id: str,
    text: str,
    *,
    strategy: Optional[str] = None,
    size: Optional[int] = None,
    overlap: Optional[int] = None,
    embedder=None,
) -> List[Chunk]:
    """Chunk a document into :class:`Chunk` objects with full metadata."""
    strategy = strategy or settings.chunk_strategy
    size = size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    text = text.strip()
    if not text:
        return []

    if strategy == "structure":
        pairs = structure_split(text, size, overlap)
    elif strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        pairs = [("", c) for c in semantic_split(text, embedder, settings.semantic_threshold, size)]
    else:  # recursive (default)
        pairs = [("", c) for c in recursive_split(text, size, overlap)]

    chunks: List[Chunk] = []
    cursor = 0
    for ordinal, (section, body) in enumerate(pairs):
        body = body.strip()
        if not body:
            continue
        start = text.find(body[:40], cursor)
        if start < 0:
            start = cursor
        end = start + len(body)
        cursor = max(cursor, start + 1)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                ordinal=ordinal,
                text=body,
                section=section,
                char_start=start,
                char_end=end,
            )
        )
    return chunks
