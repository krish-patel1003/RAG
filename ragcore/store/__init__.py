"""Storage abstraction.

A :class:`Store` is both the **vector store** and the **document registry** from
the blog.  With pgvector the two live in one Postgres table (Postgres is both a
relational DB and a vector DB); with an external vector DB you would split them,
but the interface is identical either way.

Responsibilities:

* registry semantics -- map ``doc_id`` -> live chunk ids, content hashes,
  versions, and status (``active`` | ``superseded`` | ``deleted``);
* vector semantics   -- nearest-neighbour search with MVCC-style visibility
  filtering (``status='active' AND index_version=<alias> AND valid_from<=now``);
* the index alias    -- a ``meta`` row naming the current index version so a
  freshly-built shadow index can be swapped in atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ChunkRecord:
    chunk_vector_id: str
    doc_id: str
    ordinal: int
    text: str
    section: str
    content_hash: str
    embedding: np.ndarray
    embedding_model: str
    index_version: str
    valid_from: datetime
    status: str = "active"
    version: int = 1


@dataclass
class SearchHit:
    chunk_vector_id: str
    doc_id: str
    text: str
    section: str
    ordinal: int
    score: float                 # cosine similarity
    embedding_model: str
    index_version: str


class Store(Protocol):
    def init_schema(self, dim: int) -> None: ...

    # --- index alias ---------------------------------------------------
    def get_current_index_version(self) -> str: ...
    def set_current_index_version(self, version: str) -> None: ...

    # --- registry ------------------------------------------------------
    def doc_content_hash(self, doc_id: str) -> Optional[str]: ...
    def active_chunk_ids(self, doc_id: str, index_version: str) -> List[str]: ...
    def next_doc_version(self, doc_id: str) -> int: ...
    def supersede_document(self, doc_id: str, index_version: str) -> int: ...
    def mark_deleted(self, doc_id: str, index_version: str) -> int: ...

    # --- writes --------------------------------------------------------
    def add_chunks(self, records: Sequence[ChunkRecord]) -> List[str]: ...
    def delete_chunks(self, chunk_vector_ids: Sequence[str]) -> None: ...
    def gc_index_version(self, version: str) -> int: ...

    # --- reads ---------------------------------------------------------
    def vector_search(
        self,
        query_vec: np.ndarray,
        k: int,
        index_version: str,
        at_time: Optional[datetime] = None,
    ) -> List[SearchHit]: ...
    def active_corpus(self, index_version: str) -> List[Tuple[str, str]]: ...
    def get_chunks(self, chunk_vector_ids: Sequence[str]) -> dict: ...
    def stats(self) -> dict: ...
    def list_documents(self, limit: int = 100) -> List[dict]: ...


def get_store():
    from ..config import settings

    if settings.backend == "pgvector":
        from .pg_store import PgVectorStore

        return PgVectorStore(settings.database_url)
    if settings.backend == "sqlite":
        from .sqlite_store import SQLiteStore

        return SQLiteStore(settings.sqlite_path)
    raise ValueError(f"unknown backend: {settings.backend}")
