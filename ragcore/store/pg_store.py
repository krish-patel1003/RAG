"""Postgres + pgvector store -- the default, scale-oriented backend.

One table (`chunks`) serves as both the vector store and the document registry,
which is natural with pgvector because Postgres is both a relational and a vector
database.  With an external vector DB (Qdrant/Pinecone) you would keep this same
table in Postgres minus the `embedding` column and let the vector DB hold the
vectors keyed by `chunk_vector_id`; the interface would not change.

Approximate nearest-neighbour search uses an HNSW index
(`vector_cosine_ops`), giving O(log n) retrieval that scales to millions of
vectors with a tunable recall/latency trade-off via `hnsw.ef_search`.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import psycopg
from psycopg.rows import dict_row

from ..config import settings
from . import ChunkRecord, SearchHit, utcnow


def _vec_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


class PgVectorStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextmanager
    def _conn(self):
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            yield conn

    def init_schema(self, dim: int) -> None:
        with self._conn() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_vector_id TEXT PRIMARY KEY,
                    doc_id          TEXT NOT NULL,
                    ordinal         INTEGER NOT NULL,
                    text            TEXT NOT NULL,
                    section         TEXT NOT NULL DEFAULT '',
                    content_hash    TEXT NOT NULL,
                    embedding       vector({dim}) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    index_version   TEXT NOT NULL,
                    valid_from      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    status          TEXT NOT NULL DEFAULT 'active',
                    version         INTEGER NOT NULL DEFAULT 1,
                    indexed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id, status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_ver_idx ON chunks (index_version, status)"
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = {settings.hnsw_m}, ef_construction = {settings.hnsw_ef_construction})
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('current_index_version', 'v1') "
                "ON CONFLICT (key) DO NOTHING"
            )
            conn.commit()

    # --- index alias ---------------------------------------------------
    def get_current_index_version(self) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='current_index_version'"
            ).fetchone()
            return row["value"] if row else "v1"

    def set_current_index_version(self, version: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('current_index_version', %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (version,),
            )
            conn.commit()

    # --- registry ------------------------------------------------------
    def doc_content_hash(self, doc_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content_hash FROM chunks WHERE doc_id=%s AND status='active' "
                "ORDER BY ordinal LIMIT 1",
                (doc_id,),
            ).fetchone()
            return row["content_hash"] if row else None

    def active_chunk_ids(self, doc_id: str, index_version: str) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_vector_id FROM chunks "
                "WHERE doc_id=%s AND status='active' AND index_version=%s",
                (doc_id, index_version),
            ).fetchall()
            return [r["chunk_vector_id"] for r in rows]

    def next_doc_version(self, doc_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM chunks WHERE doc_id=%s",
                (doc_id,),
            ).fetchone()
            return int(row["v"]) + 1

    def supersede_document(self, doc_id: str, index_version: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE chunks SET status='superseded' "
                "WHERE doc_id=%s AND status='active' AND index_version=%s",
                (doc_id, index_version),
            )
            conn.commit()
            return cur.rowcount

    def mark_deleted(self, doc_id: str, index_version: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE chunks SET status='deleted' "
                "WHERE doc_id=%s AND status='active' AND index_version=%s",
                (doc_id, index_version),
            )
            conn.commit()
            return cur.rowcount

    # --- writes --------------------------------------------------------
    def add_chunks(self, records: Sequence[ChunkRecord]) -> List[str]:
        if not records:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (chunk_vector_id, doc_id, ordinal, text, section,
                        content_hash, embedding, embedding_model, index_version,
                        valid_from, status, version)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_vector_id) DO NOTHING
                    """,
                    [
                        (
                            r.chunk_vector_id, r.doc_id, r.ordinal, r.text, r.section,
                            r.content_hash, _vec_literal(r.embedding), r.embedding_model,
                            r.index_version, r.valid_from, r.status, r.version,
                        )
                        for r in records
                    ],
                )
            conn.commit()
        return [r.chunk_vector_id for r in records]

    def delete_chunks(self, chunk_vector_ids: Sequence[str]) -> None:
        if not chunk_vector_ids:
            return
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE chunk_vector_id = ANY(%s)",
                (list(chunk_vector_ids),),
            )
            conn.commit()

    def gc_index_version(self, version: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE index_version=%s", (version,))
            conn.commit()
            return cur.rowcount

    # --- reads ---------------------------------------------------------
    def vector_search(
        self,
        query_vec: np.ndarray,
        k: int,
        index_version: str,
        at_time: Optional[datetime] = None,
    ) -> List[SearchHit]:
        at_time = at_time or utcnow()
        lit = _vec_literal(query_vec)
        with self._conn() as conn:
            conn.execute(f"SET hnsw.ef_search = {settings.hnsw_ef_search}")
            rows = conn.execute(
                """
                SELECT chunk_vector_id, doc_id, text, section, ordinal,
                       embedding_model, index_version,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE status='active' AND index_version=%s AND valid_from <= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (lit, index_version, at_time, lit, k),
            ).fetchall()
        return [
            SearchHit(
                chunk_vector_id=r["chunk_vector_id"], doc_id=r["doc_id"],
                text=r["text"], section=r["section"], ordinal=r["ordinal"],
                score=float(r["score"]), embedding_model=r["embedding_model"],
                index_version=r["index_version"],
            )
            for r in rows
        ]

    def active_corpus(self, index_version: str) -> List[Tuple[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_vector_id, text FROM chunks "
                "WHERE status='active' AND index_version=%s",
                (index_version,),
            ).fetchall()
            return [(r["chunk_vector_id"], r["text"]) for r in rows]

    def get_chunks(self, chunk_vector_ids: Sequence[str]) -> dict:
        if not chunk_vector_ids:
            return {}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_vector_id, doc_id, text, section, ordinal, "
                "embedding_model, index_version FROM chunks "
                "WHERE chunk_vector_id = ANY(%s)",
                (list(chunk_vector_ids),),
            ).fetchall()
            return {r["chunk_vector_id"]: dict(r) for r in rows}

    def stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status='active')     AS active_chunks,
                    COUNT(*) FILTER (WHERE status='superseded') AS superseded_chunks,
                    COUNT(*) FILTER (WHERE status='deleted')    AS deleted_chunks,
                    COUNT(DISTINCT doc_id) FILTER (WHERE status='active') AS active_docs
                FROM chunks
                """
            ).fetchone()
            return {
                "backend": "pgvector",
                "current_index_version": self.get_current_index_version(),
                **{k: int(v or 0) for k, v in row.items()},
            }

    def list_documents(self, limit: int = 100) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT doc_id,
                       COUNT(*)         AS chunks,
                       MAX(version)     AS version,
                       MAX(indexed_at)  AS indexed_at,
                       MAX(embedding_model) AS embedding_model
                FROM chunks WHERE status='active'
                GROUP BY doc_id ORDER BY MAX(indexed_at) DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
