"""SQLite store -- zero-dependency backend for tests and laptop runs.

Same interface as the pgvector store.  Vectors are stored as float32 blobs and
nearest-neighbour search is exact brute force with numpy: fine up to ~100-200k
chunks, which is plenty for the test suite and small demos.  For real scale use
the pgvector backend (HNSW).  The MVCC-style visibility filter
(status/index_version/valid_from) is identical so behaviour matches production.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import ChunkRecord, SearchHit, utcnow


def _iso(dt: datetime) -> str:
    return dt.astimezone().isoformat()


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_schema(self, dim: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_vector_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    section TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    version INTEGER NOT NULL DEFAULT 1,
                    indexed_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_ver_idx ON chunks(index_version, status)")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('current_index_version', 'v1')"
            )

    # --- index alias ---------------------------------------------------
    def get_current_index_version(self) -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='current_index_version'").fetchone()
            return row["value"] if row else "v1"

    def set_current_index_version(self, version: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('current_index_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (version,),
            )

    # --- registry ------------------------------------------------------
    def doc_content_hash(self, doc_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content_hash FROM chunks WHERE doc_id=? AND status='active' "
                "ORDER BY ordinal LIMIT 1",
                (doc_id,),
            ).fetchone()
            return row["content_hash"] if row else None

    def active_chunk_ids(self, doc_id: str, index_version: str) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_vector_id FROM chunks WHERE doc_id=? AND status='active' AND index_version=?",
                (doc_id, index_version),
            ).fetchall()
            return [r["chunk_vector_id"] for r in rows]

    def next_doc_version(self, doc_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version),0) AS v FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchone()
            return int(row["v"]) + 1

    def supersede_document(self, doc_id: str, index_version: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE chunks SET status='superseded' "
                "WHERE doc_id=? AND status='active' AND index_version=?",
                (doc_id, index_version),
            )
            return cur.rowcount

    def mark_deleted(self, doc_id: str, index_version: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE chunks SET status='deleted' "
                "WHERE doc_id=? AND status='active' AND index_version=?",
                (doc_id, index_version),
            )
            return cur.rowcount

    # --- writes --------------------------------------------------------
    def add_chunks(self, records: Sequence[ChunkRecord]) -> List[str]:
        if not records:
            return []
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO chunks (chunk_vector_id, doc_id, ordinal, text, section,
                    content_hash, embedding, dim, embedding_model, index_version, valid_from,
                    status, version, indexed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r.chunk_vector_id, r.doc_id, r.ordinal, r.text, r.section,
                        r.content_hash, r.embedding.astype(np.float32).tobytes(),
                        int(r.embedding.shape[0]), r.embedding_model, r.index_version,
                        _iso(r.valid_from), r.status, r.version, _iso(utcnow()),
                    )
                    for r in records
                ],
            )
        return [r.chunk_vector_id for r in records]

    def delete_chunks(self, chunk_vector_ids: Sequence[str]) -> None:
        if not chunk_vector_ids:
            return
        with self._conn() as conn:
            conn.executemany(
                "DELETE FROM chunks WHERE chunk_vector_id=?",
                [(cid,) for cid in chunk_vector_ids],
            )

    def gc_index_version(self, version: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE index_version=?", (version,))
            return cur.rowcount

    # --- reads ---------------------------------------------------------
    def _load_active(self, index_version: str, at_time: datetime):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_vector_id, doc_id, text, section, ordinal, embedding, dim, "
                "embedding_model, index_version, valid_from FROM chunks "
                "WHERE status='active' AND index_version=?",
                (index_version,),
            ).fetchall()
        cutoff = _iso(at_time)
        return [r for r in rows if r["valid_from"] <= cutoff]

    def vector_search(
        self,
        query_vec: np.ndarray,
        k: int,
        index_version: str,
        at_time: Optional[datetime] = None,
    ) -> List[SearchHit]:
        at_time = at_time or utcnow()
        rows = self._load_active(index_version, at_time)
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        sims = mat @ query_vec.astype(np.float32)   # rows are L2-normalised -> cosine
        top = np.argsort(-sims)[:k]
        return [
            SearchHit(
                chunk_vector_id=rows[i]["chunk_vector_id"], doc_id=rows[i]["doc_id"],
                text=rows[i]["text"], section=rows[i]["section"], ordinal=rows[i]["ordinal"],
                score=float(sims[i]), embedding_model=rows[i]["embedding_model"],
                index_version=rows[i]["index_version"],
            )
            for i in top
        ]

    def active_corpus(self, index_version: str) -> List[Tuple[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_vector_id, text FROM chunks WHERE status='active' AND index_version=?",
                (index_version,),
            ).fetchall()
            return [(r["chunk_vector_id"], r["text"]) for r in rows]

    def get_chunks(self, chunk_vector_ids: Sequence[str]) -> dict:
        if not chunk_vector_ids:
            return {}
        qmarks = ",".join("?" for _ in chunk_vector_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT chunk_vector_id, doc_id, text, section, ordinal, embedding_model, "
                f"index_version FROM chunks WHERE chunk_vector_id IN ({qmarks})",
                list(chunk_vector_ids),
            ).fetchall()
            return {r["chunk_vector_id"]: dict(r) for r in rows}

    def stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(status='active')     AS active_chunks,
                    SUM(status='superseded') AS superseded_chunks,
                    SUM(status='deleted')    AS deleted_chunks,
                    COUNT(DISTINCT CASE WHEN status='active' THEN doc_id END) AS active_docs
                FROM chunks
                """
            ).fetchone()
            return {
                "backend": "sqlite",
                "current_index_version": self.get_current_index_version(),
                "active_chunks": int(row["active_chunks"] or 0),
                "superseded_chunks": int(row["superseded_chunks"] or 0),
                "deleted_chunks": int(row["deleted_chunks"] or 0),
                "active_docs": int(row["active_docs"] or 0),
            }

    def list_documents(self, limit: int = 100) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, COUNT(*) AS chunks, MAX(version) AS version,
                       MAX(indexed_at) AS indexed_at, MAX(embedding_model) AS embedding_model
                FROM chunks WHERE status='active'
                GROUP BY doc_id ORDER BY MAX(indexed_at) DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
