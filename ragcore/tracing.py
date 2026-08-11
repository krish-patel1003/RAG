"""Lightweight RAG-aware tracing.

OpenTelemetry's generic span model doesn't natively capture RAG primitives
(which chunks were retrieved, with what scores, from which index version), so we
instrument them explicitly.  A request produces a nested span tree:

    rag_request (root)
      |- embedding.query
      |- retrieval.vector_search   (+ chunk_retrieved events)
      |- retrieval.bm25
      |- retrieval.fuse
      |- retrieval.rerank
      |- prompt.assembly
      |- llm.generate
      |- eval.judge

Traces are written to a SQLite table AND appended as JSONL so they can be shipped
to any backend (this maps cleanly onto OTel spans/attributes/events if you want
to export -- see ARCHITECTURE.md).  Every retrieval span carries the index
version so a quality regression can be correlated to an index update.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import settings


@dataclass
class Span:
    name: str
    span_id: str
    start: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def event(self, name: str, payload: Dict[str, Any]) -> None:
        self.events.append({"name": name, **payload})


class Trace:
    def __init__(self, name: str = "rag_request") -> None:
        self.trace_id = uuid.uuid4().hex
        self.name = name
        self.spans: List[Span] = []
        self.attributes: Dict[str, Any] = {}
        self._start = time.perf_counter()

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    @contextmanager
    def span(self, name: str):
        sp = Span(name=name, span_id=uuid.uuid4().hex, start=time.perf_counter())
        try:
            yield sp
        finally:
            sp.duration_ms = round((time.perf_counter() - sp.start) * 1000, 2)
            self.spans.append(sp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "attributes": self.attributes,
            "duration_ms": round((time.perf_counter() - self._start) * 1000, 2),
            "spans": [
                {
                    "name": s.name,
                    "span_id": s.span_id,
                    "duration_ms": s.duration_ms,
                    "attributes": s.attributes,
                    "events": s.events,
                }
                for s in self.spans
            ],
        }


class TraceStore:
    """Persists traces to SQLite (queryable) and JSONL (shippable)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.trace_db_path
        if os.path.dirname(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.jsonl_path = os.path.splitext(self.db_path)[0] + ".jsonl"
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    question TEXT,
                    index_version TEXT,
                    num_chunks INTEGER,
                    faithfulness REAL,
                    answer_relevance REAL,
                    duration_ms REAL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS traces_time ON traces(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS traces_faith ON traces(faithfulness)")

    def save(self, trace: Trace) -> None:
        payload = trace.to_dict()
        attrs = payload["attributes"]
        row = (
            trace.trace_id,
            time.time(),
            attrs.get("question"),
            attrs.get("index_version"),
            attrs.get("num_chunks"),
            attrs.get("faithfulness"),
            attrs.get("answer_relevance"),
            payload["duration_ms"],
            json.dumps(payload),
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, created_at, question, index_version, "
                "num_chunks, faithfulness, answer_relevance, duration_ms, payload) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                row,
            )
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(payload) + "\n")

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT trace_id, created_at, question, index_version, num_chunks, "
                "faithfulness, answer_relevance, duration_ms FROM traces "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT payload FROM traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
            return json.loads(row["payload"]) if row else None

    def low_quality(self, threshold: float = 0.7, days: int = 7) -> List[Dict[str, Any]]:
        since = time.time() - days * 86400
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT trace_id, question, faithfulness, answer_relevance, index_version "
                "FROM traces WHERE created_at >= ? AND faithfulness IS NOT NULL "
                "AND faithfulness < ? ORDER BY faithfulness ASC",
                (since, threshold),
            ).fetchall()
            return [dict(r) for r in rows]
