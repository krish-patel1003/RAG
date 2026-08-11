"""FastAPI serving layer for the production RAG system.

Thin HTTP wrapper over ``ragcore``.  The heavy objects (pipeline, indexer) are
built once at startup and reused.  Every ``/query`` response includes a
``trace_id`` you can look up via ``/traces/{id}`` to see chunk-level attribution.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragcore import Indexer, Ingestor, RAGPipeline
from ragcore.config import settings
from ragcore.tracing import TraceStore

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["indexer"] = Indexer()
    STATE["ingestor"] = Ingestor(indexer=STATE["indexer"])
    STATE["pipeline"] = RAGPipeline()
    STATE["traces"] = TraceStore()
    yield


app = FastAPI(title="Production RAG", version="1.0", lifespan=lifespan)


# ---- schemas ----------------------------------------------------------
class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    force: bool = False


class BulkDocumentsIn(BaseModel):
    documents: List[DocumentIn]


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)
    judge: bool = True


class FilesystemIn(BaseModel):
    root: str = Field(min_length=1)
    glob: str = "**/*"
    recursive: bool = True


class ResearchIn(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="papers")  # papers | wikis | web
    limit: int = Field(default=10, ge=1, le=50)


class IngestUrlsIn(BaseModel):
    urls: List[str] = Field(min_length=1)


# ---- endpoints --------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": settings.backend,
            "embedder": settings.resolved_embedder()}


@app.get("/stats")
def stats() -> dict:
    return STATE["indexer"].store.stats()


@app.get("/documents")
def list_documents(limit: int = 100) -> dict:
    return {"documents": STATE["indexer"].store.list_documents(limit)}


@app.post("/documents")
def add_document(doc: DocumentIn) -> dict:
    try:
        return STATE["indexer"].index_document(doc.doc_id, doc.text, force=doc.force)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@app.post("/documents/bulk")
def add_documents(body: BulkDocumentsIn) -> dict:
    ix = STATE["indexer"]
    return ix.index_corpus([(d.doc_id, d.text) for d in body.documents])


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    return STATE["indexer"].delete_document(doc_id)


@app.post("/ingest/filesystem")
def ingest_filesystem(body: FilesystemIn) -> dict:
    try:
        return STATE["ingestor"].ingest_filesystem(
            body.root, glob=body.glob, recursive=body.recursive
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@app.post("/research/search")
def research_search(body: ResearchIn) -> dict:
    """Discovery only — returns candidates for the user to review. No indexing."""
    try:
        results = STATE["ingestor"].research_search(
            body.query, mode=body.mode, limit=body.limit
        )
        return {"query": body.query, "mode": body.mode,
                "results": [r.to_dict() for r in results]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"search failed: {exc}") from exc


@app.post("/research/ingest")
def research_ingest(body: IngestUrlsIn) -> dict:
    """Crawl + extract the selected URLs and add them to the corpus."""
    try:
        return STATE["ingestor"].ingest_urls(body.urls)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@app.post("/query")
def query(req: QueryIn) -> dict:
    try:
        return STATE["pipeline"].query(req.question, top_k=req.top_k, judge=req.judge)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc


@app.get("/traces")
def recent_traces(limit: int = 50) -> dict:
    return {"traces": STATE["traces"].recent(limit)}


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    tr = STATE["traces"].get(trace_id)
    if not tr:
        raise HTTPException(404, "trace not found")
    return tr


@app.get("/traces/quality/low")
def low_quality(threshold: float = 0.7, days: int = 7) -> dict:
    return {"threshold": threshold, "days": days,
            "traces": STATE["traces"].low_quality(threshold, days)}
