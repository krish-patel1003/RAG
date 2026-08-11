"""Ingestion service — the single entry point for ad-hoc document addition.

Whatever the source (filesystem, a crawled URL, a research result), a document
ends up here as a :class:`~ragcore.loaders.LoadedDoc` and is pushed through the
**same online indexing pipeline** every other document uses:

    LoadedDoc → Indexer.index_document → content-hash gate → chunk → embed →
                registry (doc_id → chunk ids, version) → vector store (valid_from=now)

Because it reuses :class:`~ragcore.indexer.Indexer`, ad-hoc ingestion inherits
all of its guarantees for free: unchanged docs are skipped, updates delete the
old chunks before inserting new ones, every chunk is tagged with the embedding
model and current index version, and the new content is queryable immediately
(``valid_from = now``). No separate code path, no index rebuild.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .indexer import Indexer
from .loaders import LoadedDoc, SearchResult
from .loaders.filesystem import load_filesystem
from .loaders.web import SearXNGClient, load_urls


class Ingestor:
    def __init__(self, indexer: Optional[Indexer] = None) -> None:
        self.indexer = indexer or Indexer()

    # -- generic: index a batch of already-loaded documents -------------
    def ingest_docs(self, docs: Iterable[LoadedDoc]) -> dict:
        docs = list(docs)
        indexed = skipped = chunks = 0
        details: List[dict] = []
        for d in docs:
            rep = self.indexer.index_document(d.doc_id, d.text)
            rep["title"] = d.title
            rep["source"] = d.source
            rep["url"] = d.url
            details.append(rep)
            if rep.get("skipped"):
                skipped += 1
            else:
                indexed += 1
                chunks += rep.get("chunks", 0)
        return {
            "documents": len(docs), "indexed": indexed, "skipped": skipped,
            "chunks": chunks, "details": details,
        }

    # -- filesystem source ----------------------------------------------
    def ingest_filesystem(
        self, root: str, *, glob: str = "**/*", recursive: bool = True
    ) -> dict:
        docs = load_filesystem(root, glob=glob, recursive=recursive)
        report = self.ingest_docs(docs)
        report["source"] = "filesystem"
        report["root"] = root
        return report

    # -- research: search (no indexing) then ingest selected URLs -------
    def research_search(
        self, query: str, mode: str = "papers", limit: int = 10
    ) -> List[SearchResult]:
        return SearXNGClient().search(query, mode=mode, limit=limit)

    def ingest_urls(self, urls: List[str]) -> dict:
        docs = load_urls(urls)
        report = self.ingest_docs(docs)
        report["source"] = "web"
        report["requested"] = len(urls)
        report["fetched"] = len(docs)
        return report
