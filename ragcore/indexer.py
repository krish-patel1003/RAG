"""The indexing pipeline.

Implements the operations tutorials skip and production needs:

* **content-hash change detection** -- skip documents whose text is unchanged;
  most "updates" are metadata-only and must not trigger re-embedding.
* **correct reindex/delete semantics** -- an update is *delete the old chunk
  ids, then insert the (possibly different number of) new ones*, not an in-place
  row update; the registry maps ``doc_id`` -> chunk ids so we can do this.
* **index versioning + alias swap** -- an embedding-model upgrade or full
  rebuild is done under a fresh ``index_version`` (a shadow index), validated,
  then the alias is flipped atomically so no query ever sees a partial index.
* **model-lock metadata** -- every chunk stores the embedding model name; the
  retriever asserts it matches the query model before returning results.
"""

from __future__ import annotations

import uuid
from typing import Callable, List, Optional

from .chunking import chunk_document
from .config import settings
from .embeddings import Embedder, get_embedder
from .store import ChunkRecord, Store, get_store, utcnow
from .types import content_hash


class Indexer:
    def __init__(self, store: Optional[Store] = None, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or get_embedder()
        self.store = store or get_store()
        self.store.init_schema(self.embedder.dim)

    # -- change detection ----------------------------------------------
    def should_reindex(self, doc_id: str, new_content: str) -> bool:
        existing = self.store.doc_content_hash(doc_id)
        if existing is None:
            return True  # new document
        return existing != content_hash(new_content)

    # -- single document ------------------------------------------------
    def index_document(
        self,
        doc_id: str,
        content: str,
        *,
        index_version: Optional[str] = None,
        force: bool = False,
        valid_from=None,
    ) -> dict:
        """Insert or update one document.  Returns a small report."""
        version = index_version or self.store.get_current_index_version()

        if not force and not self.should_reindex(doc_id, content):
            return {"doc_id": doc_id, "skipped": True, "reason": "unchanged_content"}

        # 1. find + 2. delete existing chunks for this doc (blog's reindex flow)
        old_ids = self.store.active_chunk_ids(doc_id, version)
        deleted = 0
        if old_ids:
            self.store.supersede_document(doc_id, version)
            self.store.delete_chunks(old_ids)
            deleted = len(old_ids)

        # 3. re-chunk + re-embed
        chunks = chunk_document(
            doc_id, content,
            embedder=self.embedder if settings.chunk_strategy == "semantic" else None,
        )
        report = {"doc_id": doc_id, "skipped": False, "deleted": deleted, "chunks": 0}
        if not chunks:
            return report

        embeddings = self.embedder.embed([c.text for c in chunks])
        doc_version = self.store.next_doc_version(doc_id)
        chash = content_hash(content)
        records: List[ChunkRecord] = []
        for chunk, emb in zip(chunks, embeddings):
            records.append(
                ChunkRecord(
                    chunk_vector_id=uuid.uuid4().hex,
                    doc_id=doc_id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    section=chunk.section,
                    content_hash=chash,
                    embedding=emb,
                    embedding_model=self.embedder.name,
                    index_version=version,
                    valid_from=valid_from or utcnow(),
                    version=doc_version,
                )
            )
        self.store.add_chunks(records)
        report["chunks"] = len(records)
        report["index_version"] = version
        return report

    def delete_document(self, doc_id: str) -> dict:
        version = self.store.get_current_index_version()
        n = self.store.mark_deleted(doc_id, version)
        return {"doc_id": doc_id, "deleted_chunks": n}

    # -- bulk -----------------------------------------------------------
    def index_corpus(
        self,
        docs,
        *,
        index_version: Optional[str] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """docs: iterable of (doc_id, content)."""
        docs = list(docs)
        total = len(docs)
        indexed = skipped = chunks = 0
        for i, (doc_id, content) in enumerate(docs):
            rep = self.index_document(doc_id, content, index_version=index_version)
            if rep.get("skipped"):
                skipped += 1
            else:
                indexed += 1
                chunks += rep.get("chunks", 0)
            if progress:
                progress(i + 1, total)
        return {"documents": total, "indexed": indexed, "skipped": skipped, "chunks": chunks}

    # -- embedding-model upgrade (shadow index + alias swap) -----------
    def rebuild_shadow(self, docs, new_version: str) -> dict:
        """Build a shadow index under ``new_version`` without touching the live
        alias.  Callers validate it, then call :meth:`promote`."""
        report = self.index_corpus(docs, index_version=new_version)
        report["index_version"] = new_version
        return report

    def promote(self, new_version: str, gc_old: bool = False) -> dict:
        """Atomically flip the alias to ``new_version`` (zero-downtime deploy)."""
        old = self.store.get_current_index_version()
        self.store.set_current_index_version(new_version)
        removed = self.store.gc_index_version(old) if gc_old and old != new_version else 0
        return {"promoted_to": new_version, "previous": old, "gc_removed": removed}
