"""Source connectors for incremental ingestion.

A *connector* turns some external source (the local filesystem, a web page, a
search engine result) into :class:`LoadedDoc` objects. Connectors never touch
the index directly — they hand documents to :class:`~ragcore.ingest.Ingestor`,
which runs them through the same online indexing pipeline (chunk → embed →
registry → store) used for every other document. This keeps *incremental
ingestion* (adding documents at runtime) identical to bulk ingestion:
content-hash gated, versioned, and immediately queryable — no index rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LoadedDoc:
    """A document produced by a loader, ready to be indexed."""

    doc_id: str                     # stable id, source-prefixed (file:/web:/arxiv:)
    text: str
    title: str = ""
    source: str = ""                # 'filesystem' | 'web' | 'arxiv' | 'wikipedia'
    url: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A candidate from the research tool — shown to the user before ingestion."""

    title: str
    url: str
    snippet: str = ""
    engine: str = ""
    source: str = ""                # 'papers' | 'wikis' | 'web'
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title, "url": self.url, "snippet": self.snippet,
            "engine": self.engine, "source": self.source, "score": self.score,
        }
