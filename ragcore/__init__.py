"""Production-grade RAG core.

Public surface:

    from ragcore import Indexer, RAGPipeline

* :class:`~ragcore.indexer.Indexer`     -- the offline/indexing pipeline
* :class:`~ragcore.pipeline.RAGPipeline` -- the online/query pipeline
"""

from .config import settings
from .indexer import Indexer
from .ingest import Ingestor
from .pipeline import RAGPipeline

__all__ = ["settings", "Indexer", "Ingestor", "RAGPipeline"]
