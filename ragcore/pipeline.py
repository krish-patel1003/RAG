"""End-to-end RAG query pipeline.

Ties the online path together and emits one trace per request:

    embed query -> hybrid retrieve -> rerank -> assemble prompt -> generate
                -> (sampled) LLM-as-judge -> persist trace

The returned object carries the answer, the fully-attributed sources, and the
trace id so the UI/API can link any answer back to exactly what was retrieved and
why.
"""

from __future__ import annotations

from typing import Optional

from .config import settings
from .evaluation import Judge
from .generator import Generator
from .retriever import Retriever
from .store import get_store
from .tracing import Trace, TraceStore


class RAGPipeline:
    def __init__(self, trace_store: Optional[TraceStore] = None) -> None:
        self.store = get_store()
        self.retriever = Retriever(store=self.store)
        self.generator = Generator()
        self.judge = Judge()
        self.traces = trace_store or TraceStore()

    def query(self, question: str, top_k: Optional[int] = None, judge: bool = True) -> dict:
        trace = Trace("rag_request")
        trace.set("question", question)
        trace.set("backend", settings.backend)

        chunks = self.retriever.retrieve(question, trace, top_k=top_k)
        result = self.generator.generate(question, chunks, trace)

        evaluation = None
        if judge:
            evaluation = self.judge.maybe_judge(question, result["answer"], chunks, trace)

        self.traces.save(trace)
        return {
            "trace_id": trace.trace_id,
            "question": question,
            "answer": result["answer"],
            "mode": result["mode"],
            "sources": result["sources"],
            "index_version": trace.attributes.get("index_version"),
            "evaluation": evaluation,
            "duration_ms": trace.to_dict()["duration_ms"],
        }
