"""Answer generation from retrieved context.

Uses Gemini when a key is configured; otherwise falls back to an *extractive*
answer (the top chunks verbatim) so the pipeline is always runnable offline.
The prompt forces grounding: answer only from the context, cite sources by their
bracketed index, and say "I don't know" when the context is insufficient.
"""

from __future__ import annotations

from typing import List, Tuple

from .config import settings
from .tracing import Trace
from .types import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY the "
    "numbered context passages. Cite the passages you use with their bracket "
    "number, e.g. [2]. If the context does not contain the answer, say you don't "
    "know based on the provided documents. Be concise and do not invent facts."
)


def build_context(chunks: List[RetrievedChunk]) -> Tuple[str, List[dict]]:
    lines, sources = [], []
    for i, c in enumerate(chunks, start=1):
        tag = f"{c.doc_id}" + (f" § {c.section}" if c.section else "")
        lines.append(f"[{i}] ({tag})\n{c.text}")
        sources.append(
            {
                "n": i, "doc_id": c.doc_id, "section": c.section,
                "chunk_vector_id": c.chunk_vector_id,
                "final_score": round(c.final_score, 4),
                "vector_score": None if c.vector_score is None else round(c.vector_score, 4),
                "bm25_score": None if c.bm25_score is None else round(c.bm25_score, 4),
                "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
                "preview": c.text[:200],
            }
        )
    return "\n\n".join(lines), sources


class Generator:
    def __init__(self) -> None:
        self.mode = "gemini" if (settings.embedder == "gemini" or settings.gemini_api_key) and settings.gemini_api_key else "extractive"
        self._client = None
        if self.mode == "gemini":
            try:
                from google import genai

                self._client = genai.Client(api_key=settings.gemini_api_key)
            except Exception:  # noqa: BLE001
                self.mode = "extractive"

    def generate(self, question: str, chunks: List[RetrievedChunk], trace: Trace) -> dict:
        context, sources = build_context(chunks)
        with trace.span("prompt.assembly") as sp:
            prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
            sp.set("num_chunks_used", len(chunks))
            sp.set("total_chars", len(prompt))

        if not chunks:
            return {"answer": "I don't know -- no documents matched this question.",
                    "sources": [], "mode": self.mode}

        with trace.span("llm.generate") as sp:
            sp.set("mode", self.mode)
            if self.mode == "gemini":
                from google.genai import types

                resp = self._client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                answer = (resp.text or "").strip() or "I don't know based on the provided documents."
                sp.set("model", settings.gemini_model)
            else:
                # extractive fallback: stitch the top passages
                answer = (
                    "(extractive mode -- no LLM configured) Most relevant passages:\n\n"
                    + "\n\n".join(f"[{i+1}] {c.text[:400]}" for i, c in enumerate(chunks[:3]))
                )
                sp.set("model", "extractive")
            sp.set("output_chars", len(answer))
        return {"answer": answer, "sources": sources, "mode": self.mode}
