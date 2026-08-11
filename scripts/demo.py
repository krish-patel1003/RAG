"""Two runnable demos over the two real-world corpora.

    python scripts/demo.py wikipedia
    python scripts/demo.py arxiv
    python scripts/demo.py both      # default

Each demo runs a set of curated questions and prints the grounded answer, the
top attributed source, and the judge's faithfulness/relevance scores. Assumes
the corpus has been indexed (see scripts/index_corpus.py or `make demo-data`).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragcore import RAGPipeline

QUESTIONS = {
    "wikipedia": [
        "What is HNSW and why is it used for nearest-neighbor search?",
        "How does BM25 differ from cosine similarity for retrieval?",
        "What problem does the CAP theorem describe in distributed systems?",
        "Why would you use an inverted index for full-text search?",
        "What is the difference between blue-green and canary deployments?",
    ],
    "arxiv": [
        "What techniques improve retrieval quality in retrieval-augmented generation?",
        "How do dense retrievers get trained?",
        "What are the trade-offs of long-context LLMs versus retrieval?",
        "How is reranking used to improve RAG accuracy?",
    ],
}


def run(dataset: str, rag: RAGPipeline) -> None:
    print(f"\n{'='*70}\n  DEMO: {dataset.upper()}\n{'='*70}")
    for q in QUESTIONS[dataset]:
        res = rag.query(q)
        ev = res.get("evaluation") or {}
        top = res["sources"][0] if res["sources"] else {}
        print(f"\nQ: {q}")
        print(f"A: {res['answer'].splitlines()[0][:200]}")
        print(
            f"   top source: {top.get('doc_id','-')} | "
            f"vec={top.get('vector_score')} bm25={top.get('bm25_score')} "
            f"rerank={top.get('rerank_score')}"
        )
        print(
            f"   judge: faithfulness={ev.get('faithfulness')} "
            f"relevance={ev.get('answer_relevance')} | trace={res['trace_id']}"
        )


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    rag = RAGPipeline()
    datasets = ["wikipedia", "arxiv"] if which == "both" else [which]
    for ds in datasets:
        run(ds, rag)


if __name__ == "__main__":
    main()
