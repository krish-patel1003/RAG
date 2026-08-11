"""CLI query against the indexed corpus.

    python scripts/query.py "What is HNSW and why is it used for retrieval?"

Prints the grounded answer, the attributed sources, and (if a key is set) the
faithfulness / relevance scores from the LLM judge.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragcore import RAGPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/query.py "your question"')
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    res = RAGPipeline().query(question)
    print("\n=== ANSWER ===")
    print(res["answer"])
    print(f"\nmode={res['mode']}  index={res['index_version']}  "
          f"latency={res['duration_ms']}ms  trace={res['trace_id']}")
    if res.get("evaluation"):
        ev = res["evaluation"]
        print(f"judge: faithfulness={ev.get('faithfulness')} "
              f"relevance={ev.get('answer_relevance')}")
    print("\n=== SOURCES ===")
    for s in res["sources"]:
        print(f"[{s['n']}] {s['doc_id']} § {s['section']}  "
              f"final={s['final_score']} vec={s['vector_score']} "
              f"bm25={s['bm25_score']} rerank={s['rerank_score']}")
        print(f"     {s['preview'][:140]}")


if __name__ == "__main__":
    main()
