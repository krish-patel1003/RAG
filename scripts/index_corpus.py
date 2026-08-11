"""Index a JSONL corpus into the RAG store.

    python scripts/index_corpus.py data/wikipedia.jsonl

Honours all RAG_* env vars, so the same script indexes into pgvector (default)
or sqlite (RAG_BACKEND=sqlite).  Uses content-hash change detection, so re-running
it is cheap and idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragcore import Indexer
from ragcore.config import settings


def load(path: str):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                yield d["doc_id"], d["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    docs = list(load(args.path))
    if args.limit:
        docs = docs[: args.limit]
    print(f"Backend={settings.backend} embedder={settings.resolved_embedder()} "
          f"docs={len(docs)}")

    ix = Indexer()
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            rate = done / max(time.time() - t0, 1e-9)
            sys.stdout.write(f"\r  {done}/{total} docs ({rate:.1f}/s)")
            sys.stdout.flush()

    report = ix.index_corpus(docs, progress=progress)
    print(f"\nDone in {time.time()-t0:.1f}s: {report}")
    print("stats:", ix.store.stats())


if __name__ == "__main__":
    main()
