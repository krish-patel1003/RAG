"""CLI for ad-hoc ingestion from the filesystem or the web research tool.

    # index local files
    python scripts/ingest.py fs ./docs --glob "**/*.md"

    # search (discovery only) — prints candidates
    python scripts/ingest.py search "hierarchical navigable small world" --mode papers

    # crawl + index specific URLs
    python scripts/ingest.py urls https://en.wikipedia.org/wiki/Okapi_BM25

    # one-shot: search a topic and ingest the top N results
    python scripts/ingest.py research "vector database indexing" --mode papers --top 3

All commands honour the RAG_* env vars (backend, embedder) and go through the
same Indexer as bulk ingestion.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragcore import Ingestor


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fs = sub.add_parser("fs", help="ingest local files")
    p_fs.add_argument("root")
    p_fs.add_argument("--glob", default="**/*")
    p_fs.add_argument("--no-recursive", action="store_true")

    p_se = sub.add_parser("search", help="search only (no indexing)")
    p_se.add_argument("query")
    p_se.add_argument("--mode", default="papers", choices=["papers", "wikis", "web"])
    p_se.add_argument("--limit", type=int, default=10)

    p_url = sub.add_parser("urls", help="crawl + index URLs")
    p_url.add_argument("urls", nargs="+")

    p_re = sub.add_parser("research", help="search a topic and ingest top N")
    p_re.add_argument("query")
    p_re.add_argument("--mode", default="papers", choices=["papers", "wikis", "web"])
    p_re.add_argument("--top", type=int, default=3)

    args = ap.parse_args()
    ing = Ingestor()

    if args.cmd == "fs":
        rep = ing.ingest_filesystem(args.root, glob=args.glob, recursive=not args.no_recursive)
        print(rep)
    elif args.cmd == "search":
        for r in ing.research_search(args.query, mode=args.mode, limit=args.limit):
            print(f"[{r.engine:<16}] {r.title[:70]}\n    {r.url}")
    elif args.cmd == "urls":
        print(ing.ingest_urls(args.urls))
    elif args.cmd == "research":
        hits = ing.research_search(args.query, mode=args.mode, limit=args.top)
        urls = [h.url for h in hits[: args.top]]
        print("ingesting:", urls)
        print(ing.ingest_urls(urls))


if __name__ == "__main__":
    main()
