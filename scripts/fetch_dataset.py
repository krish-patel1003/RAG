"""Fetch a real-world document corpus to a JSONL file.

Two sources, both plain HTTPS (no dataset library required):

* ``wikipedia`` -- full article extracts via the MediaWiki REST API across a
  curated, varied topic list.
* ``arxiv``     -- recent paper titles+abstracts via the arXiv Atom API.

Usage::

    python scripts/fetch_dataset.py --source wikipedia --limit 200 --out data/wikipedia.jsonl
    python scripts/fetch_dataset.py --source arxiv --query "retrieval augmented generation" --limit 200 --out data/arxiv.jsonl

Each line is ``{"doc_id": ..., "title": ..., "text": ..., "url": ...}``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Iterable, List
from xml.etree import ElementTree as ET

UA = "prod-rag-demo/1.0 (educational; contact: ira-tools-dev@csulb.edu)"

# Curated, varied seed topics so retrieval demos show real topical separation.
WIKI_TOPICS = [
    "Retrieval-augmented generation", "Vector database", "Cosine similarity",
    "Hierarchical navigable small world", "Nearest neighbor search", "Word embedding",
    "Large language model", "Transformer (deep learning)", "Attention (machine learning)",
    "BM25", "Tf-idf", "Information retrieval", "Approximate nearest neighbor",
    "Sentence embedding", "Semantic search", "Knowledge graph", "PostgreSQL",
    "Database index", "B-tree", "Hash table", "Cache (computing)", "Distributed computing",
    "MapReduce", "Apache Kafka", "Elasticsearch", "Sharding (database architecture)",
    "Consistency model", "CAP theorem", "Load balancing (computing)", "Microservices",
    "Docker (software)", "Kubernetes", "OpenTelemetry", "Observability",
    "Site reliability engineering", "Latency (engineering)", "Throughput",
    "Machine learning", "Deep learning", "Neural network", "Gradient descent",
    "Backpropagation", "Overfitting", "Cross-validation (statistics)", "Precision and recall",
    "Confusion matrix", "Reinforcement learning", "Natural language processing",
    "Named-entity recognition", "Question answering", "Text summarization",
    "Optical character recognition", "Speech recognition", "Computer vision",
    "Convolutional neural network", "Recurrent neural network", "Long short-term memory",
    "Generative adversarial network", "Diffusion model", "Autoencoder",
    "Principal component analysis", "K-means clustering", "Support vector machine",
    "Random forest", "Gradient boosting", "Decision tree learning", "Bayesian network",
    "Markov chain", "Hidden Markov model", "Monte Carlo method", "Linear regression",
    "Logistic regression", "Regularization (mathematics)", "Feature engineering",
    "Dimensionality reduction", "Anomaly detection", "Recommender system",
    "Collaborative filtering", "Cold start (recommender systems)", "A/B testing",
    "Data warehouse", "Data lake", "ETL", "Stream processing", "Batch processing",
    "Message queue", "Publish–subscribe pattern", "Event-driven architecture",
    "REST", "GraphQL", "gRPC", "JSON", "Protocol Buffers", "WebSocket",
    "Transport Layer Security", "Public-key cryptography", "Hash function",
    "SHA-2", "Digital signature", "OAuth", "JSON Web Token", "Rate limiting",
    "Content delivery network", "Domain Name System", "HTTP", "TCP",
    "Garbage collection (computer science)", "Concurrency (computer science)",
    "Thread (computing)", "Lock (computer science)", "Deadlock", "Race condition",
    "Functional programming", "Object-oriented programming", "Type system",
    "Compiler", "Interpreter (computing)", "Virtual machine", "Operating system",
    "File system", "Relational database", "NoSQL", "ACID", "Database transaction",
    "Query optimization", "Materialized view", "Full-text search", "Inverted index",
    "Bloom filter", "Consistent hashing", "Raft (algorithm)", "Paxos (computer science)",
    "Two-phase commit protocol", "Idempotence", "Circuit breaker design pattern",
    "Exponential backoff", "Chaos engineering", "Feature flag", "Blue-green deployment",
    "Canary release", "Continuous integration", "Continuous delivery", "Version control",
    "Git", "Infrastructure as code", "Terraform (software)", "Serverless computing",
    "Edge computing", "Quantum computing", "Graph database", "Time series database",
    "Columnar storage", "Data compression", "Huffman coding", "Run-length encoding",
    "Reservoir sampling", "Locality-sensitive hashing", "Product quantization",
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_wikipedia(limit: int) -> Iterable[dict]:
    n = 0
    for topic in WIKI_TOPICS:
        if n >= limit:
            break
        slug = urllib.parse.quote(topic.replace(" ", "_"))
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&format=json"
            "&prop=extracts&explaintext=1&redirects=1&titles=" + slug
        )
        try:
            data = json.loads(_get(url))
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                text = (page.get("extract") or "").strip()
                if len(text) < 400:
                    continue
                yield {
                    "doc_id": "wiki:" + page.get("title", topic).replace(" ", "_"),
                    "title": page.get("title", topic),
                    "text": text,
                    "url": f"https://en.wikipedia.org/wiki/{slug}",
                }
                n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip {topic}: {exc}")
        time.sleep(0.1)  # be polite to the API


def fetch_arxiv(query: str, limit: int) -> Iterable[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    page = 100
    fetched = 0
    for start in range(0, limit, page):
        url = (
            "http://export.arxiv.org/api/query?search_query="
            + urllib.parse.quote(f"all:{query}")
            + f"&start={start}&max_results={min(page, limit - fetched)}"
            "&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            root = ET.fromstring(_get(url))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! arxiv page {start}: {exc}")
            break
        entries = root.findall("a:entry", ns)
        if not entries:
            break
        for e in entries:
            aid = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
            title = re.sub(r"\s+", " ", (e.findtext("a:title", default="", namespaces=ns) or "")).strip()
            summary = re.sub(r"\s+", " ", (e.findtext("a:summary", default="", namespaces=ns) or "")).strip()
            if len(summary) < 200:
                continue
            short = aid.rsplit("/", 1)[-1]
            yield {
                "doc_id": "arxiv:" + short,
                "title": title,
                "text": f"{title}\n\n{summary}",
                "url": aid,
            }
            fetched += 1
        time.sleep(3)  # arXiv API asks for >=3s between calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wikipedia", "arxiv"], default="wikipedia")
    ap.add_argument("--query", default="retrieval augmented generation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="data/corpus.jsonl")
    args = ap.parse_args()

    it = (
        fetch_wikipedia(args.limit)
        if args.source == "wikipedia"
        else fetch_arxiv(args.query, args.limit)
    )
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count = 0
    with open(args.out, "w") as fh:
        for doc in it:
            fh.write(json.dumps(doc) + "\n")
            count += 1
            if count % 20 == 0:
                print(f"  fetched {count}...")
    print(f"Wrote {count} documents to {args.out}")


if __name__ == "__main__":
    main()
