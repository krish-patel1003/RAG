# Production RAG

A production-grade Retrieval-Augmented Generation system — the layer *after* the
demo. It implements the three things that separate a tutorial from a system that
survives production (per the reference blog post):

1. **Indexing pipeline** — document registry, content-hash change detection,
   correct delete/reindex semantics, index versioning + alias-based zero-downtime
   deploys, and an embedding-model-lock guard.
2. **Retrieval layer** — hybrid search (vector ANN + BM25) fused with Reciprocal
   Rank Fusion, then cross-encoder reranking.
3. **Observability** — per-request traces with **chunk-level attribution**,
   LLM-as-judge faithfulness/relevance scoring, and `index_version` on every
   retrieval span so quality regressions can be traced to an index update.

Designed to scale to **1M documents** (~12M chunks) on pgvector HNSW; see
[ARCHITECTURE.md](ARCHITECTURE.md) for the full design, diagrams, and the
scaling topology.

```
ragcore/            the library (backend-agnostic)
  chunking.py       recursive / semantic / structure-aware chunking
  embeddings.py     Gemini + deterministic hashing embedders (model-lock)
  store/            Store protocol · PgVectorStore (HNSW) · SQLiteStore
  indexer.py        registry, content-hash gate, reindex, alias swap
  bm25.py           dependency-free BM25 Okapi
  retriever.py      hybrid retrieval + RRF + rerank + attribution
  reranker.py       cross-encoder (ms-marco-MiniLM), noop fallback
  generator.py      grounded Gemini generation (+ extractive fallback)
  evaluation.py     LLM-as-judge (faithfulness / relevance)
  tracing.py        RAG-aware span tree → SQLite + JSONL
  pipeline.py       end-to-end query orchestration
app/main.py         FastAPI service
ui/streamlit_app.py Streamlit demo UI
scripts/            fetch_dataset · index_corpus · query · demo
tests/              offline pipeline tests (sqlite + hashing, no network)
```

## Quick start (Docker — the full stack)

```bash
cp .env.example .env      # then set GEMINI_API_KEY
docker compose up --build
```

* API → http://localhost:8000  (docs at `/docs`)
* UI  → http://localhost:8501

Postgres+pgvector is exposed on host port **5433** (5432 is often taken by a
native Postgres).

## Quick start (local, no Docker)

```bash
pip install -r requirements.txt
docker compose up -d db                 # just the pgvector database
export $(grep -v '^#' .env | xargs)     # load GEMINI_API_KEY etc.
export DATABASE_URL=postgresql://rag:rag@localhost:5433/rag

# 1. fetch real-world corpora
python scripts/fetch_dataset.py --source wikipedia --limit 150 --out data/wikipedia.jsonl
python scripts/fetch_dataset.py --source arxiv --limit 80 --out data/arxiv.jsonl

# 2. index them (content-hash gated, idempotent)
python scripts/index_corpus.py data/wikipedia.jsonl
python scripts/index_corpus.py data/arxiv.jsonl

# 3. ask
python scripts/query.py "What is HNSW and why is it used for retrieval?"

# 4. run the two curated demos
python scripts/demo.py both

# 5. UI (talks to the API)
uvicorn app.main:app --port 8000 &
RAG_API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

## Run it fully offline (no key, no Docker)

The SQLite backend + deterministic hashing embedder run the whole pipeline
without a database, network, or API key — this is also what the tests use:

```bash
RAG_BACKEND=sqlite RAG_EMBEDDER=hashing python scripts/query.py "battery life"
python -m pytest
```

## Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `RAG_BACKEND` | `pgvector` | `pgvector` \| `sqlite` |
| `RAG_EMBEDDER` | `gemini` | `gemini` \| `hashing` (auto-downgrades if no key) |
| `DATABASE_URL` | `postgresql://rag:rag@db:5432/rag` | host uses `localhost:5433` |
| `GEMINI_MODEL` | `gemini-flash-latest` | generation |
| `GEMINI_JUDGE_MODEL` | `gemini-flash-lite-latest` | LLM-as-judge |
| `CHUNK_STRATEGY` | `recursive` | `recursive` \| `semantic` \| `structure` |
| `RAG_TOP_K` / `RAG_CANDIDATE_K` | `5` / `40` | final / per-retriever fan-out |
| `RAG_USE_RERANKER` | `1` | cross-encoder rerank |
| `RAG_JUDGE_SAMPLE_RATE` | `1.0` | fraction of requests judged |
| `PG_HNSW_M` / `PG_HNSW_EF_SEARCH` | `16` / `100` | recall/latency tuning |

## Tests

```bash
python -m pytest          # 12 tests, offline, ~0.2s
```

Covers chunking, registry/reindex semantics, alias swap, BM25 ranking, hybrid
retrieval, the model-lock guard, and `valid_from` staged visibility.
