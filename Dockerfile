# ---- rag-api image (FastAPI + retrieval + generation + incremental ingestion) ----
# Heavy: torch (CPU) for the cross-encoder reranker + Playwright/Chromium for the
# crawl4ai web-research connector. Binds to $PORT for Cloud Run.
FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the cross-encoder weights so the first query isn't slow.
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Chromium + system deps for crawl4ai (web-research connector).
RUN python -m playwright install --with-deps chromium

COPY ragcore ./ragcore
COPY app ./app
COPY ui ./ui
COPY scripts ./scripts

EXPOSE 8000
# Shell form so ${PORT} (set by Cloud Run) is expanded.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
