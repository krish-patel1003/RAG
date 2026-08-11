FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; sentence-transformers pulls torch (CPU).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the cross-encoder weights so the first query isn't slow / offline-broken.
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY ragcore ./ragcore
COPY app ./app
COPY ui ./ui
COPY scripts ./scripts

EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
