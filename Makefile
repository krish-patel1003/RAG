.PHONY: help install db up down test fetch index demo api ui offline

help:
	@echo "install   pip install requirements"
	@echo "db        start pgvector (host port 5433)"
	@echo "up/down   full docker stack (api + ui + db)"
	@echo "fetch     download wikipedia + arxiv corpora"
	@echo "index     index both corpora into the store"
	@echo "demo      run the two curated demos"
	@echo "api/ui    run FastAPI / Streamlit locally"
	@echo "test      run the offline test suite"

DB_URL ?= postgresql://rag:rag@localhost:5433/rag
export DATABASE_URL=$(DB_URL)

install:
	pip install -r requirements.txt

db:
	docker compose up -d db

up:
	docker compose up --build

down:
	docker compose down

fetch:
	python scripts/fetch_dataset.py --source wikipedia --limit 150 --out data/wikipedia.jsonl
	python scripts/fetch_dataset.py --source arxiv --limit 80 --out data/arxiv.jsonl

index:
	python scripts/index_corpus.py data/wikipedia.jsonl
	python scripts/index_corpus.py data/arxiv.jsonl

demo:
	python scripts/demo.py both

api:
	uvicorn app.main:app --port 8000 --reload

ui:
	RAG_API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py

test:
	python -m pytest -q

offline:
	RAG_BACKEND=sqlite RAG_EMBEDDER=hashing python scripts/query.py "battery life"
