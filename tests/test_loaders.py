"""Loaders + ad-hoc ingestion (offline: filesystem, url-id derivation, cleaning,
mocked SearXNG, and ingestion through the same Indexer)."""

import json

import pytest

from ragcore.loaders.filesystem import load_filesystem
from ragcore.loaders.web import SearXNGClient, _clean, doc_id_from_url


# ---- filesystem -------------------------------------------------------
def test_filesystem_loader_reads_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nAlpha content about warranties.")
    (tmp_path / "b.txt").write_text("Beta content about shipping and delivery.")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01\x02")
    docs = load_filesystem(str(tmp_path))
    ids = {d.doc_id for d in docs}
    assert "file:a.md" in ids and "file:b.txt" in ids
    assert not any(d.doc_id.endswith(".bin") for d in docs)  # unsupported skipped
    assert all(d.source == "filesystem" for d in docs)


def test_filesystem_glob_filters(tmp_path):
    (tmp_path / "a.md").write_text("markdown one two three four five")
    (tmp_path / "b.txt").write_text("text one two three four five")
    docs = load_filesystem(str(tmp_path), glob="**/*.md")
    assert [d.doc_id for d in docs] == ["file:a.md"]


# ---- url id derivation ------------------------------------------------
@pytest.mark.parametrize("url,expected_id,expected_source", [
    ("https://arxiv.org/abs/1603.09320v4", "arxiv:1603.09320", "arxiv"),
    ("http://arxiv.org/abs/2005.11401", "arxiv:2005.11401", "arxiv"),
    ("https://en.wikipedia.org/wiki/Okapi_BM25", "wiki:Okapi_BM25", "wikipedia"),
    ("https://example.com/some/page", "web:example-com-some-page", "web"),
])
def test_doc_id_from_url(url, expected_id, expected_source):
    did, src = doc_id_from_url(url)
    assert did == expected_id
    assert src == expected_source


# ---- markdown cleaning ------------------------------------------------
def test_clean_strips_links_images_and_nav():
    raw = (
        "[Jump to content](https://x/#c)\n"
        "From Wikipedia, the free encyclopedia\n"
        "![logo](https://x/logo.png)\n"
        "Okapi BM25 is a [ranking function](https://x/rf) used by search engines.\n"
    )
    out = _clean(raw)
    assert "Jump to content" not in out
    assert "From Wikipedia" not in out
    assert "logo.png" not in out
    assert "ranking function" in out          # anchor text kept
    assert "https://x/rf" not in out          # url dropped


# ---- SearXNG client (mocked network) ----------------------------------
def test_searxng_search_parses_and_dedupes(monkeypatch):
    payload = {"results": [
        {"url": "http://arxiv.org/abs/1", "title": "A", "content": "sa", "engine": "arxiv"},
        {"url": "http://arxiv.org/abs/1", "title": "A dup", "content": "", "engine": "arxiv"},
        {"url": "http://arxiv.org/abs/2", "title": "B", "content": "sb", "engine": "scholar"},
    ]}
    client = SearXNGClient("http://searxng:8080")
    monkeypatch.setattr(client, "_raw_search", lambda q, p: payload["results"])
    res = client.search("anything", mode="papers", limit=10)
    assert [r.url for r in res] == ["http://arxiv.org/abs/1", "http://arxiv.org/abs/2"]
    assert res[0].source == "papers"


def test_searxng_wikis_fallback_filters_wikipedia(monkeypatch):
    client = SearXNGClient("http://searxng:8080")
    calls = {"n": 0}

    def fake_raw(q, params):
        calls["n"] += 1
        if calls["n"] == 1:  # wikipedia engine returns nothing
            return []
        return [  # general search fallback
            {"url": "https://en.wikipedia.org/wiki/HNSW", "title": "HNSW", "engine": "ddg"},
            {"url": "https://medium.com/x", "title": "blog", "engine": "ddg"},
        ]

    monkeypatch.setattr(client, "_raw_search", fake_raw)
    res = client.search("hnsw", mode="wikis", limit=10)
    assert [r.url for r in res] == ["https://en.wikipedia.org/wiki/HNSW"]


# ---- ingestion goes through the same Indexer --------------------------
def test_ingest_filesystem_indexes_and_is_queryable(rag_env, tmp_path):
    from ragcore import Ingestor, RAGPipeline

    (tmp_path / "battery.md").write_text(
        "# Battery\n\nThe device battery lasts about ten hours on a full charge."
    )
    ing = Ingestor()
    rep = ing.ingest_filesystem(str(tmp_path))
    assert rep["indexed"] == 1 and rep["chunks"] >= 1
    # re-ingest identical -> content-hash gate skips
    rep2 = ing.ingest_filesystem(str(tmp_path))
    assert rep2["skipped"] == 1
    # queryable through the normal pipeline
    res = RAGPipeline().query("how long does the battery last", judge=False)
    assert res["sources"] and res["sources"][0]["doc_id"] == "file:battery.md"
