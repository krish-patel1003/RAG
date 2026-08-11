"""Web-research connector (on-demand corpus expansion): SearXNG + crawl4ai.

Two stages, deliberately separated so a human can review before anything is
indexed:

1. **search** — :class:`SearXNGClient` queries a self-hosted SearXNG instance's
   JSON API and returns candidate :class:`SearchResult` s. Three research modes
   map to SearXNG params:
     * ``papers`` -> ``categories=science`` (arXiv, Semantic Scholar, Scholar…)
     * ``wikis``  -> Wikipedia engine, with a general-search wikipedia.org filter
       as a fallback
     * ``web``    -> general search
2. **extract** — :class:`Crawler` fetches the chosen URLs and extracts clean
   text. crawl4ai is the primary extractor (headless-browser rendering →
   markdown); a dependency-free urllib+BeautifulSoup path is the fallback so the
   feature runs even without crawl4ai/Playwright installed.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

from ..config import settings
from . import LoadedDoc, SearchResult

UA = "prod-rag-research/1.0 (+https://github.com/krish-patel1003/RAG)"

MODE_PARAMS = {
    "papers": {"categories": "science"},
    "wikis": {"engines": "wikipedia"},
    "web": {},
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
class SearXNGClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or settings.searxng_url).rstrip("/")

    def _raw_search(self, query: str, params: dict) -> List[dict]:
        qs = urllib.parse.urlencode({"q": query, "format": "json", **params})
        req = urllib.request.Request(
            f"{self.base_url}/search?{qs}", headers={"User-Agent": UA}
        )
        import json

        with urllib.request.urlopen(req, timeout=settings.search_timeout) as resp:
            data = json.loads(resp.read())
        return data.get("results", [])

    def search(self, query: str, mode: str = "papers", limit: int = 10) -> List[SearchResult]:
        params = MODE_PARAMS.get(mode, {})
        raw = self._raw_search(query, params)

        # Fallback: the wikipedia engine can be flaky; pull wikipedia.org URLs
        # out of a general search instead.
        if mode == "wikis" and not raw:
            raw = [
                r for r in self._raw_search(query, {})
                if "wikipedia.org" in (r.get("url") or "")
            ]

        results: List[SearchResult] = []
        seen = set()
        for r in raw:
            url = r.get("url") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                SearchResult(
                    title=(r.get("title") or url).strip(),
                    url=url,
                    snippet=(r.get("content") or "").strip()[:400],
                    engine=r.get("engine", ""),
                    source=mode,
                    score=r.get("score"),
                )
            )
            if len(results) >= limit:
                break
        return results


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def doc_id_from_url(url: str) -> Tuple[str, str]:
    """Return (doc_id, source) with stable, source-aware ids."""
    u = urllib.parse.urlparse(url)
    host = u.netloc.lower()
    if "arxiv.org" in host:
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", u.path)
        if m:
            return f"arxiv:{m.group(1)}", "arxiv"
    if "wikipedia.org" in host:
        title = u.path.rsplit("/", 1)[-1]
        return f"wiki:{title}", "wikipedia"
    slug = re.sub(r"[^a-z0-9]+", "-", (host + u.path).lower()).strip("-")[:80]
    return f"web:{slug}", "web"


class Crawler:
    """Fetch a URL and extract clean text. crawl4ai primary, urllib fallback."""

    def __init__(self) -> None:
        try:
            import crawl4ai  # noqa: F401

            self.backend = "crawl4ai"
        except Exception:  # noqa: BLE001
            self.backend = "fallback"

    # -- crawl4ai path --------------------------------------------------
    def _crawl4ai(self, url: str) -> Tuple[str, str]:
        import asyncio

        from crawl4ai import (
            AsyncWebCrawler,
            CrawlerRunConfig,
            DefaultMarkdownGenerator,
            PruningContentFilter,
        )

        # Prune boilerplate (nav / headers / footers / sidebars) and keep the
        # main article body -- otherwise crawled pages (esp. Wikipedia) are
        # dominated by menu chrome that pollutes retrieval.
        md_gen = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed")
        )
        config = CrawlerRunConfig(
            markdown_generator=md_gen,
            excluded_tags=["nav", "header", "footer", "aside", "form", "script", "style"],
            word_count_threshold=15,
            exclude_external_links=True,
            only_text=False,
        )

        async def _run() -> Tuple[str, str]:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)
                md_obj = getattr(result, "markdown", None)
                text = ""
                if md_obj is not None:
                    # Prefer the pruned "fit" markdown; fall back to raw.
                    text = (getattr(md_obj, "fit_markdown", "") or
                            getattr(md_obj, "raw_markdown", "") or str(md_obj))
                meta = getattr(result, "metadata", None) or {}
                title = meta.get("title", "") if isinstance(meta, dict) else ""
                return title, str(text)

        return asyncio.run(_run())

    # -- fallback path --------------------------------------------------
    def _fallback(self, url: str) -> Tuple[str, str]:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=settings.search_timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            return self._extract_pdf(body)
        html = body.decode("utf-8", errors="ignore")
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            title = (soup.title.string if soup.title else "") or ""
            main = soup.find("main") or soup.find("article") or soup.body or soup
            return title.strip(), main.get_text(separator="\n").strip()
        except Exception:  # noqa: BLE001
            return "", re.sub(r"<[^>]+>", " ", html)

    def _extract_pdf(self, body: bytes) -> Tuple[str, str]:
        import io

        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(body))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
            title = (reader.metadata or {}).get("/Title", "") if reader.metadata else ""
            return str(title or ""), text
        except Exception as exc:  # noqa: BLE001
            return "", f"(could not extract PDF: {exc})"

    def fetch(self, url: str) -> Tuple[str, str]:
        if self.backend == "crawl4ai":
            try:
                return self._crawl4ai(url)
            except Exception as exc:  # noqa: BLE001 -- degrade, don't fail the request
                print(f"[crawler] crawl4ai failed ({exc}); using fallback")
        return self._fallback(url)


def load_urls(urls: List[str], *, min_chars: int = 200) -> List[LoadedDoc]:
    """Crawl + extract a list of URLs into LoadedDocs (ready for indexing)."""
    crawler = Crawler()
    docs: List[LoadedDoc] = []
    for url in urls:
        try:
            title, text = crawler.fetch(url)
        except Exception as exc:  # noqa: BLE001
            print(f"[crawler] skip {url}: {exc}")
            continue
        text = _clean(text)
        if len(text) < min_chars:
            print(f"[crawler] skip {url}: only {len(text)} chars extracted")
            continue
        doc_id, source = doc_id_from_url(url)
        docs.append(
            LoadedDoc(
                doc_id=doc_id, text=text, title=title or url, source=source,
                url=url, metadata={"extractor": crawler.backend},
            )
        )
    return docs


# Lines that are pure site chrome and should be dropped entirely.
_BOILERPLATE_LINES = re.compile(
    r"^\s*(jump to content|from wikipedia, the free encyclopedia|main menu|"
    r"move to sidebar|navigation|contribute|toggle .* subsection|"
    r"personal tools|search|log in|create account|views|actions|print/export|"
    r"in other projects|languages|edit links|retrieved from|this page was last)\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Turn crawled markdown into clean prose for embedding + reranking.

    Crawled markdown is full of link/image syntax and residual site chrome that
    add no semantic signal but dominate short chunks. We flatten links to their
    anchor text, drop images, and remove obvious navigation lines.
    """
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)     # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> anchor text
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # md headings
    kept = [ln for ln in text.splitlines() if not _BOILERPLATE_LINES.match(ln)]
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
