"""Filesystem connector.

Walk a directory (or match a glob) and turn supported files into
:class:`LoadedDoc` objects. Text extraction is format-aware:

* ``.txt`` / ``.md`` / ``.rst`` / code  -> read as UTF-8 text
* ``.html`` / ``.htm``                   -> strip tags (BeautifulSoup if present)
* ``.pdf``                               -> pypdf page text (if pypdf present)

The ``doc_id`` is ``file:<path-relative-to-root>`` so re-ingesting the same file
updates it in place (the content-hash gate then skips it if unchanged).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional

from . import LoadedDoc

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".text",
             ".py", ".js", ".ts", ".java", ".go", ".rb", ".json", ".yaml", ".yml"}
HTML_EXTS = {".html", ".htm"}
PDF_EXTS = {".pdf"}
SUPPORTED = TEXT_EXTS | HTML_EXTS | PDF_EXTS


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_html(path: Path) -> str:
    raw = _read_text(path)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except Exception:  # noqa: BLE001 -- bs4 missing / parse error
        import re

        return re.sub(r"<[^>]+>", " ", raw)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "reading PDFs requires pypdf (`pip install pypdf`)"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in HTML_EXTS:
        return _read_html(path)
    if ext in PDF_EXTS:
        return _read_pdf(path)
    return _read_text(path)


def iter_files(
    root: str,
    *,
    glob: str = "**/*",
    recursive: bool = True,
    exts: Optional[set] = None,
) -> Iterator[Path]:
    base = Path(root).expanduser()
    exts = exts or SUPPORTED
    if base.is_file():
        if base.suffix.lower() in exts:
            yield base
        return
    pattern = glob if recursive else glob.replace("**/", "")
    for p in sorted(base.glob(pattern)):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def load_filesystem(
    root: str,
    *,
    glob: str = "**/*",
    recursive: bool = True,
    exts: Optional[set] = None,
    min_chars: int = 20,
) -> List[LoadedDoc]:
    """Load all supported files under ``root`` into LoadedDocs."""
    base = Path(root).expanduser()
    root_for_id = base if base.is_dir() else base.parent
    docs: List[LoadedDoc] = []
    for path in iter_files(root, glob=glob, recursive=recursive, exts=exts):
        try:
            text = extract_text(path).strip()
        except Exception as exc:  # noqa: BLE001 -- skip unreadable files, keep going
            print(f"[filesystem] skip {path}: {exc}")
            continue
        if len(text) < min_chars:
            continue
        try:
            rel = path.relative_to(root_for_id)
        except ValueError:
            rel = path.name
        docs.append(
            LoadedDoc(
                doc_id=f"file:{rel}",
                text=text,
                title=path.name,
                source="filesystem",
                url=str(path),
                metadata={"path": str(path), "ext": path.suffix.lower(),
                          "bytes": str(path.stat().st_size)},
            )
        )
    return docs
