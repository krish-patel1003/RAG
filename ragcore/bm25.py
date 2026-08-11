"""A small, dependency-free BM25 Okapi index for the lexical half of hybrid search.

At production scale BM25 is delegated to Postgres full-text search or an
OpenSearch/Elasticsearch index (see ARCHITECTURE.md). For the demo corpus this
in-memory implementation is exact, fast, and keeps the dependency surface small.
It is rebuilt from the vector store's active chunks and cached by a fingerprint
(index_version + document count) so it stays consistent with the index.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: List[str] = []
        self.doc_len: List[int] = []
        self.freqs: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

    def build(self, docs: Sequence[Tuple[str, str]]) -> "BM25Index":
        """docs: sequence of (chunk_vector_id, text)."""
        self.ids, self.doc_len, self.freqs, self.df = [], [], [], {}
        total_len = 0
        for cid, text in docs:
            toks = tokenize(text)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.ids.append(cid)
            self.freqs.append(tf)
            self.doc_len.append(len(toks))
            total_len += len(toks)
            for term in tf:
                self.df[term] = self.df.get(term, 0) + 1
        n = len(self.ids)
        self.avgdl = (total_len / n) if n else 0.0
        # Okapi BM25 idf with +1 to keep it non-negative
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }
        return self

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.ids:
            return []
        q_terms = tokenize(query)
        scores: List[float] = [0.0] * len(self.ids)
        for i, tf in enumerate(self.freqs):
            dl = self.doc_len[i] or 1
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = s
        ranked = sorted(
            zip(self.ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(cid, sc) for cid, sc in ranked[:top_k] if sc > 0.0]
