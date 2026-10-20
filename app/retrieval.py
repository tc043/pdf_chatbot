"""Retrieval: vector search, optionally fused with BM25 keyword search.

Embeddings are good at paraphrases but can miss exact tokens (a hyperparameter value, an
acronym, a table caption). BM25 is the opposite. Hybrid mode runs both over the same stored
chunks and merges the two rankings with reciprocal rank fusion (RRF), which needs no score
normalisation. The keyword index is built from the chunks already stored in Chroma, so it
works the same for freshly built and reused collections and needs no extra files on disk.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from app import config

if TYPE_CHECKING:  # pragma: no cover
    from langchain_chroma import Chroma

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # standard RRF damping constant
MIN_CANDIDATES = 20  # per ranker, so fusion has something to work with even for small k
_TOKEN = re.compile(r"\w+")
_CACHE_SIZE = 4


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; Markdown and table punctuation drop out."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """A small in-memory Okapi BM25 index over a fixed list of documents."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        self._tokens = [Counter(tokenize(doc.page_content)) for doc in self.documents]
        self._lengths = [sum(counts.values()) for counts in self._tokens]
        total = len(self.documents)
        self._avg_length = (sum(self._lengths) / total) if total else 0.0
        frequency: Counter[str] = Counter()
        for counts in self._tokens:
            frequency.update(counts.keys())
        self._idf = {
            term: math.log(1 + (total - n + 0.5) / (n + 0.5)) for term, n in frequency.items()
        }

    def search(self, query: str, k: int) -> list[Document]:
        """Return up to ``k`` documents with a non-zero score, best first."""
        terms = [term for term in set(tokenize(query)) if term in self._idf]
        if not terms or not self.documents:
            return []
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self._tokens):
            norm = 1 - BM25_B + BM25_B * self._lengths[index] / (self._avg_length or 1.0)
            score = 0.0
            for term in terms:
                count = counts.get(term, 0)
                if count:
                    score += self._idf[term] * count * (BM25_K1 + 1) / (count + BM25_K1 * norm)
            if score > 0:
                scored.append((score, index))
        # Higher score first; ties keep document order so results are deterministic.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self.documents[index] for _, index in scored[:k]]


def _chunk_key(doc: Document) -> tuple[object, str]:
    return doc.metadata.get("page"), doc.page_content


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Document]], k: int) -> list[Document]:
    """Merge ranked lists; a chunk found by several rankers rises, duplicates collapse."""
    scores: dict[tuple[object, str], float] = {}
    first_seen: dict[tuple[object, str], tuple[Document, int]] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = _chunk_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            first_seen.setdefault(key, (doc, len(first_seen)))
    # Ties go to whichever ranker listed the chunk first (the vector ranking comes first).
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key][1]))
    return [first_seen[key][0] for key in ordered[:k]]


_cache_lock = threading.Lock()
_cache: dict[str, BM25Index] = {}


def _keyword_index(db: Chroma) -> BM25Index:
    """Build (or reuse) the BM25 index for the chunks stored in ``db``."""
    stored = db.get(include=["documents", "metadatas"])
    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or [{}] * len(texts)
    digest = hashlib.sha1("\x00".join(texts).encode("utf-8"), usedforsecurity=False).hexdigest()
    with _cache_lock:
        cached = _cache.get(digest)
    if cached is not None:
        return cached
    index = BM25Index(
        [
            Document(page_content=text, metadata=dict(meta or {}))
            for text, meta in zip(texts, metadatas, strict=True)
        ]
    )
    with _cache_lock:
        if len(_cache) >= _CACHE_SIZE:
            _cache.pop(next(iter(_cache)))
        _cache[digest] = index
    return index


def search(
    db: Chroma, question: str, k: int, mode: str = config.RETRIEVAL_VECTOR
) -> list[Document]:
    """Return the ``k`` best chunks for ``question`` using the given retrieval ``mode``."""
    if mode != config.RETRIEVAL_HYBRID:
        return db.similarity_search(question, k=k)
    candidates = max(4 * k, MIN_CANDIDATES)
    vector_hits = db.similarity_search(question, k=candidates)
    keyword_hits = _keyword_index(db).search(question, candidates)
    return reciprocal_rank_fusion([vector_hits, keyword_hits], k)
