"""Index storage: FAISS (dense), BM25 (sparse), and StructuralIndex (paragraph lookup)."""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from .config import AppConfig
from .document import StructuredChunk

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", flags=re.UNICODE)

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in WORD_RE.findall(text)]


class StructuralIndex:
    """Maps paragraph number → list of chunk_ids for exact structural lookup."""

    def __init__(self) -> None:
        self._map: dict[int, list[int]] = {}

    def build(self, chunks: list[StructuredChunk]) -> None:
        self._map.clear()
        for chunk in chunks:
            paragraph_numbers = tuple(getattr(chunk.metadata, "paragraph_numbers", ()))
            if not paragraph_numbers:
                n = chunk.metadata.paragraph_number
                paragraph_numbers = (n,) if n is not None else ()
            for paragraph_number in paragraph_numbers:
                self._map.setdefault(paragraph_number, []).append(chunk.metadata.chunk_id)

    def search(self, paragraph_refs: list[int]) -> list[int]:
        result: list[int] = []
        seen: set[int] = set()
        for n in paragraph_refs:
            for chunk_id in self._map.get(n, []):
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                result.append(chunk_id)
        return result

    def chunk_ids_for_paragraph(self, paragraph_number: int) -> list[int]:
        return list(self._map.get(paragraph_number, []))


class IndexStore:
    """Builds and holds FAISS dense index, BM25 sparse index, and StructuralIndex."""

    def __init__(self, chunks: list[StructuredChunk], config: AppConfig) -> None:
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        self.chunk_texts = [c.text for c in chunks]
        self.config = config
        self.structural = StructuralIndex()
        self.structural.build(chunks)
        self._tokenized: list[list[str]] = [tokenize(t) for t in self.chunk_texts]

        self._embedder = None
        self._faiss_index = None
        self._bm25 = None
        self._build()

    def _build(self) -> None:
        # Dense index
        if _ST_AVAILABLE and _FAISS_AVAILABLE:
            self._embedder = SentenceTransformer(self.config.embedding_model)
            embeddings = self._embedder.encode(
                self.chunk_texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            self._faiss_index = _faiss.IndexFlatIP(embeddings.shape[1])
            self._faiss_index.add(embeddings)

        # Sparse BM25
        if _BM25_AVAILABLE:
            self._bm25 = BM25Okapi(self._tokenized)

    @property
    def has_dense(self) -> bool:
        return self._faiss_index is not None and self._embedder is not None

    @property
    def has_sparse(self) -> bool:
        return self._bm25 is not None

    @staticmethod
    def _format_query(query: str) -> str:
        return f"Represent this sentence for searching relevant passages: {query}"

    def dense_search(self, query: str, top_k: int) -> list[int]:
        if not self.has_dense:
            return []
        qvec = self._embedder.encode(  # type: ignore[union-attr]
            [self._format_query(query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, ids = self._faiss_index.search(qvec, top_k)  # type: ignore[union-attr]
        result: list[int] = []
        for doc_id, score in zip(ids[0], scores[0]):
            if doc_id < 0:
                continue
            if float(score) <= 0:
                continue
            result.append(int(doc_id))
        return result

    def sparse_search(self, query: str, top_k: int) -> list[int]:
        if not self.has_sparse:
            return self._tfidf_search(query, top_k)
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)  # type: ignore[union-attr]
        ids = np.argsort(-scores)[:top_k]
        return [int(i) for i in ids]

    def _tfidf_search(self, query: str, top_k: int) -> list[int]:
        """Minimal overlap-based fallback when BM25 is unavailable."""
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return list(range(min(top_k, len(self.chunk_texts))))
        scored: list[tuple[int, int]] = []
        for idx, toks in enumerate(self._tokenized):
            overlap = len(q_tokens & set(toks))
            if overlap:
                scored.append((idx, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored[:top_k]]
