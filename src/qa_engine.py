"""
qa_engine.py
============
Dense and hybrid search engine for scientific documents.

Implements:
- QAEngine: Manages FAISS IndexFlatIP (cosine similarity) dense retrieval and
  optional BM25 lexical retrieval via LexicalRetriever.
- reciprocal_rank_fusion: Standalone, deterministic Reciprocal Rank Fusion (RRF)
  combining multiple ranked candidate lists into a unified ranking:
      RRF_score(chunk) = sum_{m in M} 1.0 / (rrf_k + rank_m(chunk))
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Optional, Sequence
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.lexical_retriever import LexicalRetriever


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load dense embedding model from Sentence Transformers.
    Defaults to 384-dimensional all-MiniLM-L6-v2.
    """
    return SentenceTransformer(model_name)


def _get_chunk_id(chunk: Any) -> str:
    """Extract stable chunk identifier from a Chunk object or dict."""
    if hasattr(chunk, "chunk_id") and chunk.chunk_id:
        return str(chunk.chunk_id)
    if isinstance(chunk, dict):
        if chunk.get("chunk_id"):
            return str(chunk["chunk_id"])
        doc = chunk.get("source_filename") or chunk.get("document_id") or ""
        page = chunk.get("page_number", "")
        text = str(chunk.get("text") or "")
    else:
        doc = getattr(chunk, "source_filename", "") or getattr(chunk, "document_id", "")
        page = getattr(chunk, "page_number", "")
        text = str(getattr(chunk, "text", "") or "")

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc}_p{page}_{digest}"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Any]],
    k: int = 10,
    rrf_k: int = 60,
) -> List[Any]:
    """
    Fuse multiple ranked lists of chunks using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(chunk) = sum_{m in M} (1.0 / (rrf_k + rank_m(chunk)))
    where rank_m(chunk) is 1-based (rank 1, 2, ...).

    Design invariants:
    - Deterministic: identical input lists always produce identical fused rankings.
    - Ties broken deterministically: highest RRF score first; ties broken by
      earliest discovery order across candidate lists, then by chunk_id string.
    - Deduplication: each unique chunk appears at most once in the output.
    - Provenance: original chunk metadata (source_filename, page_number,
      document_id, chunk_id, etc.) is preserved. Returns shallow copies
      with chunk.score set to the fused RRF score without mutating originals.
    - Empty inputs: empty candidate lists are skipped; if all lists are empty,
      returns [].

    Args:
        ranked_lists: Sequence of ranked candidate sequences (e.g. [dense, lexical]).
        k: Maximum number of fused results to return.
        rrf_k: RRF smoothing constant in denominator (default 60).

    Returns:
        List of fused Chunk objects sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Any] = {}
    first_seen: Dict[str, int] = {}
    discovery_counter = 0

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        seen_in_current_list: set[str] = set()
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = _get_chunk_id(chunk)
            if chunk_id in seen_in_current_list:
                continue
            seen_in_current_list.add(chunk_id)

            rrf_term = 1.0 / (rrf_k + rank)
            if chunk_id not in scores:
                scores[chunk_id] = rrf_term
                chunk_map[chunk_id] = chunk
                first_seen[chunk_id] = discovery_counter
                discovery_counter += 1
            else:
                scores[chunk_id] += rrf_term

    if not scores:
        return []

    # Sort descending by RRF score; break ties deterministically
    sorted_chunk_ids = sorted(
        scores.keys(),
        key=lambda cid: (-scores[cid], first_seen[cid], cid),
    )

    effective_k = min(max(1, k), len(sorted_chunk_ids))

    results: List[Any] = []
    for cid in sorted_chunk_ids[:effective_k]:
        original_chunk = chunk_map[cid]
        fused_chunk = copy.copy(original_chunk)
        fused_chunk.score = float(scores[cid])
        results.append(fused_chunk)

    return results


class QAEngine:
    """
    Search engine built on FAISS IndexFlatIP (cosine similarity via L2-normalized vectors)
    with optional BM25 lexical retrieval and Reciprocal Rank Fusion (RRF).
    Preserves chunk metadata and physical page provenance throughout indexing and retrieval.
    """

    def __init__(
        self,
        chunks: Sequence[Any],
        model: Any,
        default_mode: str = "dense",
    ) -> None:
        """
        Args:
            chunks: Sequence of Chunk objects containing text and provenance metadata.
            model: SentenceTransformer instance or compatible mock.
            default_mode: Default retrieval mode ('dense', 'hybrid', or 'lexical').
                          Defaults to 'dense' for backward compatibility.
        """
        self.chunks = list(chunks)
        self.texts = [c.text for c in self.chunks]
        self.model = model
        self.default_mode = default_mode
        self.dim: Optional[int] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.lexical_retriever: Optional[LexicalRetriever] = None

        if self.chunks:
            self._build_index()
            try:
                self.lexical_retriever = LexicalRetriever(self.chunks)
            except Exception:
                self.lexical_retriever = None

    def _build_index(self) -> None:
        """
        Encodes chunk texts into unit-normalized dense float32 vectors
        and populates an in-memory FAISS IndexFlatIP index.
        """
        raw_embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = np.array(raw_embeddings).astype("float32")
        self.dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def search_dense(self, query: str, k: int = 10) -> List[Any]:
        """
        Execute dense semantic search using FAISS IndexFlatIP (cosine similarity).

        Args:
            query: Natural language query string.
            k: Number of nearest-neighbor chunks to retrieve.

        Returns:
            List of Chunk copies sorted by cosine similarity in descending order,
            with chunk.score populated.
        """
        if not self.chunks or self.index is None or self.index.ntotal == 0:
            return []

        effective_k = min(max(1, k), len(self.chunks))

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, effective_k)

        results: List[Any] = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = copy.copy(self.chunks[idx])
                chunk.score = float(distances[0][i])
                results.append(chunk)

        return results

    def search_lexical(self, query: str, k: int = 10) -> List[Any]:
        """
        Execute sparse lexical search using BM25Okapi via LexicalRetriever.

        Args:
            query: Keyword or natural language query string.
            k: Maximum number of lexical chunks to retrieve.

        Returns:
            List of Chunk copies sorted by BM25 score in descending order.
        """
        if not self.chunks or self.lexical_retriever is None:
            return []
        return self.lexical_retriever.search(query, k=k)

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        rrf_k: int = 60,
        candidate_k: Optional[int] = None,
        dense_k: Optional[int] = None,
        lexical_k: Optional[int] = None,
    ) -> List[Any]:
        """
        Execute hybrid retrieval combining dense FAISS search and BM25 lexical
        search via Reciprocal Rank Fusion (RRF).

        Args:
            query: Natural language or keyword query string.
            k: Maximum number of fused results to return.
            rrf_k: RRF smoothing constant (default 60).
            candidate_k: Optional candidate depth for both dense and lexical retrieval.
            dense_k: Optional override for dense candidate depth.
            lexical_k: Optional override for lexical candidate depth.

        Returns:
            List of fused Chunk copies sorted by RRF score in descending order.
        """
        if not self.chunks:
            return []

        default_candidate_depth = max(k * 2, 20)
        d_k = dense_k if dense_k is not None else (candidate_k if candidate_k is not None else default_candidate_depth)
        l_k = lexical_k if lexical_k is not None else (candidate_k if candidate_k is not None else default_candidate_depth)

        dense_candidates = self.search_dense(query, k=d_k)
        lexical_candidates = self.search_lexical(query, k=l_k)

        return reciprocal_rank_fusion(
            [dense_candidates, lexical_candidates],
            k=k,
            rrf_k=rrf_k,
        )

    def search(
        self,
        query: str,
        k: int = 10,
        mode: Optional[str] = None,
    ) -> List[Any]:
        """
        Execute search across the indexed corpus.

        Args:
            query: Query string.
            k: Maximum number of candidates to retrieve.
            mode: Retrieval mode ('dense', 'hybrid', or 'lexical').
                  Defaults to self.default_mode ('dense').

        Returns:
            List of Chunk copies sorted by retrieval score descending.
        """
        active_mode = (mode or self.default_mode).lower()
        if active_mode == "hybrid":
            return self.search_hybrid(query, k=k)
        elif active_mode == "dense":
            return self.search_dense(query, k=k)
        elif active_mode == "lexical":
            return self.search_lexical(query, k=k)
        else:
            raise ValueError(
                f"Unknown retrieval mode: {mode!r}. Expected 'dense', 'hybrid', or 'lexical'."
            )