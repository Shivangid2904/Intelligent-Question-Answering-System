"""
lexical_retriever.py
====================
Standalone BM25 lexical retrieval for IntelliAsk.

Builds a BM25Okapi index over pre-chunked Chunk objects and returns ranked
results with their original document/page provenance preserved.  This module
is intentionally *isolated* from the dense FAISS pipeline in qa_engine.py.
Reciprocal Rank Fusion (RRF) will be added in a later step (Week 3, Step 2).

Design invariants:
- Chunk objects are never copied or mutated, except that ``chunk.score`` is
  set to the BM25 score on the *returned* list copy.
- Physical page numbers, document IDs, chunk IDs, and source filenames
  survive retrieval unchanged.
- Deterministic: identical query and corpus always produce identical ranking.
- Zero network or filesystem I/O at search time.
"""

from __future__ import annotations

import copy
import string
from typing import List, Optional

from rank_bm25 import BM25Okapi

# Import the canonical Chunk class so provenance fields are guaranteed present.
from src.document_processor import Chunk


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Punctuation table that maps non-hyphen punctuation characters to spaces so
# that slashed or bracketed terms (e.g. "LiDAR/sensor", "(OS1-32)?") are separated
# into clean tokens, while preserving hyphens inside identifiers (e.g. "OS1-32",
# "all-MiniLM-L6-v2").
_PUNCT_NO_HYPHEN = string.punctuation.replace("-", "")
_SPACE_TABLE = str.maketrans({c: " " for c in _PUNCT_NO_HYPHEN})


def _tokenize(text: str) -> List[str]:
    """Convert a string into a lowercase token list suitable for BM25Okapi.

    Behaviour:
    - Lowercases the entire string.
    - Replaces non-hyphen punctuation with spaces so adjacent words separated
      by slashes, colons, brackets, or commas become distinct tokens.
    - Preserves internal hyphens within technical compound identifiers such as
      ``OS1-32`` or ``all-MiniLM-L6-v2``.
    - Strips leading and trailing hyphens from each token so that standalone
      punctuation dashes (e.g. ``-`` or ``---``) and bounding hyphens are
      discarded.
    - Splits on whitespace and discards empty tokens.

    Args:
        text: Raw text string.

    Returns:
        List of non-empty lowercase token strings.
    """
    lowered = text.lower()
    spaced = lowered.translate(_SPACE_TABLE)
    tokens: List[str] = []
    for raw in spaced.split():
        clean = raw.strip("-")
        if clean:
            tokens.append(clean)
    return tokens


# ---------------------------------------------------------------------------
# LexicalRetriever
# ---------------------------------------------------------------------------


class LexicalRetriever:
    """BM25Okapi-based lexical retriever over a fixed corpus of Chunk objects.

    The index is built once at construction time from the provided chunks.
    Subsequent ``search()`` calls are read-only against the index.

    Attributes:
        chunks (List[Chunk]): The original chunk objects in index order.
        k1 (float): BM25 term-frequency saturation parameter (default 1.5).
        b (float): BM25 length normalisation parameter (default 0.75).
        epsilon (float): BM25 IDF smoothing floor (default 0.25).
    """

    def __init__(
        self,
        chunks: List[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        """Build the BM25 index.

        Args:
            chunks: List of Chunk objects.  The list may be empty; ``search``
                    will return ``[]`` gracefully in that case.
            k1: BM25 k1 parameter (term-frequency saturation).
            b: BM25 b parameter (document-length normalisation).
            epsilon: BM25 epsilon parameter (IDF smoothing).

        Raises:
            TypeError: If any element of ``chunks`` is not a Chunk instance.
        """
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(
                    f"All elements of 'chunks' must be Chunk instances; "
                    f"got {type(chunk).__name__} at index {i}."
                )

        self.chunks: List[Chunk] = list(chunks)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Tokenised corpus in index order — same length as self.chunks.
        self._tokenized_corpus: List[List[str]] = [
            _tokenize(c.text) for c in self.chunks
        ]

        # BM25Okapi index.  None when corpus is empty.
        self._bm25: Optional[BM25Okapi] = None
        if self.chunks:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[Chunk]:
        """Execute a BM25 lexical search and return the top-k chunks.

        Provenance fields (``page_number``, ``source_filename``,
        ``document_id``, ``chunk_id``) are preserved from the original Chunk
        objects.  Only ``chunk.score`` is overwritten with the BM25 score.

        Args:
            query: Natural language or keyword query string.  An empty or
                   whitespace-only query returns an empty list.
            k: Maximum number of results to return.  Clamped to
               ``[1, len(chunks)]`` internally.

        Returns:
            List of Chunk objects sorted by BM25 score descending.
            Returns ``[]`` when the corpus is empty, the query is empty, or
            all chunks score zero (no lexical overlap with the query).

        Notes:
            Zero/negative-score filtering: BM25Okapi assigns a score of 0.0
            when a query token is absent from a document, and a *negative*
            score when the token's IDF is negative (term frequency across all
            documents equals corpus size, making log(N/df) negative before
            epsilon smoothing).  This method returns ``[]`` when the highest
            chunk score is <= 0, covering both the zero-overlap case and the
            all-documents-match case where BM25 has no discriminative power.
        """
        if not self.chunks or self._bm25 is None:
            return []

        if not query or not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # get_scores returns a numpy array aligned with self.chunks.
        raw_scores = self._bm25.get_scores(query_tokens)

        # Sort (original_index, score) pairs descending by score.
        scored = sorted(
            enumerate(raw_scores), key=lambda x: x[1], reverse=True
        )

        # BM25Okapi can return negative scores when a query token's IDF is
        # negative (term appears in every document; df == N makes IDF < 0
        # before epsilon smoothing).  We treat any best-score <= 0 as
        # "no useful lexical signal" and return an empty list.
        if not scored or scored[0][1] <= 0.0:
            return []

        effective_k = min(max(1, k), len(self.chunks))

        results: List[Chunk] = []
        for idx, score in scored[:effective_k]:
            if score <= 0.0:
                break
            chunk = copy.copy(self.chunks[idx])
            chunk.score = float(score)
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunks)

    @property
    def avg_doc_length(self) -> float:
        """Average token count per chunk (useful for sanity checks)."""
        if not self._tokenized_corpus:
            return 0.0
        return sum(len(t) for t in self._tokenized_corpus) / len(
            self._tokenized_corpus
        )

    def __repr__(self) -> str:
        return (
            f"<LexicalRetriever corpus_size={self.corpus_size} "
            f"k1={self.k1} b={self.b}>"
        )
