from typing import Any, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load dense embedding model from Sentence Transformers.
    Defaults to 384-dimensional all-MiniLM-L6-v2.
    """
    return SentenceTransformer(model_name)


class QAEngine:
    """
    Dense semantic search engine built on FAISS IndexFlatIP (cosine similarity
    via L2-normalized vectors). Preserves chunk metadata and physical page
    provenance throughout indexing and retrieval.
    """

    def __init__(self, chunks: List[Any], model: Any):
        """
        Args:
            chunks: List of Chunk objects containing text and provenance metadata.
            model: SentenceTransformer instance.
        """
        self.chunks = list(chunks)
        self.texts = [c.text for c in self.chunks]
        self.model = model
        self.dim: Optional[int] = None
        self.index: Optional[faiss.IndexFlatIP] = None

        if self.chunks:
            self._build_index()

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

    def search(self, query: str, k: int = 10) -> List[Any]:
        """
        Execute dense semantic search for a query string.

        Args:
            query: Natural language query string.
            k: Number of nearest-neighbor chunks to retrieve (e.g., 3, 5, 10).

        Returns:
            List of Chunk objects sorted by cosine similarity in descending order,
            with chunk.score populated.
        """
        if not self.chunks or self.index is None or self.index.ntotal == 0:
            return []

        # Bound k by available indexed chunks
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
                chunk = self.chunks[idx]
                chunk.score = float(distances[0][i])
                results.append(chunk)

        return results