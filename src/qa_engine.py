import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


class QAEngine:
    def __init__(self, chunks, model):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        self.model = model

        self.embeddings = model.encode(
            self.texts,
            normalize_embeddings=True
        )
        self.embeddings = np.array(self.embeddings).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query, k=10):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding).astype("float32")

        D, I = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(I[0]):
            if 0 <= idx < len(self.texts):
                chunk = self.chunks[idx]
                chunk.score = float(D[0][i])
                results.append(chunk)

        return results