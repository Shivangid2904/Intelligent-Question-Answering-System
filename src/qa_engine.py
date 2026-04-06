import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


class QAEngine:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]

        self.embeddings = model.encode(self.texts)
        self.embeddings = np.array(self.embeddings).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query, k=3):
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        D, I = self.index.search(query_embedding, k)

        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.texts):
                results.append(self.chunks[idx])

        return results