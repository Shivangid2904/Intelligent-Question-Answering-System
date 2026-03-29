# Create and save FAISS index

import faiss
import numpy as np
import pickle
import os

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


def save_index(index, chunks):
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, "models/faiss_index.bin")

    with open("models/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index():
    index = faiss.read_index("models/faiss_index.bin")

    with open("models/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks