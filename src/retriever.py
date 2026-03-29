# src/retriever.py

import numpy as np
from src.embedder import model

def retrieve(query, index, chunks, k=3):

    if len(chunks) == 0:
        return []

    # ensure k valid
    k = min(k, len(chunks))

    query_embedding = model.encode([query])

    D, I = index.search(np.array(query_embedding), k)

    results = []

    for idx in I[0]:
        # 🔥 KEY FIX: clip index
        if idx >= len(chunks):
            continue
        if idx < 0:
            continue

        results.append(chunks[idx])

    return results