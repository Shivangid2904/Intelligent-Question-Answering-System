# Convert text into embeddings

from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(texts):
    return model.encode(texts)