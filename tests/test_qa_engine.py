import numpy as np
import pytest
from src.document_processor import Chunk
from src.qa_engine import QAEngine


class MockEmbeddingModel:
    """Mock embedding model producing deterministic vectors for unit tests."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        embeddings = []
        for text in texts:
            # Deterministic pseudo-vector based on character values
            vec = np.zeros(self.dim, dtype=np.float32)
            for i, ch in enumerate(text[: self.dim]):
                vec[i % self.dim] += ord(ch) % 10 + 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


def test_retrieval_retains_source_metadata_and_page_numbers():
    """Test 5: Verify retrieved results retain source filename, page number, and chunk ID."""
    mock_model = MockEmbeddingModel(dim=8)

    chunks = [
        Chunk(
            text="Transformer attention models are effective for scientific text.",
            source_filename="paper_one.pdf",
            page_number=3,
            document_id="doc_1",
            chunk_index=0,
        ),
        Chunk(
            text="Convolutional networks perform feature extraction on image data.",
            source_filename="paper_two.pdf",
            page_number=7,
            document_id="doc_2",
            chunk_index=1,
        ),
    ]

    engine = QAEngine(chunks, mock_model)
    results = engine.search("attention models", k=2)

    assert len(results) == 2
    for r in results:
        assert hasattr(r, "page_number")
        assert hasattr(r, "source_filename")
        assert hasattr(r, "chunk_id")
        assert hasattr(r, "score")
        assert isinstance(r.score, float)

    # Verify original chunks are mapped
    pages = {r.page_number for r in results}
    assert 3 in pages
    assert 7 in pages


def test_k_values_behave_correctly_when_fewer_than_k_chunks_exist():
    """Test 6: Verify k=3, 5, 10 behave correctly when fewer chunks exist."""
    mock_model = MockEmbeddingModel(dim=8)

    chunks = [
        Chunk(
            text="Only one chunk in this minimal index.",
            source_filename="solo.pdf",
            page_number=1,
            document_id="solo_doc",
            chunk_index=0,
        )
    ]

    engine = QAEngine(chunks, mock_model)

    for k_val in [3, 5, 10]:
        results = engine.search("test query", k=k_val)
        assert len(results) == 1
        assert results[0].source_filename == "solo.pdf"
        assert results[0].page_number == 1


def test_empty_engine_returns_empty_results():
    """Verify empty QAEngine handles search gracefully without crashing."""
    mock_model = MockEmbeddingModel(dim=8)
    engine = QAEngine([], mock_model)

    results = engine.search("any query", k=5)
    assert results == []
