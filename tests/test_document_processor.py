import io
import fitz
import pytest
from src.document_processor import Chunk, DocumentProcessor, get_nlp


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


def create_synthetic_pdf(pages_text: list) -> bytes:
    """Creates an in-memory PDF using PyMuPDF with specified page texts."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_pdf_page_numbers_survive_extraction_and_chunking(nlp):
    """Test 1: Verify physical 1-based PDF page numbers survive extraction."""
    page_texts = [
        "This is the first sentence on page one. Here is the second sentence on page one. And a third sentence.",
        "Page two begins here with distinct text. It discusses transformer attention mechanisms. This concludes page two.",
        "Page three contains experimental methodology. We evaluated across several benchmarks. Results showed positive gains.",
    ]
    pdf_bytes = create_synthetic_pdf(page_texts)

    processor = DocumentProcessor(sentences_per_chunk=3)
    file_obj = io.BytesIO(pdf_bytes)
    file_obj.name = "test_study.pdf"

    chunks = processor.process_file(file_obj, nlp)

    assert len(chunks) >= 3
    # Check that chunks from each page have the correct 1-based page number
    page_numbers = [c.page_number for c in chunks]
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers
    assert min(page_numbers) == 1
    assert max(page_numbers) == 3


def test_chunk_metadata_completeness_and_consistency(nlp):
    """Test 2: Verify chunk metadata is consistent and complete."""
    pdf_bytes = create_synthetic_pdf([
        "Sentence one of the paper. Sentence two of the paper. Sentence three completes chunk one. Sentence four starts chunk two. Sentence five finishes chunk two.",
    ])

    processor = DocumentProcessor(sentences_per_chunk=3)
    file_obj = io.BytesIO(pdf_bytes)
    file_obj.name = "metadata_test.pdf"

    chunks = processor.process_file(file_obj, nlp)
    assert len(chunks) > 0

    for i, chunk in enumerate(chunks):
        assert chunk.document_id.startswith("metadata_test_")
        assert chunk.source_filename == "metadata_test.pdf"
        assert chunk.source == "metadata_test.pdf"  # Backwards compatibility
        assert chunk.page_number == 1
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"{chunk.document_id}_p1_c{i}"
        assert len(chunk.text) >= processor.min_chunk_chars
        assert hasattr(chunk, "score")
        assert hasattr(chunk, "pos_tags")

        meta = chunk.metadata
        assert meta["document_id"] == chunk.document_id
        assert meta["chunk_id"] == chunk.chunk_id
        assert meta["page_number"] == 1


def test_empty_and_textless_pages_handled_gracefully(nlp):
    """Test 3: Verify blank or whitespace-only pages are skipped without shifting subsequent page numbers."""
    page_texts = [
        "First page has substantive text. It contains enough words to form a valid chunk.",
        "",  # Empty page 2
        "   \n\t  ",  # Whitespace-only page 3
        "Fourth page has substantive text. We verify that page four retains physical page index 4.",
    ]
    pdf_bytes = create_synthetic_pdf(page_texts)

    processor = DocumentProcessor(sentences_per_chunk=2)
    file_obj = io.BytesIO(pdf_bytes)
    file_obj.name = "sparse_document.pdf"

    chunks = processor.process_file(file_obj, nlp)

    assert len(chunks) >= 2
    pages = [c.page_number for c in chunks]
    assert 1 in pages
    assert 4 in pages
    assert 2 not in pages
    assert 3 not in pages
    assert 2 in processor.skipped_pages
    assert 3 in processor.skipped_pages


def test_document_boundaries_not_merged(nlp):
    """Test 4: Verify processing multiple documents preserves distinct document IDs and boundaries."""
    doc1_bytes = create_synthetic_pdf(["Paper A presents novel retrieval findings. It was published in 2023."])
    doc2_bytes = create_synthetic_pdf(["Paper B proposes reranking architectures. It was published in 2024."])

    proc = DocumentProcessor(sentences_per_chunk=2)

    f1 = io.BytesIO(doc1_bytes)
    f1.name = "paper_a.pdf"
    chunks1 = proc.process_file(f1, nlp)

    f2 = io.BytesIO(doc2_bytes)
    f2.name = "paper_b.pdf"
    chunks2 = proc.process_file(f2, nlp)

    assert chunks1[0].document_id != chunks2[0].document_id
    assert chunks1[0].source_filename == "paper_a.pdf"
    assert chunks2[0].source_filename == "paper_b.pdf"
    assert chunks1[0].chunk_id != chunks2[0].chunk_id
