import io
import fitz
import pytest
from src.document_processor import DocumentProcessor, get_nlp
from src.qa_engine import QAEngine
from src.llm_engine import generate_answer
from tests.test_qa_engine import MockEmbeddingModel


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


def test_app_py_interfaces_and_data_contracts(nlp):
    """
    Test 10: Verify the exact contracts and interfaces used by app.py
    remain 100% compatible with the revised modules.
    """
    # 1. Verify expected imports succeed
    assert callable(get_nlp)
    assert callable(DocumentProcessor)
    assert callable(QAEngine)
    assert callable(generate_answer)

    # 2. Emulate app.py document processing workflow
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "First sentence in uploaded document. Second sentence discusses RAG. Third sentence concludes page.")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Fourth sentence begins second page. Fifth sentence details retrieval indexing. Sixth sentence ends document.")
    pdf_bytes = doc.write()
    doc.close()

    uploaded_file = io.BytesIO(pdf_bytes)
    uploaded_file.name = "NLP_Sample.pdf"

    processor = DocumentProcessor()
    chunks = processor.process_file(uploaded_file, nlp)

    assert len(chunks) > 0
    assert hasattr(processor, "chunks")

    # 3. Emulate app.py build_engine(tuple(processor.chunks), embedding_model)
    chunks_tuple = tuple(processor.chunks)
    mock_model = MockEmbeddingModel(dim=8)
    engine = QAEngine(chunks_tuple, mock_model)

    # 4. Emulate app.py search and result inspection
    query = "What is RAG?"
    results = engine.search(query, k=10)
    assert len(results) > 0

    # 5. Emulate app.py keyword boosting loop (lines 319-325)
    content_words = ["sentence", "rag"]
    best = None
    best_score = -1.0
    for r in results:
        # app.py accesses r.text, r.score, r.source
        assert hasattr(r, "text") and isinstance(r.text, str)
        assert hasattr(r, "score") and isinstance(r.score, float)
        assert hasattr(r, "source") and r.source == "NLP_Sample.pdf"
        assert hasattr(r, "pos_tags")

        match_count = sum(word in r.text.lower() for word in content_words)
        combined_score = r.score + (0.15 * match_count)
        if combined_score > best_score:
            best_score = combined_score
            best = r

    assert best is not None

    # 6. Emulate app.py filtered_passages construction (lines 328-333)
    SCORE_THRESHOLD = 0.0  # mock embeddings have varied scores
    filtered_passages = []
    for r in results:
        if r.score < SCORE_THRESHOLD:
            continue
        filtered_passages.append({"text": r.text, "score": round(float(r.score), 3)})

    assert len(filtered_passages) > 0

    # 7. Emulate app.py history entry construction (line 406)
    entry = {
        "question": query,
        "answer": "Mock generated answer",
        "answer_source": "mock",
        "source": best.source,
        "score": best.score,
        "passages": filtered_passages,
    }
    assert entry["source"] == "NLP_Sample.pdf"
    assert isinstance(entry["score"], float)
