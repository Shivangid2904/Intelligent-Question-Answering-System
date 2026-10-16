"""
test_lexical_retriever.py
=========================
Unit tests for src/lexical_retriever.py (Week 3, Step 1).

Coverage:
  1. Exact technical term ranking  -- "Ouster OS1-32"
  2. Rare identifiers and model names
  3. Queries with no useful lexical matches (returns [])
  4. Empty query string and whitespace-only query
  5. Duplicate terms in query (handled stably by BM25)
  6. Preservation of document and page provenance
  7. Type validation on construction
  8. Empty corpus
  9. k-clamping (k > corpus size)
 10. Real-corpus integration test: Ouster page 5 ranks first
"""

from __future__ import annotations

import io
import os
import sys

import pytest

# Ensure repository root on sys.path (conftest.py handles this, but belt-and-
# suspenders for direct invocation).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.document_processor import Chunk
from src.lexical_retriever import LexicalRetriever, _tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    text: str,
    source_filename: str = "test.pdf",
    page_number: int = 1,
    document_id: str = "doc_test",
    chunk_index: int = 0,
) -> Chunk:
    """Create a Chunk for testing without requiring a PDF."""
    return Chunk(
        text=text,
        source_filename=source_filename,
        page_number=page_number,
        document_id=document_id,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# Tokeniser tests
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases_input(self):
        assert _tokenize("Ouster OS1-32") == ["ouster", "os1-32"]

    def test_strips_punctuation_but_keeps_hyphens(self):
        tokens = _tokenize("LiDAR-based (model: OS1-32).")
        # Parentheses, colon, and period stripped; hyphens kept
        assert "os1-32" in tokens
        assert "lidar-based" in tokens
        assert "model" in tokens

    def test_empty_string_returns_empty_list(self):
        assert _tokenize("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _tokenize("   \t\n  ") == []

    def test_preserves_internal_hyphens(self):
        tokens = _tokenize("all-MiniLM-L6-v2 sentence-transformers")
        assert "all-minilm-l6-v2" in tokens
        assert "sentence-transformers" in tokens

    def test_punctuation_heavy_slashes_and_brackets(self):
        tokens = _tokenize('"Ouster OS1-32" (LiDAR/sensor)?')
        assert "ouster" in tokens
        assert "os1-32" in tokens
        assert "lidar" in tokens
        assert "sensor" in tokens
        assert "lidarsensor" not in tokens

    def test_standalone_dashes_and_boundaries_stripped(self):
        tokens = _tokenize("pothole - detection: a survey [2024]!")
        assert "-" not in tokens
        assert tokens == ["pothole", "detection", "a", "survey", "2024"]

    def test_dashes_only_returns_empty_list(self):
        assert _tokenize("---") == []
        assert _tokenize(" - ") == []

    def test_leading_trailing_hyphens_stripped(self):
        assert _tokenize("--OS1-32--") == ["os1-32"]


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestLexicalRetrieverConstruction:
    def test_empty_corpus_builds_without_error(self):
        lr = LexicalRetriever([])
        assert lr.corpus_size == 0
        assert lr.avg_doc_length == 0.0

    def test_single_chunk_corpus(self):
        chunk = make_chunk("The Ouster OS1-32 LiDAR sensor was used.")
        lr = LexicalRetriever([chunk])
        assert lr.corpus_size == 1

    def test_type_error_on_non_chunk_input(self):
        with pytest.raises(TypeError, match="Chunk instances"):
            LexicalRetriever(["this is a plain string, not a Chunk"])

    def test_type_error_reports_index(self):
        good = make_chunk("valid chunk")
        with pytest.raises(TypeError, match="index 1"):
            LexicalRetriever([good, "bad element"])

    def test_repr_contains_corpus_size(self):
        chunks = [make_chunk("hello world", chunk_index=i) for i in range(3)]
        lr = LexicalRetriever(chunks)
        assert "corpus_size=3" in repr(lr)

    def test_avg_doc_length_is_positive_for_non_empty_corpus(self):
        chunks = [
            make_chunk("The quick brown fox", chunk_index=0),
            make_chunk("jumps over the lazy dog", chunk_index=1),
        ]
        lr = LexicalRetriever(chunks)
        assert lr.avg_doc_length > 0.0


# ---------------------------------------------------------------------------
# Search — edge cases
# ---------------------------------------------------------------------------


class TestSearchEdgeCases:
    def test_empty_corpus_returns_empty_list(self):
        lr = LexicalRetriever([])
        assert lr.search("Ouster OS1-32") == []

    def test_empty_query_returns_empty_list(self):
        chunk = make_chunk("The Ouster OS1-32 LiDAR sensor.")
        lr = LexicalRetriever([chunk])
        assert lr.search("") == []

    def test_whitespace_only_query_returns_empty_list(self):
        chunk = make_chunk("The Ouster OS1-32 LiDAR sensor.")
        lr = LexicalRetriever([chunk])
        assert lr.search("   ") == []

    def test_no_lexical_overlap_returns_empty_list(self):
        """Query shares no tokens with the corpus — should return []."""
        chunks = [
            make_chunk("The Ouster OS1-32 sensor captures point clouds.", chunk_index=0),
            make_chunk("LiDAR calibration uses rotation matrices.", chunk_index=1),
        ]
        lr = LexicalRetriever(chunks)
        # Query contains only completely novel tokens absent from all chunks
        results = lr.search("quantum entanglement photon polarisation")
        assert results == [], (
            f"Expected [] for zero-overlap query, got {results}"
        )

    def test_k_larger_than_corpus_returns_all_matching_chunks(self):
        """k clamped to corpus size; all positively-scored chunks returned.

        A fourth unrelated chunk ensures BM25 IDF is positive for the query
        tokens so BM25 returns results rather than triggering the <= 0 filter.
        """
        chunks = [
            make_chunk("Ouster OS1-32 LiDAR resolution 2048x32", chunk_index=i)
            for i in range(3)
        ]
        # Add a non-matching decoy so BM25 IDF is positive
        chunks.append(
            make_chunk(
                "Deep learning training on GPU with batch normalisation.",
                chunk_index=3,
            )
        )
        lr = LexicalRetriever(chunks)
        results = lr.search("Ouster OS1-32", k=100)
        # Only the 3 Ouster-matching chunks should return with positive scores;
        # the decoy does not contain 'Ouster' so it may or may not appear.
        assert len(results) >= 3, (
            f"Expected at least 3 results for 'Ouster OS1-32', got {len(results)}"
        )

    def test_k_equals_one_returns_single_best(self):
        chunks = [
            make_chunk("Ouster OS1-32 LiDAR sensor for pothole detection.", chunk_index=0),
            make_chunk("RGB camera captures image data at 30 fps.", chunk_index=1),
            make_chunk("Neural network training on GPU hardware.", chunk_index=2),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("Ouster OS1-32", k=1)
        assert len(results) == 1
        assert "os1-32" in results[0].text.lower() or "ouster" in results[0].text.lower()

    def test_duplicate_query_terms_do_not_crash(self):
        """Repeated query tokens are valid BM25 input and should not error.

        BM25Okapi computes IDF = log((N - df + 0.5) / (df + 0.5)).  With
        exactly N=2 documents, any term present in 1 doc gives IDF = log(1) =
        0, so all scores are zero.  Three chunks are required for IDF > 0.
        """
        chunk = make_chunk("Ouster OS1-32 Ouster sensor LiDAR Ouster.", chunk_index=0)
        decoy1 = make_chunk(
            "Deep neural network training on GPU hardware.", chunk_index=1
        )
        decoy2 = make_chunk(
            "RGB camera captures road surface images at 30 fps.", chunk_index=2
        )
        lr = LexicalRetriever([chunk, decoy1, decoy2])
        results = lr.search("Ouster Ouster Ouster OS1-32 OS1-32")
        assert len(results) >= 1, (
            "Duplicate query tokens should not crash or return empty when IDF > 0"
        )


# ---------------------------------------------------------------------------
# Search — exact technical terms
# ---------------------------------------------------------------------------


class TestExactTechnicalTerms:
    """Tests 1 & 2: Exact technical terms and rare identifiers rank first."""

    def test_ouster_os1_32_ranks_first(self):
        """Chunk containing 'Ouster OS1-32' must rank above generic chunks."""
        chunks = [
            make_chunk(
                "The system employs the Ouster OS1-32 having a resolution "
                "of 2048 x 32, and the sensor is mounted on the vehicle roof.",
                source_filename="karukayil.pdf",
                page_number=5,
                document_id="karukayil_doc",
                chunk_index=0,
            ),
            make_chunk(
                "LiDAR sensors are increasingly popular for road inspection "
                "tasks. They provide dense point cloud representations.",
                source_filename="karukayil.pdf",
                page_number=3,
                document_id="karukayil_doc",
                chunk_index=1,
            ),
            make_chunk(
                "Deep learning models such as YOLOv5 and YOLOv8 detect "
                "road defects in real time.",
                source_filename="paper.pdf",
                page_number=2,
                document_id="lam_doc",
                chunk_index=2,
            ),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("Ouster OS1-32", k=3)

        assert len(results) >= 1, "Expected at least one result."
        top = results[0]
        assert top.page_number == 5, (
            f"Expected rank-1 chunk from page 5 (Ouster text), "
            f"got page {top.page_number}. "
            f"Top chunk text: {top.text!r}"
        )
        assert top.source_filename == "karukayil.pdf"

    def test_iphone_14_pro_ranks_first(self):
        """'iPhone 14 Pro' must rank above generic smartphone mentions.

        BM25 requires at least one non-matching document for IDF to be
        positive.  A third unrelated chunk is included deliberately.
        """
        chunks = [
            make_chunk(
                "Data was collected using an iPhone 14 Pro equipped with an "
                "integrated LiDAR sensor in Lahore, Pakistan.",
                page_number=2,
                source_filename="butt.pdf",
                document_id="butt_doc",
                chunk_index=0,
            ),
            make_chunk(
                "Smartphone cameras provide RGB images at high resolution "
                "suitable for monocular depth estimation.",
                page_number=1,
                source_filename="butt.pdf",
                document_id="butt_doc",
                chunk_index=1,
            ),
            make_chunk(
                "Neural network training uses GPU hardware for acceleration.",
                page_number=6,
                source_filename="survey.pdf",
                document_id="survey_doc",
                chunk_index=2,
            ),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("iPhone 14 Pro LiDAR", k=3)
        assert len(results) > 0, "Expected BM25 to return results for 'iPhone 14 Pro LiDAR'"
        assert results[0].page_number == 2

    def test_mbtp_method_identifier_ranks_first(self):
        """Rare acronym 'MBTP' must surface the correct defining chunk.

        A third unrelated chunk is included so BM25 IDF is positive for 'MBTP'.
        """
        chunks = [
            make_chunk(
                "The Minimum Bounding Triangulated Pixel (MBTP) method "
                "approximates pothole area using an elliptical assumption.",
                page_number=7,
                source_filename="wang2025.pdf",
                document_id="wang_doc",
                chunk_index=0,
            ),
            make_chunk(
                "Pothole area estimation methods vary across the literature. "
                "Various geometric shapes have been proposed.",
                page_number=2,
                source_filename="wang2025.pdf",
                document_id="wang_doc",
                chunk_index=1,
            ),
            make_chunk(
                "GPU hardware accelerates neural network training for vision tasks.",
                page_number=4,
                source_filename="survey.pdf",
                document_id="survey_doc",
                chunk_index=2,
            ),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("MBTP Minimum Bounding Triangulated Pixel", k=3)
        assert len(results) > 0, "Expected BM25 to return results for 'MBTP'"
        assert results[0].page_number == 7

    def test_isprs_author_name_ranks_first(self):
        """Author name 'Hanzheng Wang' must surface the ISPRS abstract chunk."""
        chunks = [
            make_chunk(
                "Hanzheng Wang, Shishuo Xu, and Danyang Hu propose a "
                "binocular stereo vision system for pothole 3D recognition "
                "published in ISPRS Archives.",
                page_number=1,
                source_filename="isprs.pdf",
                document_id="isprs_doc",
                chunk_index=0,
            ),
            make_chunk(
                "Vehicle speed was evaluated at 15 km/h and 40 km/h. "
                "Relative depth errors were measured under each condition.",
                page_number=2,
                source_filename="ponnusamy.pdf",
                document_id="ponnusamy_doc",
                chunk_index=1,
            ),
            make_chunk(
                "Relative measurement errors decrease as vehicle speed "
                "increases in most depth sensing studies.",
                page_number=5,
                source_filename="butt.pdf",
                document_id="butt_doc",
                chunk_index=2,
            ),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("Hanzheng Wang ISPRS binocular stereo", k=3)
        assert results[0].source_filename == "isprs.pdf", (
            f"Expected 'isprs.pdf' at rank 1, got '{results[0].source_filename}'"
        )


# ---------------------------------------------------------------------------
# Search — provenance preservation
# ---------------------------------------------------------------------------


class TestProvenancePreservation:
    """Test 5: Verify document and page provenance survives retrieval."""

    def test_page_number_preserved_after_search(self):
        chunks = [
            make_chunk(
                "Ouster OS1-32 sensor mounts on vehicle rooftop rack.",
                page_number=5,
                source_filename="karukayil.pdf",
                document_id="karukayil_abc123",
                chunk_index=0,
            ),
            make_chunk(
                "RGB camera is synchronised with LiDAR via ROS2.",
                page_number=3,
                source_filename="karukayil.pdf",
                document_id="karukayil_abc123",
                chunk_index=1,
            ),
            make_chunk(
                "Neural network training on GPU hardware enables real-time detection.",
                page_number=7,
                source_filename="karukayil.pdf",
                document_id="karukayil_abc123",
                chunk_index=2,
            ),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("Ouster OS1-32", k=3)
        assert len(results) > 0, "Expected results for 'Ouster OS1-32'"

        top = results[0]
        assert top.page_number == 5
        assert top.source_filename == "karukayil.pdf"
        assert top.document_id == "karukayil_abc123"
        assert top.chunk_id == "karukayil_abc123_p5_c0"

    def test_chunk_id_unchanged_after_search(self):
        """chunk_id must be preserved exactly through indexing and retrieval.

        BM25Okapi requires N >= 3 documents for IDF to be positive (with N=2
        and df=1, IDF = log(1) = 0).  Two decoy chunks are included.
        """
        chunk = make_chunk(
            "The Ouster OS1-32 LiDAR has 32 beams.",
            source_filename="sensor_paper.pdf",
            page_number=7,
            document_id="sensor_doc",
            chunk_index=42,
        )
        decoy1 = make_chunk(
            "Deep learning models process image data using GPU acceleration.",
            source_filename="sensor_paper.pdf",
            page_number=1,
            document_id="sensor_doc",
            chunk_index=0,
        )
        decoy2 = make_chunk(
            "RGB camera captures high-resolution road surface images.",
            source_filename="sensor_paper.pdf",
            page_number=2,
            document_id="sensor_doc",
            chunk_index=1,
        )
        expected_id = chunk.chunk_id  # "sensor_doc_p7_c42"
        lr = LexicalRetriever([chunk, decoy1, decoy2])
        results = lr.search("Ouster OS1-32")
        assert len(results) > 0, "Expected results for 'Ouster OS1-32'"
        assert results[0].chunk_id == expected_id

    def test_score_is_float_and_positive_for_matching_chunk(self):
        """BM25 score is a float.  With >= 3 chunks the uniquely-matching
        chunk scores positively (IDF is positive when df < N)."""
        target = make_chunk(
            "The Ouster OS1-32 LiDAR sensor captures 3D point clouds.",
            chunk_index=0,
        )
        decoy1 = make_chunk(
            "RGB cameras provide high-resolution colour images at 30 fps.",
            chunk_index=1,
        )
        decoy2 = make_chunk(
            "Deep learning models trained on GPU hardware detect defects.",
            chunk_index=2,
        )
        lr = LexicalRetriever([target, decoy1, decoy2])
        results = lr.search("Ouster OS1-32")
        assert len(results) > 0, "Expected results for 'Ouster OS1-32'"
        assert isinstance(results[0].score, float)
        assert results[0].score > 0.0

    def test_multi_document_provenance_integrity(self):
        """Different documents must retain distinct source filenames and IDs.

        A third neutral chunk ensures BM25 IDF is positive for OS1-32.
        """
        chunk_a = make_chunk(
            "Ouster OS1-32 captures LiDAR data at 20 Hz.",
            source_filename="karukayil.pdf",
            page_number=5,
            document_id="doc_karukayil",
            chunk_index=0,
        )
        chunk_b = make_chunk(
            "Ouster is a leading LiDAR manufacturer for automotive sensors.",
            source_filename="survey.pdf",
            page_number=2,
            document_id="doc_survey",
            chunk_index=0,
        )
        neutral = make_chunk(
            "Deep learning models improve accuracy on benchmark datasets.",
            source_filename="other.pdf",
            page_number=1,
            document_id="doc_other",
            chunk_index=0,
        )
        lr = LexicalRetriever([chunk_a, chunk_b, neutral])
        results = lr.search("Ouster OS1-32", k=3)
        assert len(results) > 0, "Expected results for 'Ouster OS1-32'"
        filenames = [r.source_filename for r in results]
        assert "karukayil.pdf" in filenames
        # karukayil chunk has OS1-32 so it must be rank 1
        assert results[0].source_filename == "karukayil.pdf"
        # Confirm document_id integrity for each result
        for r in results:
            if r.source_filename == "karukayil.pdf":
                assert r.document_id == "doc_karukayil"
            elif r.source_filename == "survey.pdf":
                assert r.document_id == "doc_survey"
            else:
                assert r.document_id == "doc_other"

    def test_results_sorted_descending_by_score(self):
        chunks = [
            make_chunk(
                "Ouster OS1-32 LiDAR sensor 2048x32 resolution.",
                chunk_index=0,
            ),
            make_chunk(
                "Ouster OS1-32 is used for point cloud capture.",
                chunk_index=1,
            ),
            make_chunk(
                "Deep learning models improve detection accuracy.",
                chunk_index=2,
            ),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("Ouster OS1-32", k=3)
        # Scores must be non-increasing
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, (
                f"Score at rank {i} ({results[i].score}) < "
                f"score at rank {i+1} ({results[i+1].score})"
            )

    def test_indexed_chunks_not_mutated(self):
        """Search must return copies with scores; chunks in lr.chunks must NOT be mutated."""
        chunk = make_chunk(
            "Ouster OS1-32 LiDAR sensor for autonomous vehicles.", chunk_index=0
        )
        decoy1 = make_chunk("Deep learning models on GPU hardware.", chunk_index=1)
        decoy2 = make_chunk("Camera image capture at 30 fps.", chunk_index=2)
        assert chunk.score == 0.0

        lr = LexicalRetriever([chunk, decoy1, decoy2])
        results = lr.search("Ouster OS1-32", k=1)

        assert len(results) == 1
        assert results[0].score > 0.0
        # The original chunk object passed by the caller must still have score 0.0
        assert chunk.score == 0.0
        # The chunk stored in the retriever's index must still have score 0.0
        assert lr.chunks[0].score == 0.0


# ---------------------------------------------------------------------------
# Search — determinism and ties
# ---------------------------------------------------------------------------


class TestDeterminismAndTies:
    def test_identical_queries_produce_identical_rankings(self):
        chunks = [
            make_chunk("Ouster OS1-32 sensor", chunk_index=0),
            make_chunk("OS1-32 LiDAR unit", chunk_index=1),
            make_chunk("Unrelated text about pavement", chunk_index=2),
        ]
        lr = LexicalRetriever(chunks)
        run1 = lr.search("Ouster OS1-32", k=3)
        run2 = lr.search("Ouster OS1-32", k=3)
        assert [r.chunk_id for r in run1] == [r.chunk_id for r in run2]
        assert [r.score for r in run1] == [r.score for r in run2]

    def test_tied_scores_preserve_original_corpus_order(self):
        """When multiple chunks have identical BM25 scores, stable sort maintains index order.

        For df=2 to have positive IDF in BM25Okapi, N must be >= 5 (since
        (N - df + 0.5) / (df + 0.5) > 1 requires N > 2*df = 4).
        """
        chunks = [
            make_chunk("Identical target text for pothole detection.", chunk_index=0),
            make_chunk("Identical target text for pothole detection.", chunk_index=1),
            make_chunk("Unrelated alpha text to keep N large enough.", chunk_index=2),
            make_chunk("Unrelated beta text to keep N large enough.", chunk_index=3),
            make_chunk("Unrelated gamma text to keep N large enough.", chunk_index=4),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("Identical target", k=2)
        assert len(results) == 2
        assert results[0].chunk_index == 0
        assert results[1].chunk_index == 1
        assert results[0].score == results[1].score

    def test_all_common_terms_in_corpus_returns_empty_list(self):
        """When every document contains the query term, BM25 IDF goes <= 0, returning []."""
        chunks = [
            make_chunk("the road surface", chunk_index=0),
            make_chunk("the asphalt condition", chunk_index=1),
        ]
        lr = LexicalRetriever(chunks)
        results = lr.search("the")
        assert results == []


# ---------------------------------------------------------------------------
# Real-corpus integration test (requires data/papers/ to be present)
# ---------------------------------------------------------------------------

_KARUKAYIL_PDF = os.path.join(
    _REPO_ROOT,
    "data",
    "papers",
    "Deep_Learning_Enhanced_Feature_Extraction_of_Potholes_Using_Vision_and_LiDAR_Data_for_Road_Maintenance.pdf",
)


@pytest.mark.skipif(
    not os.path.exists(_KARUKAYIL_PDF),
    reason="Karukayil et al. PDF not present in data/papers/",
)
class TestRealCorpusOusterRanking:
    """Integration test: BM25 over real ingested chunks for Q04 scenario."""

    @pytest.fixture(scope="class")
    def karukayil_chunks(self):
        """Ingest the Karukayil et al. PDF and return all chunks."""
        from src.document_processor import DocumentProcessor, get_nlp

        nlp = get_nlp()
        processor = DocumentProcessor(sentences_per_chunk=3)
        chunks = processor.process_file(_KARUKAYIL_PDF, nlp)
        assert len(chunks) > 0, "Karukayil PDF produced zero chunks."
        return chunks

    def test_ouster_os1_32_ranks_first_in_real_corpus(self, karukayil_chunks):
        """
        BM25 query 'Ouster OS1-32' must surface a chunk from physical page 5
        (where the sensor is explicitly named) at rank 1.

        If this test fails, the Ouster text on page 5 was not indexed or was
        tokenised in a way that prevents matching.  Investigate rather than
        weakening the assertion.
        """
        lr = LexicalRetriever(karukayil_chunks)
        results = lr.search("Ouster OS1-32", k=5)

        assert len(results) > 0, (
            "BM25 returned no results for 'Ouster OS1-32' over the real "
            "Karukayil corpus.  Check tokenisation and chunk content."
        )

        top = results[0]

        # Diagnose what was retrieved for a clear failure message.
        retrieved_pages = [(r.page_number, r.score, r.text[:80]) for r in results]

        assert top.page_number == 5, (
            f"Expected rank-1 chunk from physical page 5 (Ouster OS1-32 text), "
            f"but got page {top.page_number}.\n"
            f"Top-5 retrieved:\n"
            + "\n".join(
                f"  rank {i+1}: p.{pg} score={sc:.4f} | {txt!r}"
                for i, (pg, sc, txt) in enumerate(retrieved_pages)
            )
        )

        assert "ouster" in top.text.lower() or "os1" in top.text.lower(), (
            f"Rank-1 chunk on page {top.page_number} does not contain 'Ouster' "
            f"or 'OS1'.  Text: {top.text!r}"
        )

    def test_real_corpus_provenance_preserved(self, karukayil_chunks):
        """All retrieved chunks retain valid page numbers and source filename."""
        lr = LexicalRetriever(karukayil_chunks)
        results = lr.search("LiDAR pothole detection YOLOv8", k=5)

        for r in results:
            assert isinstance(r.page_number, int) and r.page_number >= 1
            assert r.source_filename.endswith(".pdf")
            assert r.document_id  # non-empty
            assert r.chunk_id  # non-empty
