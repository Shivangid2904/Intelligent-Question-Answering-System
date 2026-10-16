import hashlib
import os
import re
from typing import Any, List, Optional, Union
import fitz  # PyMuPDF
import spacy


def get_nlp():
    """
    Load spaCy English pipeline with sentencizer for fast sentence boundary detection.
    Disables NER and full dependency parser for optimal throughput.
    """
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


class Chunk:
    """
    Represents a page-bounded text chunk extracted from a scientific document.
    Maintains provenance metadata linking text to physical PDF page numbers.
    """

    def __init__(
        self,
        text: str,
        source_filename: str,
        page_number: int,
        document_id: str,
        chunk_index: int,
        chunk_id: Optional[str] = None,
        page_chunk_index: int = 0,
        pos_tags: Optional[List[Any]] = None,
    ):
        self.text = text
        self.source_filename = source_filename
        self.page_number = int(page_number)  # 1-based physical PDF page index
        self.document_id = document_id
        self.chunk_index = int(chunk_index)
        self.page_chunk_index = int(page_chunk_index)
        self.chunk_id = (
            chunk_id
            if chunk_id
            else f"{document_id}_p{self.page_number}_c{self.chunk_index}"
        )
        self.pos_tags = pos_tags if pos_tags is not None else []
        self.score: float = 0.0

        # Backwards compatibility alias for app.py and legacy components
        self.source = source_filename

    @property
    def metadata(self) -> dict:
        return {
            "document_id": self.document_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_chunk_index": self.page_chunk_index,
            "score": self.score,
        }

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", " ")
        return f"<Chunk {self.chunk_id} p.{self.page_number} ({len(self.text)} chars): '{snippet}...'>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return False
        return self.chunk_id == other.chunk_id and self.text == other.text

    def __hash__(self) -> int:
        return hash((self.chunk_id, self.text))


class DocumentProcessor:
    """
    Extracts text page-by-page from PDF documents and generates page-bounded,
    provenance-preserving chunks for dense and hybrid retrieval.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 0,
        min_chunk_chars: int = 40,
        max_chunks: Optional[int] = None,
    ):
        """
        Args:
            sentences_per_chunk: Number of consecutive sentences per chunk.
            sentence_overlap: Number of overlapping sentences between consecutive chunks.
            min_chunk_chars: Minimum character length for a valid chunk.
            max_chunks: Optional upper safety bound on total chunks (None = unlimited).
        """
        self.sentences_per_chunk = max(1, sentences_per_chunk)
        self.sentence_overlap = max(0, min(sentence_overlap, self.sentences_per_chunk - 1))
        self.min_chunk_chars = min_chunk_chars
        self.max_chunks = max_chunks

        self.chunks: List[Chunk] = []
        self.document_id: str = ""
        self.source_filename: str = ""
        self.page_count: int = 0
        self.skipped_pages: List[int] = []

    def _compute_document_id(self, file_content: bytes, filename: str) -> str:
        """
        Generates a stable, deterministic document identifier based on
        content hash combined with sanitized filename slug.
        """
        content_hash = hashlib.sha256(file_content).hexdigest()[:12]
        clean_name = re.sub(r"[^\w\-.]", "_", os.path.splitext(filename)[0])[:30]
        return f"{clean_name}_{content_hash}"

    def process_file(
        self,
        uploaded_file: Any,
        nlp: Any,
    ) -> List[Chunk]:
        """
        Process a PDF document into page-bounded, provenance-preserving chunks.

        Args:
            uploaded_file: Streamlit UploadedFile, file-like object with .read(),
                           bytes buffer, or string/Path to a local PDF file.
            nlp: spaCy Language model instance with sentencizer.

        Returns:
            List of Chunk instances with 1-based page numbers and metadata.
        """
        self.chunks = []
        self.skipped_pages = []

        # Determine source filename and read binary content
        if isinstance(uploaded_file, (str, os.PathLike)):
            filepath = str(uploaded_file)
            self.source_filename = os.path.basename(filepath)
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        elif hasattr(uploaded_file, "read"):
            self.source_filename = getattr(uploaded_file, "name", "document.pdf")
            file_bytes = uploaded_file.read()
            # Reset pointer if seekable for caller reuse
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        elif isinstance(uploaded_file, bytes):
            self.source_filename = "document.pdf"
            file_bytes = uploaded_file
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        else:
            raise ValueError(f"Unsupported file input type: {type(uploaded_file)}")

        self.document_id = self._compute_document_id(file_bytes, self.source_filename)
        self.page_count = len(doc)

        global_chunk_idx = 0
        chunks: List[Chunk] = []

        step = max(1, self.sentences_per_chunk - self.sentence_overlap)

        # Process each page independently to ensure page-bounded provenance
        for page_idx, page in enumerate(doc, start=1):
            page_text = page.get_text()

            if not page_text or not page_text.strip():
                self.skipped_pages.append(page_idx)
                continue

            # Normalize common whitespace artifacts while preserving text content
            normalized_text = page_text.replace("\r\n", "\n").replace("\r", "\n")

            doc_spacy = nlp(normalized_text)
            sentences = [sent.text.strip() for sent in doc_spacy.sents if sent.text.strip()]

            if not sentences:
                self.skipped_pages.append(page_idx)
                continue

            page_chunk_idx = 0
            for i in range(0, len(sentences), step):
                chunk_sentences = sentences[i : i + self.sentences_per_chunk]
                chunk_text = " ".join(chunk_sentences).strip()

                if len(chunk_text) >= self.min_chunk_chars:
                    chunk = Chunk(
                        text=chunk_text,
                        source_filename=self.source_filename,
                        page_number=page_idx,
                        document_id=self.document_id,
                        chunk_index=global_chunk_idx,
                        page_chunk_index=page_chunk_idx,
                    )

                    # Compute POS tags on the leading tokens for backwards compatibility
                    try:
                        token_doc = nlp(chunk.text[:200])
                        chunk.pos_tags = [(token.text, token.pos_) for token in token_doc[:10]]
                    except Exception:
                        chunk.pos_tags = []

                    chunks.append(chunk)
                    global_chunk_idx += 1
                    page_chunk_idx += 1

                    if self.max_chunks and len(chunks) >= self.max_chunks:
                        break

            if self.max_chunks and len(chunks) >= self.max_chunks:
                break

        doc.close()
        self.chunks = chunks
        return self.chunks