import re
import io
from typing import List
import spacy

nlp = spacy.load("en_core_web_sm")


class Chunk:
    def __init__(self, text, source):
        self.text = text
        self.source = source
        self.tokens = []
        self.pos_tags = []
        self.lemmas = []


class DocumentProcessor:
    def __init__(self):
        self.chunks = []

    def process_file(self, uploaded_file):
        import fitz  # PyMuPDF

        text = ""
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()

        text = self.clean_text(text)
        sentences = self.split_sentences(text)

        chunks = self.create_chunks(sentences, uploaded_file.name)
        self.annotate_chunks(chunks)

        self.chunks.extend(chunks)

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_sentences(self, text):
        doc = nlp(text[:100000])
        return [sent.text for sent in doc.sents]

    def create_chunks(self, sentences, source):
        chunks = []
        size = 5
        overlap = 2

        for i in range(0, len(sentences), size - overlap):
            chunk_text = " ".join(sentences[i:i+size])
            if len(chunk_text) > 30:
                chunks.append(Chunk(chunk_text, source))

        return chunks

    def annotate_chunks(self, chunks):
        for doc, chunk in zip(nlp.pipe([c.text for c in chunks]), chunks):
            chunk.tokens = [t.text for t in doc]
            chunk.pos_tags = [(t.text, t.pos_) for t in doc]
            chunk.lemmas = [t.lemma_ for t in doc if not t.is_stop]