import spacy
import fitz  # PyMuPDF

def get_nlp():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


class Chunk:
    def __init__(self, text, source):
        self.text = text
        self.source = source
        self.pos_tags = []


class DocumentProcessor:
    def __init__(self):
        self.chunks = []

    def process_file(self, uploaded_file, nlp):
        text = ""

        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()

        text = text[:80000]

        doc_spacy = nlp(text)
        sentences = [sent.text for sent in doc_spacy.sents]

        chunks = []

        # 🔥 SMALLER chunks = better retrieval
        for i in range(0, len(sentences), 3):
            chunk_text = " ".join(sentences[i:i+3])

            if len(chunk_text) > 40:
                chunks.append(Chunk(chunk_text, uploaded_file.name))

        # 🔥 Increase chunk count
        self.chunks = chunks[:200]

        for chunk in self.chunks:
            doc = nlp(chunk.text)
            chunk.pos_tags = [(token.text, token.pos_) for token in doc[:10]]