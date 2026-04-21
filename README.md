# 📚 Intelligent Question Answering System

A RAG (Retrieval-Augmented Generation) based Question Answering system built with **Streamlit**, **FAISS**, **Sentence Transformers**, and **Google Gemini**.

Upload any PDF and ask questions — the system retrieves the most relevant passages and uses Gemini AI to generate accurate, synthesized answers.

---

## ✨ Features

- 📄 **PDF Upload** — Extract and process text from any PDF
- 🔍 **Semantic Search** — FAISS vector search with `all-MiniLM-L6-v2` embeddings
- 🤖 **AI Answers** — Gemini 1.5 Flash generates natural language answers from retrieved context
- 📊 **Relevance Scoring** — Hybrid scoring (semantic + keyword match)
- 🗂️ **Q&A History** — Persists previous questions and answers across sessions
- 🔒 **Secure Key Storage** — API key stored locally via Streamlit secrets

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| NLP Preprocessing | spaCy |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (IndexFlatIP) |
| LLM | Google Gemini 1.5 Flash |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Intelligent-Question-Answering-System.git
cd Intelligent-Question-Answering-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set up your Gemini API key
Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey), then:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and paste your key:
```toml
GEMINI_API_KEY = "your-key-here"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── history.json                  # Q&A history (auto-generated, gitignored)
├── src/
│   ├── document_processor.py     # PDF parsing & chunking with spaCy
│   ├── qa_engine.py              # FAISS vector index & semantic search
│   └── llm_engine.py             # Gemini LLM answer generation
└── .streamlit/
    ├── secrets.toml              # Your API key (gitignored, never committed)
    └── secrets.toml.example      # Template for others to follow
```

---

## ⚠️ Important

- **Never commit `.streamlit/secrets.toml`** — it contains your API key and is already in `.gitignore`
- The app works **without a Gemini key** (falls back to extracted answers), but AI-generated answers are much better

---

## 📄 License

MIT License