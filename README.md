# 📚 Intelligent Question Answering System

A RAG (Retrieval-Augmented Generation) based Question Answering system built with **Streamlit**, **FAISS**, **Sentence Transformers**, and **Ollama (local LLM)**.

Upload any PDF and ask questions — the system retrieves the most relevant passages and uses a local AI model to generate accurate, synthesized answers. **No API key required. Completely free and private.**

---

## ✨ Features

- 📄 **PDF Upload** — Extract and process text from any PDF
- 🔍 **Semantic Search** — FAISS vector search with `all-MiniLM-L6-v2` embeddings
- 🦙 **Local AI Answers** — Ollama runs LLMs locally (llama3.2, mistral, phi3, etc.)
- ⚡ **Streaming** — Answers appear word-by-word in real time
- 📊 **Relevance Filtering** — Hybrid scoring (semantic + keyword match), threshold ≥ 0.40
- 🗂️ **Q&A History** — Persists previous questions and answers across sessions
- 🔒 **100% Private** — Everything runs on your machine, no data sent anywhere

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| NLP Preprocessing | spaCy |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (IndexFlatIP) |
| LLM | Ollama (llama3.2 / any local model) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Shivangid2904/Intelligent-Question-Answering-System.git
cd Intelligent-Question-Answering-System
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Install Ollama
Download and install from [ollama.com/download](https://ollama.com/download), then pull a model:
```bash
ollama pull llama3.2:1b   # Fast, lightweight (~700MB)
# OR
ollama pull llama3.2      # Better quality (~2GB)
```

### 4. Run the app
```bash
streamlit run app.py
```

Ollama runs automatically in the background after installation — no extra setup needed!

---

## 📁 Project Structure

```
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── history.json                  # Q&A history (auto-generated, gitignored)
└── src/
    ├── document_processor.py     # PDF parsing & chunking with spaCy
    ├── qa_engine.py              # FAISS vector index & semantic search
    └── llm_engine.py             # Ollama LLM streaming answer generation
```

---

## 💡 How It Works

```
PDF Upload → Text Extraction → spaCy Chunking
    → Sentence Embeddings → FAISS Index
    → Query → Semantic Search → Relevance Filter
    → Ollama LLM → Streamed Answer
```

---

## ⚙️ Model Selection

In the sidebar, you can switch between models:

| Model | Size | Speed | Quality |
|---|---|---|---|
| `llama3.2:1b` | ~700MB | ⚡ Fast | Good |
| `llama3.2` | ~2GB | 🐢 Moderate | Better |
| `mistral` | ~4GB | 🐢 Slower | Great |
| `phi3` | ~2GB | ⚡ Fast | Good |

---

## 📄 License

MIT License