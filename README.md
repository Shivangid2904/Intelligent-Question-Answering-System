# IntelliAsk – Intelligent Question Answering System

IntelliAsk is a Retrieval-Augmented Generation (RAG) based Question Answering System that enables users to upload PDF documents and ask natural language questions. The application retrieves the most relevant document passages using semantic search and generates context-aware answers using a locally hosted Large Language Model (LLM) through Ollama.

Unlike cloud-based AI applications, IntelliAsk runs entirely on the user's machine, requiring no API keys or external services, thereby ensuring complete privacy and offline functionality.

---

## Features

- Upload and process PDF documents
- Semantic search using FAISS vector indexing
- Context-aware answer generation using local LLMs
- Real-time streaming responses
- Hybrid relevance scoring (semantic + keyword matching)
- Supporting passage visualization with similarity scores
- Persistent question history
- Multiple Ollama model support
- Fully offline and privacy-preserving architecture

---

## Screenshots
### Home Interface

<img width="955" alt="Home Interface" src="https://github.com/user-attachments/assets/24a3cff5-cc30-4230-9db7-b74dfff3de0c">

The landing page provides an intuitive interface for selecting a local language model, uploading PDF documents, and asking natural language questions.

---

### Document Upload & Model Selection

<img width="955" alt="Document Upload" src="https://github.com/user-attachments/assets/fb812e78-6dc3-4c12-bb38-f34369f3cf72">

Users can upload PDF documents and switch between multiple locally installed Ollama models without restarting the application.

---

### AI Generated Answer

<img width="955" alt="Generated Answer" src="https://github.com/user-attachments/assets/f2406a4a-9dc5-4800-99e8-b2b838fa164c">

The system retrieves the most relevant document passages using semantic similarity search and synthesizes an answer using a locally hosted Large Language Model.

---

### Supporting Passages & Question History

<img width="850" alt="Supporting Passages" src="https://github.com/user-attachments/assets/e9e0badf-1d1c-488f-a1a1-0f464a06c081">

Every response includes supporting document passages, semantic relevance scores, and persistent question history to improve transparency and explainability.

---

## Demo Video

A complete walkthrough of the application is available below.

**Demo:** 
**[Watch Demo Video](https://drive.google.com/file/d/1q8-x_RziRFKCrJf_UVMTwpO0EpHAUsKa/view?usp=sharing)**


---

## Architecture

```text
                    PDF Upload
                         │
                  PyMuPDF Extraction
                         │
                  spaCy Text Chunking
                         │
         Sentence Transformer Embeddings
                         │
                  FAISS Vector Index
                         │
             Semantic Similarity Retrieval
                         │
              Relevant Context Selection
                         │
                 Ollama Local LLM
                         │
             Streamed AI Generated Answer
                         │
                Streamlit User Interface
```

---

## Technology Stack

| Layer | Technology |
|--------|------------|
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| NLP Preprocessing | spaCy |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Large Language Model | Ollama |
| Supported Models | Llama 3.2, Mistral, Phi-3, Gemma |
| Programming Language | Python |

---

## Project Workflow

```text
PDF Upload
      │
      ▼
Text Extraction
      │
      ▼
Text Chunking
      │
      ▼
Sentence Embeddings
      │
      ▼
FAISS Index Construction
      │
      ▼
User Question
      │
      ▼
Semantic Retrieval
      │
      ▼
Relevant Context
      │
      ▼
Local LLM (Ollama)
      │
      ▼
AI Generated Answer
```

---

## Project Structure

```text
Intelligent-Question-Answering-System/

├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── llm_engine.py
│   └── qa_engine.py
│
├── .gitignore
├── app.py
├── history.json
├── README.md
└── requirements.txt
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Shivangid2904/Intelligent-Question-Answering-System.git
cd Intelligent-Question-Answering-System
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

If required, install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

### Install Ollama

Download Ollama from:

https://ollama.com/download

Pull a supported model:

```bash
ollama pull llama3.2
```

or

```bash
ollama pull llama3.2:1b
```

---

### Run the Application

```bash
streamlit run app.py
```

---

## Supported Models

| Model | Approx. Size | Speed | Quality |
|--------|-------------:|-------|----------|
| llama3.2:1b | ~700 MB | Fast | Good |
| llama3.2 | ~2 GB | Moderate | Better |
| phi3 | ~2 GB | Fast | Good |
| mistral | ~4 GB | Moderate | Great |
| gemma2 | Varies | Moderate | Good |

---

## Core Components

### Document Processing

- PDF parsing using PyMuPDF
- Text preprocessing with spaCy
- Intelligent document chunking

### Semantic Retrieval

- Sentence embeddings with all-MiniLM-L6-v2
- FAISS vector similarity search
- Hybrid relevance filtering

### Local AI Inference

- Multiple Ollama model support
- Context-aware answer generation
- Token streaming

### User Experience

- Interactive Streamlit interface
- Persistent Q&A history
- Supporting passage visualization
- Semantic relevance scoring

---

## Advantages

- Runs completely offline
- No API keys required
- Privacy-preserving architecture
- Fast semantic retrieval
- Multiple local LLM support
- Modular and extensible codebase
- Easy deployment

---

## Future Improvements

- Multi-document knowledge base
- OCR support for scanned PDFs
- Citation-aware answer generation
- Conversation memory
- Document summarization
- Docker deployment
- User authentication
- Advanced reranking models

---

## License

This project is licensed under the MIT License.

---

## Author

**Shivangi Dubey**
