# IntelliAsk Research — System Architecture & Technical Design

This document details the software architecture, design decisions, index-lifecycle management, and system boundaries of **IntelliAsk Research**, contrasting the existing prototype implementation with the target V1 architecture.

---

## 1. Architectural Philosophy

IntelliAsk Research is engineered as a **general-purpose, local-first scientific literature analysis platform** that helps researchers analyze, synthesize, and compare scientific papers using traceable, evidence-grounded retrieval. The design adheres to three core tenets:

1. **Local-First Execution**: Ingestion, vector embedding, similarity search, and model inference run locally on user hardware via Ollama and local embedding models. Local processing eliminates the transmission of research manuscripts to external cloud APIs, though individual host environments still require standard operating system access controls.
2. **Deterministic Evidence Traceability & Calibrated Abstention**: Answers cite verified document source locations (paper and page level). If retrieved context is inadequate or ambiguous, the system is designed to abstain rather than synthesize speculative claims. Neither citation parsing nor sufficiency checks constitute a mathematical guarantee of factual accuracy; they are structured safeguards designed to reduce unsupported answers.
3. **Reproducible Evaluation**: Retrieval and generation pipelines are treated as empirical systems measured across standard quantitative information retrieval (IR) and grounding metrics on curated research corpora before claiming performance improvements.

---

## 2. Current Implementation Analysis

The existing codebase consists of an interactive Streamlit application (`app.py`) backed by three modular source files in `src/`.

### Component Inspection

| File | Implemented Functionality | Architectural Characteristics & Constraints |
| :--- | :--- | :--- |
| [`app.py`](file:///c:/Users/SHIVANGI/OneDrive/Documents/Desktop/Projects/Intelligent-Question-Answering-System/app.py) | UI presentation, session state, basic keyword-boosting heuristic, response streaming, JSON history | Single-page Streamlit app. Monolithic query block combining FAISS retrieval, threshold filtering (`SCORE_THRESHOLD = 0.38`), token matching heuristic, and Ollama streaming. |
| [`src/document_processor.py`](file:///c:/Users/SHIVANGI/OneDrive/Documents/Desktop/Projects/Intelligent-Question-Answering-System/src/document_processor.py) | `DocumentProcessor`, `Chunk`, `get_nlp()` | Uses PyMuPDF (`fitz`) to extract text across pages into a single string (`text += page.get_text()`). Truncates text at 80,000 characters. Splits sentences using spaCy's `sentencizer`. Creates 3-sentence sliding windows (capped at 200 chunks). **Page boundaries are discarded during extraction.** |
| [`src/qa_engine.py`](file:///c:/Users/SHIVANGI/OneDrive/Documents/Desktop/Projects/Intelligent-Question-Answering-System/src/qa_engine.py) | `QAEngine`, `get_embedding_model()` | Loads `sentence-transformers/all-MiniLM-L6-v2`. Computes L2-normalized embeddings. Constructs an in-memory `faiss.IndexFlatIP` (inner product = cosine similarity). Performs top-$k$ similarity queries. |
| [`src/llm_engine.py`](file:///c:/Users/SHIVANGI/OneDrive/Documents/Desktop/Projects/Intelligent-Question-Answering-System/src/llm_engine.py) | `generate_answer()` | Connects to local Ollama REST server (`http://localhost:11434/api/generate`). Formats top-3 retrieved passages into a structured prompt. Streams tokens via Python generator to Streamlit. |
| `history.json` | Persistent query history | Flat JSON array storing questions, generated answers, top passages, and similarity scores. Appended on every query. |

### Existing Prototype Data Flow Diagram

The diagram below reflects the **actual** flow implemented in the current repository:

```mermaid
flowchart TD
    subgraph Client ["Client Browser"]
        UI["Streamlit UI (app.py)"]
    end

    subgraph Ingestion ["Ingestion Pipeline (document_processor.py)"]
        PDF["PDF File Upload"] --> FITZ["PyMuPDF (fitz)<br/>Page Text Extraction"]
        FITZ --> TRUNC["Text Buffer<br/>(Capped at 80,000 chars)"]
        TRUNC --> SPACY["spaCy Sentencizer<br/>Sentence Segmentation"]
        SPACY --> CHUNK["3-Sentence Chunking<br/>(Capped at 200 chunks)"]
    end

    subgraph Indexing ["Embedding & Indexing (qa_engine.py)"]
        CHUNK --> EMB["all-MiniLM-L6-v2<br/>Normalized Embeddings"]
        EMB --> FAISS["faiss.IndexFlatIP<br/>(In-Memory Cosine Index)"]
    end

    subgraph Retrieval ["Query & Retrieval (app.py)"]
        QUERY["User Question"] --> QEMB["Encode Question<br/>(all-MiniLM-L6-v2)"]
        QEMB --> SEARCH["FAISS Search (k=10)"]
        FAISS --> SEARCH
        SEARCH --> BOOST["Heuristic Keyword Boost<br/>Score >= 0.38 & Word Overlap"]
        BOOST --> TOP3["Select Top 3 Passages"]
    end

    subgraph Generation ["Local Inference (llm_engine.py)"]
        TOP3 --> PROMPT["Context Assembly<br/>([Passage 1..3])"]
        PROMPT --> OLLAMA["Ollama Local REST API<br/>(/api/generate)"]
        OLLAMA --> STREAM["Token Stream Generator"]
    end

    STREAM --> UI
    STREAM --> HIST["Append to history.json"]

    classDef current fill:#1e293b,stroke:#3b82f6,stroke-width:1px,color:#f8fafc;
    class UI,PDF,FITZ,TRUNC,SPACY,CHUNK,EMB,FAISS,QUERY,QEMB,SEARCH,BOOST,TOP3,PROMPT,OLLAMA,STREAM,HIST current;
```

---

## 3. Architecture Gaps & V1 Target State

To evolve IntelliAsk into a platform suitable for scientific research literature, several core technical limitations must be reconciled:

| Functional Area | Current Implementation Gap | Target V1 Architecture Solution |
| :--- | :--- | :--- |
| **Document Granularity** | Single text buffer without page retention; 80,000 character cutoff; arbitrary 200-chunk cap. | Page-aware chunking preserving `page_number` (or `page_start`/`page_end`), bounding metadata, and document ID without arbitrary truncation. |
| **Document Scope** | Single PDF in-memory index; uploading a new PDF overwrites state. | Multi-document project collections stored in persistent SQLite tables with derived FAISS vector indices. |
| **Retrieval Strategy** | Bi-encoder FAISS search + rudimentary word-match booster heuristic. | Two-stage hybrid retrieval: Sparse BM25 + Dense FAISS fused via Reciprocal Rank Fusion (RRF), followed by optional Cross-Encoder reranking. |
| **Citations & Grounding** | Passages presented with similarity scores; no page-level citations in LLM answer. | Prompt-enforced page citation syntax (`[Paper, p. X]`) validated against retrieved context chunks. |
| **Evidence Sufficiency** | Rudimentary fallback if score < 0.38; LLM may hallucinate on low-relevance context. | Explicit evidence sufficiency gate that triggers an abstention response when evidence is incomplete or missing. |
| **Structured Intelligence** | No structured paper metadata or synthesis across documents. | Pydantic-validated JSON extraction for literature matrices (objectives, methods, datasets, findings) with CSV export. |
| **Persistence & Lifecycle** | Flat `history.json` file; no deletion or vector synchronization. | Normalized SQLite database as canonical store; derived FAISS and BM25 indices synchronized or rebuilt on collection changes. |
| **Service Decoupling** | Monolithic query handling inside `app.py`. | Modular Python service packages (`src/ingestion/`, `src/retrieval/`, `src/synthesis/`, `src/storage/`) invoked directly by Streamlit without premature API microservices. |

---

## 4. Planned V1 Architecture

The target V1 system organizes functionality into modular Python services without requiring FastAPI as a prerequisite. Streamlit directly coordinates the services while SQLite serves as the canonical store of truth.

The domain-independent retrieval and answer flow is:
$$\text{PDF Ingestion} \rightarrow \text{Page-Aware Chunks} \rightarrow \text{Dense \& Lexical Retrieval} \rightarrow \text{Rank Fusion} \rightarrow \text{Optional Reranking} \rightarrow \text{Sufficiency Check} \rightarrow \text{Answer / Abstention}$$

```mermaid
flowchart TB
    subgraph UI_Layer ["Presentation Layer (Streamlit)"]
        UI_QA["Q&A & Citation Inspector"]
        UI_DOCS["Collection Manager"]
        UI_MATRIX["Literature Matrix & CSV Export"]
        UI_EVAL["Evaluation Dashboard"]
    end

    subgraph Service_Layer ["Modular Python Service Layer (In-Process)"]
        INGEST["Ingestion Service<br/>(Page-Aware Parsing)"]
        RETRIEVAL["Hybrid Retrieval Service<br/>(BM25 + FAISS + RRF)"]
        RERANK["Reranking Service<br/>(Cross-Encoder / Optional)"]
        LLM_SERV["Grounded Inference Service"]
        EXTRACT["Structured Extraction Service<br/>(Pydantic Validation)"]
        EVAL_SERV["Evaluation Benchmark Runner"]
    end

    subgraph Retrieval_Engine ["Planned Retrieval Pipeline"]
        direction TB
        subgraph Stage1 ["Stage 1: Candidate Generation"]
            BM25["BM25 Okapi<br/>(Keyword Index)"]
            DENSE["FAISS IndexFlatIP<br/>(Dense Semantic Search)"]
        end
        RRF["Reciprocal Rank Fusion (RRF)"]
        Stage1 --> RRF
        subgraph Stage2 ["Stage 2: Cross-Encoder Reranking (Optional)"]
            CE["cross-encoder/ms-marco-MiniLM-L-6-v2<br/>(Relative Ranking Signal)"]
        end
        RRF --> CE
        CE --> GATE{"Evidence Sufficiency<br/>Verification Gate"}
    end

    subgraph Model_Layer ["Local Model Layer"]
        SENT_TRANS["Sentence Transformers<br/>(all-MiniLM-L6-v2)"]
        OLLAMA_SRV["Ollama Server<br/>(llama3.2 / mistral / phi3)"]
    end

    subgraph Storage_Layer ["Canonical & Derived Storage"]
        SQLITE[("Canonical Store: SQLite Database<br/>Collections, Documents, Chunks, Metadata, Queries")]
        FAISS_DISK[("Derived Index: FAISS File<br/>(Reconstructible from Chunks)")]
    end

    UI_DOCS --> INGEST
    INGEST --> SQLITE
    INGEST --> SENT_TRANS
    SENT_TRANS --> FAISS_DISK

    UI_QA --> RETRIEVAL
    RETRIEVAL --> BM25
    RETRIEVAL --> DENSE
    FAISS_DISK --> DENSE
    SQLITE --> BM25

    GATE -- "Sufficient Evidence" --> LLM_SERV
    GATE -- "Insufficient Evidence" --> ABSTAIN["Calibrated Abstention Generator"]

    LLM_SERV --> OLLAMA_SRV
    LLM_SERV --> UI_QA
    ABSTAIN --> UI_QA

    UI_MATRIX --> EXTRACT
    EXTRACT --> OLLAMA_SRV
    EXTRACT --> SQLITE

    UI_EVAL --> EVAL_SERV
    EVAL_SERV --> RETRIEVAL
    EVAL_SERV --> SQLITE

    classDef planned fill:#0f172a,stroke:#10b981,stroke-width:1px,color:#f8fafc;
    classDef existing fill:#1e293b,stroke:#3b82f6,stroke-width:1px,color:#f8fafc;
    classDef storage fill:#1e1b4b,stroke:#8b5cf6,stroke-width:1px,color:#f8fafc;

    class UI_QA,DENSE,SENT_TRANS,OLLAMA_SRV existing;
    class UI_DOCS,UI_MATRIX,UI_EVAL,INGEST,RETRIEVAL,RERANK,LLM_SERV,EXTRACT,EVAL_SERV,BM25,RRF,CE,GATE,ABSTAIN planned;
    class SQLITE,FAISS_DISK storage;
```

---

## 5. Document Ingestion & Page-Aware Chunking Strategy

### Page-Bounded Chunks for V1
To guarantee that citations resolve to unambiguous physical pages, V1 enforces **page-bounded chunking**:
1. PyMuPDF extracts text page-by-page, retaining the 1-indexed physical PDF page number.
2. Text within each page is split into semantic paragraphs/sentences via spaCy.
3. Sliding chunk windows (typically 200–350 words with 30–50 word overlap) are constrained to remain within the boundaries of a single page:
   $$\text{Chunk}_{\text{id}} = \text{doc\_id} + \text{"\_p"} + \text{page\_num} + \text{"\_c"} + \text{chunk\_index}$$
4. If a single uninterrupted section spans across a page boundary, chunks record both `page_start` and `page_end` in chunk metadata.

```mermaid
flowchart TD
    A["PDF Document Upload"] --> B["Format & Header Validation"]
    B --> C{"Is Searchable PDF?"}
    
    C -- "Yes (Digital Text)" --> D["PyMuPDF (fitz)<br/>Page-by-Page Extraction"]
    C -- "No (Scanned Images)" --> E["Optional Tesseract OCR Path<br/>[Planned / Deferred]"]
    
    D --> F["Text Normalization<br/>Strip Running Headers & Fix Hyphenation"]
    E --> F
    
    F --> G["Page-Bounded Chunking<br/>(200-350 words, Page-Constrained)"]
    G --> H["Assign Stable Identifiers & Metadata<br/>{doc_id, page_number, chunk_id, text}"]
    
    H --> I["Batch Encode with all-MiniLM-L6-v2"]
    H --> J["Tokenize for BM25 Lexical Index"]
    
    I --> K["Append / Sync Vectors in FAISS Index"]
    J --> L["Commit Canonical Chunks & Metadata to SQLite"]
    K --> M["Ready for Research Queries"]
    L --> M

    classDef stage fill:#1e293b,stroke:#64748b,stroke-width:1px,color:#f8fafc;
    classDef branch fill:#312e81,stroke:#a78bfa,stroke-width:1px,color:#f8fafc;
    class A,B,D,F,G,H,I,J,K,L,M stage;
    class C,E branch;
```

---

## 6. Retrieval, Reranking & Evidence Assessment Sequence

### Understanding Cross-Encoder Scores
In the planned retrieval flow, candidates are generated by dense FAISS search and sparse BM25 search, merged via Reciprocal Rank Fusion (RRF), and then optionally reranked by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Nature of Scores**: Cross-encoder logits represent relative ranking discriminators between candidate passages for a given query. They are **not** calibrated posterior probabilities of factual truth, nor do they guarantee that the passage fully answers the research inquiry.
- **Sufficiency Gate**: The evidence sufficiency gate evaluates whether the top reranked chunks exceed a minimum relevance threshold and contain necessary query terms or entities. If this criteria fails, the system executes an explicit abstention path.

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher
    participant UI as Streamlit UI
    participant Hybrid as Hybrid Retrieval Service
    participant Reranker as Cross-Encoder Reranker
    participant Gate as Evidence Sufficiency Gate
    participant LLM as Ollama (Local LLM)
    participant DB as SQLite (Canonical Store)

    User->>UI: Submit Research Question
    UI->>Hybrid: Query(question, collection_id)
    
    par Dense Semantic Search
        Hybrid->>Hybrid: Encode query with all-MiniLM-L6-v2
        Hybrid->>Hybrid: FAISS top-20 nearest vectors
    and Sparse Lexical Search
        Hybrid->>Hybrid: BM25 score against SQLite chunk tokens
        Hybrid->>Hybrid: BM25 top-20 lexical matches
    end

    Hybrid->>Hybrid: Reciprocal Rank Fusion (RRF) -> Top 15 Candidates
    Hybrid->>Reranker: Rerank(question, top_15_candidates)
    Reranker->>Reranker: Cross-Encoder score query-chunk pairs (Relative Ranks)
    Reranker->>Gate: Top 5 Reranked Passages with Source Pages
    
    alt Insufficient Evidence (Max score < threshold OR key entities missing)
        Gate-->>UI: Calibrated abstention notice ("Insufficient evidence in collection to answer [Question]")
        UI->>DB: Log query with status 'abstained'
    else Evidence Sufficient
        Gate->>LLM: Stream generate with context passages & page citations
        LLM-->>UI: Streamed grounded answer with [Paper, p. X] citations
        UI->>DB: Log query, answer, and cited chunk IDs
    end
```

---

## 7. Storage Architecture & Index Lifecycle Management

### SQLite as the Canonical Store
In IntelliAsk Research V1, **SQLite is the sole canonical persistence layer**. All raw text, page numbers, extracted paper schemas, collection mappings, and query logs live in SQLite.
- **FAISS and BM25 are Derived Indexes**: The vector index and BM25 index are transient derived artifacts that can be fully rebuilt from the SQLite database at any time.

### Stable Identifiers & Vector Mapping
1. **Document Identifier**: A deterministic hash of document content or UUID (`doc_id`).
2. **Chunk Identifier**: `chunk_id = f"{doc_id}_p{page_number}_c{chunk_index}"`.
3. **Vector ID Mapping**: FAISS requires integer IDs. An integer mapping table `vector_mappings (vector_int_id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE)` in SQLite connects each vector in FAISS back to its canonical chunk record.

### Deletion and Re-Indexing Strategy (Avoiding Stale Vectors)
When a paper is deleted or updated in a collection:
- **Strategy A (Rebuild on Collection Modification)**: For typical research collections (5–10 papers, ~500–1,500 chunks), rebuilding the FAISS index and BM25 index directly from SQLite takes under 2 seconds. Rebuilding from canonical SQLite guarantees zero index drift, avoids orphaned vectors, and ensures index consistency.
- **Strategy B (Incremental Removal via IndexIDMap2)**: When using larger collections, `faiss.IndexIDMap2` supports explicit deletion of vector IDs using `index.remove_ids(np.array([vector_id]))`. SQLite deletes the corresponding chunk records and vector mappings in a single transaction.

```mermaid
erDiagram
    COLLECTIONS ||--o{ DOCUMENTS : contains
    DOCUMENTS ||--o{ CHUNKS : splits_into
    DOCUMENTS ||--o{ PAPER_METADATA : has_one
    CHUNKS ||--o{ VECTOR_MAPPINGS : maps_to
    CHUNKS ||--o{ QUERY_CITATIONS : cited_in
    QUERIES ||--o{ QUERY_CITATIONS : references

    COLLECTIONS {
        string id PK
        string name
        string description
        datetime created_at
    }

    DOCUMENTS {
        string id PK
        string collection_id FK
        string file_name
        string file_hash
        int page_count
        datetime indexed_at
    }

    CHUNKS {
        string id PK
        string document_id FK
        int page_number
        int chunk_index
        string text
        int token_count
    }

    VECTOR_MAPPINGS {
        int vector_int_id PK
        string chunk_id FK
    }

    PAPER_METADATA {
        string id PK
        string document_id FK
        string title
        string authors
        string publication_year
        string target_task
        string methodology
        string datasets
        string quantitative_findings
        json raw_json
    }

    QUERIES {
        string id PK
        string collection_id FK
        string question
        string answer
        string status
        float latency_ms
        datetime created_at
    }

    QUERY_CITATIONS {
        string id PK
        string query_id FK
        string chunk_id FK
        float relevance_score
    }
```

---

## 8. Architectural Boundaries & Non-Goals

To maintain high research fidelity and execution discipline, the following boundaries are strictly enforced for V1:

- **Domain-Independent Architecture**: The ingestion, indexing, retrieval, and synthesis engines operate over general scientific PDF literature without hardcoded assumptions about specific research domains.
- **No Remote APIs as Hard Dependencies**: IntelliAsk V1 runs fully on local computational resources (CPU/GPU) with local models via Ollama.
- **No Premature Microservices or FastAPI**: All application logic runs as modular Python packages in the same process as the Streamlit application.
- **No Complex Autonomous Multi-Agent Loops**: All retrieval and comparison pipelines are deterministic, controlled state machines rather than open-ended autonomous agents.
- **Derived Indexes Reconstructible from SQLite**: Vector indices are never treated as the primary store of truth; they are always reconstructible from canonical SQLite tables.
