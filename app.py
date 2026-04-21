import streamlit as st
import json
import os

from src.document_processor import DocumentProcessor, get_nlp
from src.qa_engine import QAEngine, get_embedding_model
from src.llm_engine import generate_answer

st.set_page_config(
    page_title="IntelliAsk — AI Document QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background: #0f0f1a; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(124,58,237,0.3);
    box-shadow: 0 8px 32px rgba(124,58,237,0.15);
}
.app-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
}
.app-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Upload Zone ── */
.upload-section {
    background: linear-gradient(135deg, #1e1b4b 0%, #1a1a2e 100%);
    border: 2px dashed rgba(124,58,237,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s ease;
}

/* ── Question Input ── */
.stTextInput > div > div > input {
    background: #1e1e2e !important;
    border: 2px solid rgba(124,58,237,0.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.3s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}

/* ── Answer Card ── */
.answer-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin: 1rem 0 1.5rem 0;
    line-height: 1.8;
    font-size: 1.05rem;
    color: #e2e8f0;
    box-shadow: 0 4px 20px rgba(124,58,237,0.1);
}
.answer-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #a78bfa;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* ── Score/Source Pills ── */
.meta-pill {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    color: #c4b5fd;
    margin-right: 0.5rem;
    margin-bottom: 1.5rem;
}

/* ── Passage Card ── */
.passage-card {
    background: #13131f;
    border-left: 4px solid #7c3aed;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.9rem;
    line-height: 1.75;
    font-size: 0.92rem;
    color: #cbd5e1;
    position: relative;
}
.passage-score {
    float: right;
    font-size: 0.72rem;
    color: #7c3aed;
    background: rgba(124,58,237,0.1);
    border-radius: 10px;
    padding: 0.15rem 0.5rem;
    font-weight: 600;
}

/* ── Section Headers ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #64748b;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── History Card ── */
.history-card {
    background: #13131f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.history-question {
    font-size: 0.88rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 0.4rem;
}
.history-answer {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(124,58,237,0.2);
}
[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    color: #fca5a5 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(239,68,68,0.2) !important;
    border-color: rgba(239,68,68,0.5) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #13131f;
    border-radius: 10px;
    border: 1px solid rgba(124,58,237,0.2);
}

/* ── Hide default streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
HISTORY_FILE = "history.json"

# ── AUTO CREATE HISTORY ───────────────────────────────────────────────────────
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 IntelliAsk")
    st.markdown("<p style='color:#64748b;font-size:0.82rem;margin-top:-0.5rem;'>Local RAG • Powered by Ollama</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p style='color:#94a3b8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;'>🦙 Model</p>", unsafe_allow_html=True)
    ollama_model = st.selectbox(
        "model_select",
        ["llama3.2", "llama3.2:1b", "llama3", "mistral", "phi3", "gemma2"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);
    border-radius:8px;padding:0.6rem 0.8rem;margin:0.8rem 0;'>
        <span style='color:#34d399;font-size:0.82rem;font-weight:600;'>✓ Running locally</span><br>
        <span style='color:#64748b;font-size:0.75rem;'>No API key • No data sent anywhere</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#64748b;font-size:0.78rem;'>Pull a model:</p>", unsafe_allow_html=True)
    st.code(f"ollama pull {ollama_model}", language="bash")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
    st.session_state.engine = None
    st.session_state.file_name = None

if "history" not in st.session_state:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            st.session_state.history = json.loads(content) if content else []
    except Exception:
        st.session_state.history = []

# ── CACHED MODEL LOADERS ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔤 Loading spaCy language model...")
def load_nlp():
    return get_nlp()

@st.cache_resource(show_spinner="🧠 Loading embedding model (first time only)...")
def load_embedding_model():
    return get_embedding_model()

@st.cache_resource(show_spinner="⚙️ Building search index...")
def build_engine(_chunks, _model):
    return QAEngine(_chunks, _model)

nlp = load_nlp()
embedding_model = load_embedding_model()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">🧠 IntelliAsk</div>
    <p class="app-subtitle">Ask questions about any PDF — answered by a local AI, entirely on your machine.</p>
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>📄 Document</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a PDF to get started", type="pdf", label_visibility="collapsed")

if uploaded_file:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("📖 Reading and indexing document..."):
            st.session_state.processor = DocumentProcessor()
            st.session_state.processor.process_file(uploaded_file, nlp)
            st.session_state.engine = build_engine(
                tuple(st.session_state.processor.chunks),
                embedding_model
            )
            st.session_state.file_name = uploaded_file.name

        st.markdown(f"""
        <div style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);
        border-radius:8px;padding:0.6rem 1rem;margin:0.5rem 0 1rem 0;font-size:0.88rem;color:#34d399;'>
            ✅ <strong>{uploaded_file.name}</strong> indexed successfully
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.2);
        border-radius:8px;padding:0.5rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#93c5fd;'>
            📄 <strong>{uploaded_file.name}</strong> — ready
        </div>""", unsafe_allow_html=True)

# ── QUESTION INPUT ─────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>💬 Ask a Question</p>", unsafe_allow_html=True)
query = st.text_input("question_input", placeholder="e.g. What is perplexity? How do N-grams work?", label_visibility="collapsed")

# ── SEARCH + GENERATE ─────────────────────────────────────────────────────────
if query and st.session_state.engine:
    results = st.session_state.engine.search(query, k=10)

    if len(results) == 0:
        st.markdown("<div style='color:#f87171;padding:0.75rem;'>⚠️ No results found in the document.</div>", unsafe_allow_html=True)
    else:
        import re
        # Strip punctuation from query words to avoid "smoothing?" not matching "smoothing"
        query_words = [re.sub(r'[^\w]', '', w) for w in query.lower().split() if re.sub(r'[^\w]', '', w)]
        # Use only meaningful content words (len > 3) for keyword filtering
        content_words = [w for w in query_words if len(w) > 3]
        if not content_words:
            content_words = query_words  # fallback if all words are short

        best = None
        best_score = -1
        for r in results:
            match_count = sum(word in r.text.lower() for word in content_words)
            combined_score = r.score + (0.15 * match_count)
            if combined_score > best_score:
                best_score = combined_score
                best = r

        SCORE_THRESHOLD = 0.38
        filtered_passages = []
        for r in results:
            if r.score < SCORE_THRESHOLD:
                continue
            if not any(word in r.text.lower() for word in content_words):
                continue
            filtered_passages.append({"text": r.text, "score": round(float(r.score), 3)})
        filtered_passages.sort(key=lambda x: x["score"], reverse=True)

        # ── AI ANSWER ────────────────────────────────────────────────────────
        if filtered_passages:
            passages_to_use = filtered_passages
            use_ollama = True
        else:
            # Fallback: search ALL results (ignore score threshold) for keyword match
            keyword_fallback = [
                {"text": r.text, "score": round(float(r.score), 3)}
                for r in results
                if any(word in r.text.lower() for word in content_words)
            ]
            keyword_fallback.sort(key=lambda x: x["score"], reverse=True)

            if keyword_fallback:
                passages_to_use = keyword_fallback[:3]
                use_ollama = True
            else:
                passages_to_use = []
                use_ollama = False

        if use_ollama:
            st.markdown("<p class='section-header'>🦙 AI Answer</p>", unsafe_allow_html=True)
            st.markdown("<div class='answer-card'><div class='answer-label'>Ollama · " + ollama_model + "</div>", unsafe_allow_html=True)
            with st.spinner(f"Thinking with {ollama_model}..."):
                stream_gen = generate_answer(query, passages_to_use, model=ollama_model)
                llm_answer = st.write_stream(stream_gen)
            st.markdown("</div>", unsafe_allow_html=True)
            answer_source = "ollama"
            # Use best keyword-matching chunk for meta info
            best_passage = passages_to_use[0] if passages_to_use else None
        else:
            llm_answer = "This topic doesn't appear to be covered in the uploaded document."
            answer_source = "not_found"
            st.markdown(
                f"<div class='answer-card'><div class='answer-label'>Not Found</div>"
                f"⚠️ {llm_answer}</div>",
                unsafe_allow_html=True
            )

        # ── META PILLS ────────────────────────────────────────────────────────
        st.markdown(
            f"<span class='meta-pill'>📊 Score: {round(best.score, 3)}</span>"
            f"<span class='meta-pill'>📄 {best.source}</span>",
            unsafe_allow_html=True
        )

        # ── SUPPORTING PASSAGES ───────────────────────────────────────────────
        if filtered_passages:
            st.markdown("<p class='section-header'>📚 Supporting Passages</p>", unsafe_allow_html=True)
            seen = set()
            shown = 0
            for item in filtered_passages:
                if shown >= 4:
                    break
                p_text = item.get("text", "") if isinstance(item, dict) else item
                p_score = item.get("score", None) if isinstance(item, dict) else None
                key = p_text[:100]
                if key in seen:
                    continue
                seen.add(key)
                clean_p = p_text.strip()
                if len(clean_p) < 30:
                    continue
                badge = f"<span class='passage-score'>relevance {p_score}</span>" if p_score else ""
                st.markdown(f"<div class='passage-card'>{badge}{clean_p}</div>", unsafe_allow_html=True)
                shown += 1
            if shown == 0:
                st.markdown("<p style='color:#64748b;font-size:0.88rem;'>No highly relevant passages found.</p>", unsafe_allow_html=True)

        # ── SAVE TO HISTORY ───────────────────────────────────────────────────
        entry = {
            "question": query,
            "answer": llm_answer,
            "answer_source": answer_source,
            "source": best.source,
            "score": best.score,
            "passages": filtered_passages
        }
        st.session_state.history.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, indent=2)

elif query and not st.session_state.engine:
    st.markdown("<div style='color:#fbbf24;padding:0.75rem;font-size:0.9rem;'>⚠️ Please upload a PDF first.</div>", unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────
history_to_show = st.session_state.history[:-1] if (
    query and st.session_state.history and
    st.session_state.history[-1].get("question") == query
) else st.session_state.history

if history_to_show:
    st.markdown("<p class='section-header'>🕘 History</p>", unsafe_allow_html=True)

    col_h, col_btn = st.columns([6, 1])
    with col_btn:
        if st.button("🗑 Clear"):
            st.session_state.history = []
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            st.rerun()

    for item in reversed(history_to_show):
        q = item.get("question", "")
        a = item.get("answer", "")
        short_a = a[:180] + "..." if len(a) > 180 else a
        st.markdown(f"""
        <div class='history-card'>
            <div class='history-question'>❓ {q}</div>
            <div class='history-answer'>{short_a}</div>
        </div>""", unsafe_allow_html=True)