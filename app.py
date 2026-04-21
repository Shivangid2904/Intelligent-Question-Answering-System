import streamlit as st
import json
import os

from src.document_processor import DocumentProcessor
from src.qa_engine import QAEngine
from src.llm_engine import generate_answer

st.set_page_config(page_title="Intelligent QA System", layout="wide")

st.title("📚 Intelligent Question Answering System")

HISTORY_FILE = "history.json"

# --- AUTO CREATE HISTORY ---
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# --- SIDEBAR: API KEY ---
# Auto-load from secrets.toml if available
_default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""

with st.sidebar:
    st.header("⚙️ Settings")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=_default_key,
        placeholder="Paste your key here...",
        help="Get a free key at https://aistudio.google.com/app/apikey"
    )
    if gemini_api_key:
        st.success("✅ API key set — LLM answers enabled")
    else:
        st.warning("⚠️ No API key — using extracted answers")

    st.markdown("---")
    st.markdown("**How to get a free key:**")
    st.markdown("1. Go to [aistudio.google.com](https://aistudio.google.com/app/apikey)")
    st.markdown("2. Sign in with Google")
    st.markdown("3. Click **Create API Key**")
    st.markdown("4. Paste it above")

# --- INIT SESSION STATE ---
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
    st.session_state.engine = None
    st.session_state.file_name = None

# --- LOAD HISTORY ---
if "history" not in st.session_state:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            st.session_state.history = json.loads(content) if content else []
    except Exception:
        st.session_state.history = []

# --- CACHE ENGINE ---
@st.cache_resource
def build_engine(_chunks):
    return QAEngine(_chunks)

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if st.session_state.file_name != uploaded_file.name:
        st.info("Processing document...")

        st.session_state.processor = DocumentProcessor()
        st.session_state.processor.process_file(uploaded_file)

        st.session_state.engine = build_engine(st.session_state.processor.chunks)
        st.session_state.file_name = uploaded_file.name

        st.success("✅ Document processed!")

# --- QUESTION INPUT ---
query = st.text_input("Ask a question:")

if query and st.session_state.engine:
    results = st.session_state.engine.search(query, k=10)

    if len(results) == 0:
        st.warning("No results found")
    else:
        query_words = query.lower().split()

        # --- BEST CHUNK SELECTION (semantic + keyword hybrid) ---
        best = None
        best_score = -1
        for r in results:
            match_count = sum(word in r.text.lower() for word in query_words)
            combined_score = r.score + (0.15 * match_count)
            if combined_score > best_score:
                best_score = combined_score
                best = r

        # --- FILTER PASSAGES ---
        SCORE_THRESHOLD = 0.40
        filtered_passages = []
        for r in results:
            if r.score < SCORE_THRESHOLD:
                continue
            if not any(word in r.text.lower() for word in query_words):
                continue
            filtered_passages.append({"text": r.text, "score": round(float(r.score), 3)})

        # Sort best first
        filtered_passages.sort(key=lambda x: x["score"], reverse=True)

        # --- GENERATE ANSWER ---
        if gemini_api_key and filtered_passages:
            with st.spinner("🤖 Generating answer with Gemini..."):
                llm_answer = generate_answer(gemini_api_key, query, filtered_passages)
            answer_source = "gemini"
        else:
            # Fallback: extract sentences with keyword match
            text = best.text.strip()
            sentences = text.split(".")
            relevant = [s.strip() for s in sentences if any(w in s.lower() for w in query_words)]
            llm_answer = ". ".join(relevant[:2]) if relevant else sentences[0]
            answer_source = "extracted"

        # --- SAVE TO HISTORY ---
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

# --- CLEAR HISTORY ---
if st.button("🗑 Clear History"):
    st.session_state.history = []
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# --- DISPLAY CURRENT ANSWER ---
if query and st.session_state.history:
    latest = st.session_state.history[-1]

    # Only show if it matches the current question
    if latest.get("question") == query:
        is_llm = latest.get("answer_source") == "gemini"
        label = "🤖 AI Answer (Gemini)" if is_llm else "🤖 Current Answer (Extracted)"
        st.subheader(label)
        st.success(latest["answer"])

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 Score: {round(latest['score'], 3)}")
        with col2:
            st.info(f"📄 Source: {latest['source']}")

        # --- SUPPORTING PASSAGES ---
        st.subheader("📄 Supporting Passages")

        seen = set()
        shown = 0
        for item in latest["passages"]:
            if shown >= 4:
                break

            if isinstance(item, dict):
                p_text = item.get("text", "")
                p_score = item.get("score", None)
            else:
                p_text = item
                p_score = None

            key = p_text[:100]
            if key in seen:
                continue
            seen.add(key)

            clean_p = p_text.strip()
            if len(clean_p) < 30:
                continue

            score_badge = (
                f"<span style='float:right;font-size:0.75rem;color:#a78bfa;'>relevance: {p_score}</span>"
                if p_score is not None else ""
            )

            st.markdown(
                f"<div style='background:#1e1e2e;border-left:4px solid #7c3aed;padding:12px 16px;"
                f"border-radius:6px;margin-bottom:12px;line-height:1.7;font-size:0.95rem;'>"
                f"{score_badge}{clean_p}"
                f"</div>",
                unsafe_allow_html=True
            )
            shown += 1

        if shown == 0:
            st.info("No highly relevant passages found for this query.")

        st.markdown("---")

# --- HISTORY ---
if st.session_state.history:
    st.subheader("💬 Previous Questions")

    # Skip the last entry if it matches the current query (already shown above)
    history_to_show = st.session_state.history[:-1] if (
        query and st.session_state.history and
        st.session_state.history[-1].get("question") == query
    ) else st.session_state.history

    for item in reversed(history_to_show):
        badge = "🤖" if item.get("answer_source") == "gemini" else "📝"
        st.markdown(f"### ❓ {item['question']}")
        st.success(f"{badge} {item['answer']}")
        st.markdown("---")