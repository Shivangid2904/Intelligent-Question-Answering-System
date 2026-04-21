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

# --- SIDEBAR: OLLAMA SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**🦙 LLM: Ollama (Local)**")
    ollama_model = st.selectbox(
        "Model",
        ["llama3.2", "llama3", "mistral", "phi3", "gemma2"],
        index=0,
        help="Make sure the model is pulled: ollama pull <model>"
    )
    st.success("✅ Running locally — no API key needed!")
    st.markdown("---")
    st.markdown("**Make sure Ollama is running:**")
    st.code(f"ollama serve", language="bash")
    st.markdown(f"**To pull a model:**")
    st.code(f"ollama pull {ollama_model}", language="bash")

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
        if filtered_passages:
            st.subheader(f"🦙 AI Answer (Ollama)")
            with st.spinner(f"Thinking with {ollama_model}..."):
                # Stream tokens and collect full answer
                stream_gen = generate_answer(query, filtered_passages, model=ollama_model)
                llm_answer = st.write_stream(stream_gen)
            answer_source = "ollama"
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
        is_llm = latest.get("answer_source") == "ollama"
        if not is_llm:
            # For extracted answers, show the subheader + answer text
            st.subheader("🤖 Current Answer (Extracted)")
            st.success(latest["answer"])
        # For ollama: answer was already streamed above — skip duplicate display

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