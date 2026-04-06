import streamlit as st
import json
import os

from src.document_processor import DocumentProcessor
from src.qa_engine import QAEngine

st.set_page_config(page_title="Intelligent QA System", layout="wide")

st.title("📚 Intelligent Question Answering System")

HISTORY_FILE = "history.json"

# --- AUTO CREATE ---
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# --- INIT ---
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
    except:
        st.session_state.history = []

# 🔥 CACHE ENGINE
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

# --- QUESTION ---
query = st.text_input("Ask a question:")

if query and st.session_state.engine:
    results = st.session_state.engine.search(query, k=10)

    if len(results) == 0:
        st.warning("No results found")
    else:
        # 🔥 FINAL SMART SELECTION
        best = None
        best_score = -1

        query_words = query.lower().split()

        for r in results:
            text = r.text.lower()

            match_count = sum(word in text for word in query_words)

            # 🔥 combine semantic + keyword
            combined_score = r.score + (0.15 * match_count)

            if combined_score > best_score:
                best_score = combined_score
                best = r

        # ✅ CLEAN
        text = best.text.replace("", "").strip()

        # 🔥 Split into sentences
        sentences = text.split(".")

        query_words = query.lower().split()

        relevant_sentences = []

        for sent in sentences:
            sent_lower = sent.lower()

            # keep sentence if it contains keyword
            if any(word in sent_lower for word in query_words):
                relevant_sentences.append(sent.strip())

        # if nothing matched, fallback
        if relevant_sentences:
            clean_answer = ". ".join(relevant_sentences[:2])
        else:
            clean_answer = sentences[0]

        entry = {
            "question": query,
            "answer": clean_answer,
            "source": best.source,
            "score": best.score,
            "passages": [r.text for r in results]
        }

        st.session_state.history.append(entry)

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, indent=2)

# --- CLEAR ---
if st.button("🗑 Clear History"):
    st.session_state.history = []
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# --- CURRENT ANSWER ---
if query and st.session_state.history:
    latest = st.session_state.history[-1]

    st.subheader("🤖 Current Answer")
    st.success(latest["answer"])

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 Score: {round(latest['score'], 3)}")
    with col2:
        st.info(f"📄 Source: {latest['source']}")

    st.subheader("📄 Supporting Passages")

    seen = set()
    for p in latest["passages"]:
        key = p[:100]
        if key in seen:
            continue
        seen.add(key)

        clean_p = p.replace("", "").replace("", "").strip()

        # split into sentences
        sentences = clean_p.split(".")

        # show only meaningful ones
        for s in sentences[:2]:
            if len(s.strip()) > 20:
                st.write("•", s.strip())

    st.markdown("---")

# --- HISTORY ---
if st.session_state.history:
    st.subheader("💬 Previous Questions")

    for item in reversed(st.session_state.history[:-1]):
        st.markdown(f"### ❓ {item['question']}")
        st.success(item["answer"])
        st.markdown("---")