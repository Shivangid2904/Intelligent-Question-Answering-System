import streamlit as st
import os

from src.loader import load_pdf
from src.preprocess import clean_text, chunk_text
from src.embedder import get_embeddings
from src.vector_store import create_faiss_index, save_index, load_index
from src.retriever import retrieve
from src.llm import generate_answer
from src.summarizer import summarize_text

st.set_page_config(page_title="QA System", layout="wide")

st.title("📚 Intelligent Question Answering System")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    os.makedirs("data", exist_ok=True)

    file_path = "data/temp.pdf"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing document...")

    # Load and clean text
    text = load_pdf(file_path)

    # ❗ Check if extraction failed (PPT / scanned PDF)
    if not text or len(text.strip()) < 50:
        st.error("❌ Could not extract meaningful text from this PDF. Try a proper text-based PDF.")
        st.stop()

    text = clean_text(text)

    # Chunking
    chunks = chunk_text(text)

    # Debug info
    st.write("📊 Text length:", len(text))
    st.write("📊 Number of chunks:", len(chunks))

    if len(chunks) == 0:
        st.error("❌ No chunks created. Cannot proceed.")
        st.stop()

    # Embeddings
    embeddings = get_embeddings(chunks)

    # FAISS index
    index = create_faiss_index(embeddings)

    # Save
    save_index(index, chunks)

    st.success("✅ Document processed successfully!")

# --- Question Input ---
query = st.text_input("Ask a question:")

if query:
    try:
        index, chunks = load_index()

        results = retrieve(query, index, chunks)

        # ❗ Safety check
        if len(results) == 0:
            st.warning("⚠️ No relevant content found. Try another document or question.")
            st.stop()

        # Limit context length (VERY IMPORTANT FIX)
        context = " ".join(results[:2])
        context = context[:800]   # 🔥 truncate to safe size

        answer = generate_answer(query, context)

        st.subheader("🤖 Answer")
        st.write(answer)

        st.subheader("📄 Sources")
        for r in results:
            st.write("-", r[:200], "...")

        if st.button("Summarize"):
            summary = summarize_text(context)

            st.subheader("📝 Summary")
            st.write(summary)

    except Exception as e:
        st.error(f"Error: {e}")