import streamlit as st
from src.document_processor import DocumentProcessor
from src.qa_engine import QAEngine

st.title("📚 Intelligent Question Answering System")

# Initialize
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
    st.session_state.engine = None

# Upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    st.info("Processing document...")

    st.session_state.processor.process_file(uploaded_file)

    st.session_state.engine = QAEngine(st.session_state.processor.chunks)

    st.success("✅ Document processed!")

# Query
query = st.text_input("Ask a question:")

if query and st.session_state.engine:
    results = st.session_state.engine.search(query)

    if len(results) == 0:
        st.warning("No results found")
    else:
        st.subheader("🤖 Answer")

        # ✅ BEST PASSAGE
        st.write(results[0].text)

        st.subheader("📄 Supporting Passages")

        for r in results:
            st.write("-", r.text[:200], "...")