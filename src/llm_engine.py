import google.generativeai as genai


def generate_answer(api_key: str, question: str, passages: list[dict]) -> str:
    """
    Use Gemini to generate a synthesized answer from retrieved passages.

    Args:
        api_key:  Google Gemini API key
        question: The user's question
        passages: List of dicts with 'text' and 'score' keys

    Returns:
        A clean, synthesized answer string
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    if not passages:
        return "I could not find any relevant passages in the document to answer this question."

    # Build context from top passages
    context_parts = []
    for i, p in enumerate(passages[:4], 1):
        text = p["text"] if isinstance(p, dict) else p
        context_parts.append(f"[Passage {i}]\n{text.strip()}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are an intelligent question-answering assistant. 
Answer the question below using ONLY the provided context passages from the document.
Be concise, accurate, and directly answer the question in 2-4 sentences.
Do NOT make up information. If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}"
