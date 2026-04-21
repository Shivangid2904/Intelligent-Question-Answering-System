import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"


def generate_answer(question: str, passages: list, model: str = DEFAULT_MODEL):
    """
    Use Ollama (local LLM) to generate a synthesized answer from retrieved passages.
    Returns a generator that yields text tokens for streaming.

    Args:
        question: The user's question
        passages:  List of dicts with 'text' and 'score' keys (or plain strings)
        model:     Ollama model name (default: llama3.2:1b)

    Yields:
        Text tokens as strings for streaming display
    """
    if not passages:
        yield "I could not find any relevant passages in the document to answer this question."
        return

    # Build context from top passages (limit to 3 for good coverage)
    context_parts = []
    for i, p in enumerate(passages[:3], 1):
        text = p["text"] if isinstance(p, dict) else p
        context_parts.append(f"[Passage {i}]\n{text.strip()[:800]}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful study assistant. Use the context passages below to answer the question.
- Start with a clear definition or direct answer
- Add 1-2 sentences of explanation or examples from the context
- Be informative but concise (3-5 sentences total)
- Only use information from the context; do not make things up

Context:
{context}

Question: {question}
Answer:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=60
        )

        if response.status_code == 404:
            yield f"❌ Model '{model}' not found. Run: ollama pull {model}"
            return
        elif response.status_code != 200:
            yield f"⚠️ Ollama error {response.status_code}: {response.text}"
            return

        for line in response.iter_lines():
            if line:
                try:
                    chunk = requests.compat.json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except Exception:
                    continue

    except requests.exceptions.ConnectionError:
        yield "🔌 Cannot connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        yield "⏳ Ollama timed out. Please try again."
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"
