import requests

from src.citation import build_grounded_context


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"


def generate_answer(question: str, passages: list, model: str = DEFAULT_MODEL):
    """
    Use Ollama (local LLM) to generate a synthesized answer from retrieved passages.
    Returns a generator that yields text tokens for streaming.

    Passages may be either Chunk objects (with .text, .source_filename, .page_number)
    or legacy dicts with 'text' and 'score' keys.

    Args:
        question: The user's question.
        passages: List of Chunk objects or dicts with 'text'/'score' keys.
        model:    Ollama model name (default: llama3.2:1b).

    Yields:
        Text tokens as strings for streaming display.
    """
    if not passages:
        yield "I could not find any relevant passages in the document to answer this question."
        return

    # Build grounded context with page-level citations.
    # Accepts both Chunk objects and legacy dict format from app.py.
    try:
        grounded_context = build_grounded_context(
            passages, max_chunks=5, max_chars_per_chunk=800
        )
    except Exception:
        # Graceful fallback to legacy plain-text context
        context_parts = []
        for i, p in enumerate(passages[:3], 1):
            text = p["text"] if isinstance(p, dict) else getattr(p, "text", str(p))
            context_parts.append(f"[Passage {i}]\n{text.strip()[:800]}")
        grounded_context = "\n\n".join(context_parts)

    prompt = f"""You are a research assistant helping analyse scientific literature.
Answer using only the supplied passages. Do not refuse solely because a cited paper's publication date is newer than your training data. If the passages directly support the answer, answer from them. If they do not provide sufficient evidence, state that the information is unavailable.

Rules:
- Start with a clear, direct answer.
- Reference the specific citation(s) that support each claim, e.g. [paper.pdf, p. 4].
- Add 1-2 sentences of explanation using evidence from the context.
- Be informative but concise (3-6 sentences total).
- Do NOT speculate, extrapolate, or assume information beyond the provided text.

Context:
{grounded_context}

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
