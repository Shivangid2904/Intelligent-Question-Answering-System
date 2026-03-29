# src/summarizer.py
# Safe summarizer using text-generation

from transformers import pipeline

# lightweight model
generator = pipeline("text-generation", model="distilgpt2")


def summarize_text(text):
    # limit size (important)
    text = text[:800]

    prompt = f"""
Summarize the following text:

{text}

Summary:
"""

    result = generator(prompt, max_new_tokens=80, do_sample=True)

    return result[0]["generated_text"]