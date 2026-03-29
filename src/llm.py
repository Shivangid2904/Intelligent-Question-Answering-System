# src/llm.py

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")


def generate_answer(query, context):
    # Limit context again (double safety)
    context = context[:800]

    prompt = f"""
Answer the question based on the context.

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        truncation=True   # 🔥 VERY IMPORTANT
    )

    return result[0]["generated_text"]