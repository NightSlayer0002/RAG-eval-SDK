"""
evaluators.py
-------------
Core evaluation metrics for LLM responses in a RAG pipeline.

This module computes two standard scores:
  1. Relevance/Completeness Score: How well does the response align with
     the full retrieved context? (higher = more relevant)
  2. Hallucination Score: How much does the response deviate from any
     single context chunk? (higher = more hallucinated)

Both use cosine similarity on semantic embeddings — meaning we compare
the "meaning" of texts, not just keyword overlap.

The embedding model is imported from embeddings.py (loaded once globally
to avoid redundant memory usage across modules).
"""

from sentence_transformers import util
from embeddings import embed_text


def score_relevance_completeness(response: str, context_chunks: list) -> float:
    """
    Measures how semantically aligned the response is with the
    combined retrieved context as a whole.

    Method:
      - Combine all chunk texts into a single string.
      - Embed both the response and the combined context.
      - Return cosine similarity (0 to 1).

    A score of 1.0 means the response perfectly mirrors the context.
    A score near 0.0 means the response is about something completely different.
    """
    texts = [
        chunk["text"]
        for chunk in context_chunks
        if isinstance(chunk, dict) and "text" in chunk and chunk["text"]
    ]
    combined_context = " ".join(texts)

    response_embedding = embed_text(response)
    context_embedding = embed_text(combined_context)

    similarity = util.cos_sim(response_embedding, context_embedding).item()
    return round(similarity, 3)


def score_hallucination(response: str, context_chunks: list) -> float:
    """
    Measures how much the response is NOT supported by any context chunk.

    Method:
      - Compare the response embedding to each chunk embedding individually.
      - Take the MAX similarity (best-case support from any single chunk).
      - Hallucination = 1 - max_similarity.

    So:
      - Score near 0.0 = response closely matches at least one chunk = not hallucinated
      - Score near 1.0 = response doesn't match any chunk = likely hallucinated

    BUG FIX: original code had dead code after the return statement which
    would crash on empty context. Guard is now placed correctly before max().
    """
    response_embedding = embed_text(response)

    similarities = []
    for chunk in context_chunks:
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text", "")
        if not text:
            continue

        chunk_embedding = embed_text(text)
        sim = util.cos_sim(response_embedding, chunk_embedding).item()
        similarities.append(sim)

    # FIXED: guard before max() — avoids crash on empty list
    if not similarities:
        return 0.0

    max_similarity = max(similarities)
    hallucination_score = 1 - max_similarity
    return round(hallucination_score, 3)
