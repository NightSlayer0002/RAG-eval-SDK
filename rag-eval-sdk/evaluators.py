"""
evaluators.py — Core evaluation metrics for RAG responses.

Computes two standard scores using cosine similarity on semantic embeddings:
  1. Relevance Score: how aligned the response is with the full context.
  2. Hallucination Score: how much the response deviates from any single chunk.
"""

from sentence_transformers import util
from .embeddings import embed_text


def score_relevance_completeness(response: str, context_chunks: list) -> float:
    """
    Measures how semantically aligned the response is with the combined context.
    Returns cosine similarity (can be slightly negative for unrelated texts).
    """
    texts = [
        chunk["text"]
        for chunk in context_chunks
        if isinstance(chunk, dict) and "text" in chunk and chunk["text"]
    ]
    combined_context = " ".join(texts)
    if not combined_context.strip():
        return 0.0

    response_embedding = embed_text(response)
    context_embedding = embed_text(combined_context)

    similarity = util.cos_sim(response_embedding, context_embedding).item()
    return round(similarity, 3)


def score_hallucination(response: str, context_chunks: list) -> float:
    """
    Measures how much the response is NOT supported by any context chunk.
    hallucination = 1 - max_similarity_to_any_chunk.

    Score near 0.0 = well grounded.
    Score near 1.0 = likely hallucinated.
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

    if not similarities:
        return 0.0

    max_similarity = max(similarities)
    return round(1 - max_similarity, 3)
