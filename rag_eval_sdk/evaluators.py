"""
evaluators.py — Core evaluation metrics for RAG responses (v2).

Computes evaluation scores using cosine similarity on semantic embeddings:
  1. Relevance Score: how aligned the response is with the full context.
  2. Hallucination Score: how much the response deviates from any single chunk.
  3. Sentence-Level Hallucination (NEW): per-sentence grounding analysis.
  4. Adaptive Threshold (NEW): auto-calibrates detection thresholds per dataset.
"""

import re
import numpy as np
from .embeddings import cosine_similarity, embed_text


def _split_into_sentences(text: str) -> list:
    """
    Splits text into sentences using punctuation-based splitting.
    Filters out very short fragments (< 10 chars) that aren't real sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


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

    similarity = cosine_similarity(response_embedding, context_embedding)
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
        sim = cosine_similarity(response_embedding, chunk_embedding)
        similarities.append(sim)

    if not similarities:
        # Absence of evidence must never look like a perfectly grounded answer.
        return 1.0

    max_similarity = max(similarities)
    return round(1 - max_similarity, 3)


def score_hallucination_sentence_level(response: str, context_chunks: list) -> dict:
    """
    IMPROVED: Sentence-level hallucination scoring.

    Instead of comparing the ENTIRE response against each chunk (coarse),
    this decomposes the response into individual sentences, scores each
    sentence against every chunk, and computes the fraction of sentences
    that lack support from any chunk.

    Why this is better:
      - A 5-sentence response where 4 are grounded and 1 is hallucinated
        gets a high score with whole-response scoring (false negative).
      - Sentence-level catches that 1 unsupported sentence.

    Returns:
        Dict with:
          - sentence_scores: per-sentence breakdown
          - unsupported_fraction: 0.0 (all grounded) to 1.0 (all hallucinated)
          - hallucination_score: same scale as old score for backward compat
          - avg_max_similarity: average of each sentence's best chunk match
    """
    sentences = _split_into_sentences(response)

    if not sentences:
        return {
            "sentence_scores": [],
            "unsupported_fraction": 0.0,
            "hallucination_score": score_hallucination(response, context_chunks),
            "avg_max_similarity": 0.0,
            "sentence_count": 0
        }

    # Get all chunk embeddings once
    chunk_embeddings = []
    for chunk in context_chunks:
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text", "")
        if not text:
            continue
        chunk_embeddings.append(embed_text(text))

    if not chunk_embeddings:
        return {
            "sentence_scores": [],
            "unsupported_fraction": 1.0,
            "hallucination_score": 1.0,
            "avg_max_similarity": 0.0,
            "sentence_count": len(sentences)
        }

    sentence_scores = []
    supported_threshold = 0.35  # a sentence is "supported" if max_sim >= this

    for sent in sentences:
        sent_emb = embed_text(sent)
        max_sim = max(
            cosine_similarity(sent_emb, chunk_emb)
            for chunk_emb in chunk_embeddings
        )
        is_supported = max_sim >= supported_threshold
        sentence_scores.append({
            "sentence": sent[:100] + ("..." if len(sent) > 100 else ""),
            "max_chunk_similarity": round(max_sim, 3),
            "is_supported": is_supported
        })

    unsupported_count = sum(1 for s in sentence_scores if not s["is_supported"])
    unsupported_fraction = round(unsupported_count / len(sentence_scores), 3)
    avg_max_sim = round(
        sum(s["max_chunk_similarity"] for s in sentence_scores) / len(sentence_scores), 3
    )

    return {
        "sentence_scores": sentence_scores,
        "unsupported_fraction": unsupported_fraction,
        "hallucination_score": unsupported_fraction,  # backward-compatible field
        "avg_max_similarity": avg_max_sim,
        "sentence_count": len(sentence_scores),
        "supported_count": len(sentence_scores) - unsupported_count,
        "unsupported_count": unsupported_count
    }


def compute_adaptive_threshold(scores: list, method: str = "otsu") -> dict:
    """
    NOVEL: Adaptive threshold computation for hallucination detection.

    Instead of hardcoding a threshold (e.g., 0.40), this analyzes the
    distribution of hallucination scores across a batch and finds the
    optimal split point automatically.

    Two methods available:
      1. "otsu" — Otsu's method: finds the threshold that maximally separates
         two populations (hallucinated vs. faithful). Same principle used in
         image binarization, applied here to score distributions.
      2. "percentile" — Flags the top X% of scores as hallucinated.

    Args:
        scores: list of hallucination scores (0-1) from a batch evaluation.
        method: "otsu" or "percentile"

    Returns:
        Dict with:
          - threshold: the computed optimal threshold
          - method: which method was used
          - distribution stats: mean, std, median, etc.
    """
    if not scores or len(scores) < 3:
        return {
            "threshold": 0.40,
            "method": "default",
            "reason": "Too few scores for adaptive computation (need >= 3).",
            "stats": {}
        }

    arr = np.array(scores)

    if method == "otsu":
        # Otsu's method: find threshold that maximizes between-class variance
        best_threshold = 0.40
        best_variance = 0.0

        # Test 100 candidate thresholds between min and max scores
        candidates = np.linspace(arr.min(), arr.max(), 100)

        for t in candidates:
            class_low = arr[arr <= t]
            class_high = arr[arr > t]

            if len(class_low) == 0 or len(class_high) == 0:
                continue

            w_low = len(class_low) / len(arr)
            w_high = len(class_high) / len(arr)

            # Between-class variance
            variance = w_low * w_high * (class_low.mean() - class_high.mean()) ** 2

            if variance > best_variance:
                best_variance = variance
                best_threshold = t

        threshold = round(float(best_threshold), 3)
        reason = (
            f"Otsu's method found optimal split at {threshold} "
            f"(between-class variance: {round(best_variance, 4)})"
        )

    elif method == "percentile":
        # Flag top 30% as hallucinated
        threshold = round(float(np.percentile(arr, 70)), 3)
        reason = f"70th percentile threshold: scores above {threshold} flagged."

    else:
        threshold = 0.40
        reason = "Unknown method, using default."

    return {
        "threshold": threshold,
        "method": method,
        "reason": reason,
        "stats": {
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
            "median": round(float(np.median(arr)), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
            "count": len(scores)
        }
    }
