"""
position_bias_detector.py
-------------------------
NOVEL FEATURE #4: "Lost in the Middle" Position Bias Detector

Based on: "Lost in the Middle: How Language Models Use Long Contexts"
(Liu et al., Stanford, 2023).

LLMs pay more attention to information at the BEGINNING and END of their
context window, and tend to ignore the MIDDLE -- even if that middle
information is highly relevant.

This module detects whether chunks were ignored due to position, not quality.
If detected, the fix is trivial: reorder chunks. No retraining needed.

Detection thresholds (tuned to avoid false positives):
  U-shape gap > 0.08 and middle phantom ratio > 0.0
  Primacy/Recency gap > 0.12
  _MARGIN = 0.08 (middle must be this far below each endpoint)
"""

from sentence_transformers import util
from .embeddings import embed_text


def detect_position_bias(
    response: str,
    context_chunks: list,
    retriever_scores: list = None
) -> dict:
    """
    Detects whether the LLM ignored chunks due to their position.

    Args:
        response:         The LLM's generated answer.
        context_chunks:   List of chunk dicts.
        retriever_scores: Optional list of retriever confidence scores.

    Returns:
        position_analysis, zone_averages, bias_detected, bias_type,
        bias_confidence, middle_phantom_ratio, wasted_good_chunks, bias_summary
    """
    response_embedding = embed_text(response)

    position_data = []
    total_chunks = 0

    for position, chunk in enumerate(context_chunks):
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text", "")
        if not text.strip():
            continue

        chunk_emb = embed_text(text)
        sim = util.cos_sim(response_embedding, chunk_emb).item()

        retriever_score = None
        if retriever_scores and position < len(retriever_scores):
            retriever_score = retriever_scores[position]

        position_data.append({
            "chunk_id": chunk.get("id", -1),
            "position": position + 1,
            "similarity_to_response": round(sim, 3),
            "retriever_score": round(retriever_score, 3) if retriever_score else None,
            "is_phantom": sim < 0.30,
            "preview": text[:80].replace("\n", " ").strip() + "..."
        })
        total_chunks += 1

    if total_chunks < 3:
        return {
            "position_analysis": position_data,
            "bias_detected": False,
            "bias_type": "NONE",
            "bias_confidence": "LOW",
            "middle_phantom_ratio": 0.0,
            "wasted_good_chunks": 0,
            "bias_summary": "Too few chunks to assess position bias (need >= 3).",
            "recommended_action": "No action needed."
        }

    third = max(1, total_chunks // 3)
    beginning = position_data[:third]
    middle = position_data[third:total_chunks - third]
    end = position_data[total_chunks - third:]

    if not middle:
        middle = position_data[third:third + 1] if third < total_chunks else []

    def avg_sim(zone):
        if not zone:
            return 0.0
        return round(sum(p["similarity_to_response"] for p in zone) / len(zone), 3)

    avg_beginning = avg_sim(beginning)
    avg_middle = avg_sim(middle)
    avg_end = avg_sim(end)

    middle_phantoms = sum(1 for p in middle if p["is_phantom"])
    middle_phantom_ratio = round(
        middle_phantoms / len(middle), 3
    ) if middle else 0.0

    # "Smoking gun": high-retriever-score chunks in the middle that were ignored
    wasted_good_chunks = [
        p for p in middle
        if p["is_phantom"] and p["retriever_score"] and p["retriever_score"] > 0.4
    ]

    u_shape_gap = ((avg_beginning + avg_end) / 2) - avg_middle
    primacy_gap = avg_beginning - avg_end
    recency_gap = avg_end - avg_beginning

    # Geometric bias detection: middle must be lower than BOTH endpoints.
    # This works correctly with dense embeddings (MiniLM) where even
    # "ignored" chunks have moderate absolute similarity (~0.40) to the
    # response because they're all domain-adjacent.
    _MARGIN = 0.08  # middle must be at least this much lower than each endpoint
    middle_lower_than_both = (
        avg_middle < avg_beginning - _MARGIN
        and avg_middle < avg_end - _MARGIN
    )
    middle_lower_than_start = avg_middle < avg_beginning - _MARGIN
    middle_lower_than_end = avg_middle < avg_end - _MARGIN

    bias_detected = False
    bias_type = "NONE"
    bias_confidence = "LOW"

    if u_shape_gap > 0.08 and middle_lower_than_both and middle_phantom_ratio > 0.0:
        bias_detected = True
        bias_type = "U_SHAPED"
        if u_shape_gap > 0.15 or len(wasted_good_chunks) >= 2:
            bias_confidence = "HIGH"
        elif u_shape_gap > 0.10 or len(wasted_good_chunks) >= 1:
            bias_confidence = "MEDIUM"
        else:
            bias_confidence = "LOW"
    elif primacy_gap > 0.12 and middle_lower_than_start and middle_phantom_ratio > 0.0:
        bias_detected = True
        bias_type = "PRIMACY"
        bias_confidence = "MEDIUM" if primacy_gap > 0.18 else "LOW"
    elif recency_gap > 0.12 and middle_lower_than_end and middle_phantom_ratio > 0.0:
        bias_detected = True
        bias_type = "RECENCY"
        bias_confidence = "MEDIUM" if recency_gap > 0.18 else "LOW"

    if len(wasted_good_chunks) >= 2 and bias_detected:
        bias_confidence = "HIGH"

    if not bias_detected:
        summary = (
            f"No significant position bias detected. "
            f"Zone similarities -- Beginning: {avg_beginning}, "
            f"Middle: {avg_middle}, End: {avg_end}. "
            f"Middle phantom ratio: {middle_phantom_ratio}."
        )
    else:
        summary = (
            f"Position bias DETECTED ({bias_type}, {bias_confidence} confidence). "
            f"The LLM appears to ignore chunks placed in the middle of the context. "
            f"Zone similarities -- Beginning: {avg_beginning}, "
            f"Middle: {avg_middle}, End: {avg_end}. "
            f"Middle phantom ratio: {middle_phantom_ratio} "
            f"({middle_phantoms}/{len(middle) if middle else 0} middle chunks ignored)."
        )
        if wasted_good_chunks:
            ids = [str(w["chunk_id"]) for w in wasted_good_chunks]
            summary += (
                f" SMOKING GUN: Chunk(s) {', '.join(ids)} had high retriever "
                f"scores but were ignored -- likely due to position, not relevance."
            )

    if bias_detected:
        recommendation = (
            "Reorder chunks in the prompt: place the most important chunks at "
            "the BEGINNING and END of the context section, not in the middle. "
            "This fix requires no model retraining -- just reordering."
        )
    else:
        recommendation = "No action needed -- chunk positions are not causing issues."

    return {
        "position_analysis": position_data,
        "zone_averages": {
            "beginning": avg_beginning,
            "middle": avg_middle,
            "end": avg_end
        },
        "bias_detected": bias_detected,
        "bias_type": bias_type,
        "bias_confidence": bias_confidence,
        "middle_phantom_ratio": middle_phantom_ratio,
        "wasted_good_chunks": len(wasted_good_chunks),
        "bias_summary": summary,
        "recommended_action": recommendation
    }
