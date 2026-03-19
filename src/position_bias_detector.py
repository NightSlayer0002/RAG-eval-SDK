"""
position_bias_detector.py
-------------------------
NOVEL FEATURE #4: "Lost in the Middle" Position Bias Detector

Based on the landmark paper: "Lost in the Middle: How Language Models Use Long
Contexts" (Liu et al., Stanford, 2023). The paper proved that LLMs pay more
attention to information at the BEGINNING and END of their context window,
and tend to IGNORE information placed in the MIDDLE — even if that middle
information is highly relevant.

The Problem:
  Imagine your retriever fetches 10 excellent chunks. Chunks at positions
  1-2 and 9-10 get used by the LLM, but chunks at positions 4-6 (the middle)
  get silently ignored — NOT because they were irrelevant, but because of
  WHERE they appeared in the prompt.

  Standard evaluation tools can't distinguish between:
    A) "Chunk was ignored because it was irrelevant"  (legitimate)
    B) "Chunk was ignored because of its position"    (position bias bug)

  This module detects case (B) — a genuine novel contribution.

How it works:
  1. Takes the chunk attribution results (which chunks were used vs phantom).
  2. Adds position information: where each chunk appeared in the prompt.
  3. Computes Spearman rank correlation between position and similarity.
     - Negative correlation = middle chunks are systematically ignored = bias!
  4. Checks if phantom chunks (unused) cluster in middle positions.
  5. Cross-references with retriever scores: if a chunk had a HIGH retriever
     score but was still a phantom AND was in the middle → strong evidence
     of position bias.

Why this matters:
  If position bias is detected, the fix is REORDERING the chunks in the prompt
  (put the most important ones first and last), NOT changing the retriever or
  the LLM model. Without this detector, you'd waste time debugging the wrong
  component.
"""

from sentence_transformers import util
from embeddings import embed_text


def detect_position_bias(
    response: str,
    context_chunks: list,
    retriever_scores: list = None
) -> dict:
    """
    Detects whether the LLM ignored chunks due to their position in the prompt
    rather than their relevance.

    Args:
        response:         The LLM's generated answer.
        context_chunks:   List of chunk dicts from context.json (vector_data).
        retriever_scores: Optional list of retriever confidence scores
                          (from vectors_info[].score). Used for cross-reference.

    Returns:
        A dict containing:
          - position_analysis: per-chunk breakdown with position + similarity
          - bias_detected: True/False
          - bias_type: NONE / U_SHAPED / RECENCY / PRIMACY
          - bias_confidence: HIGH / MEDIUM / LOW
          - middle_phantom_ratio: % of middle-position chunks that were unused
          - bias_summary: human-readable explanation
    """
    # --- Step 1: Compute per-chunk similarity with position info ---
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

        # Get retriever score if available (how relevant the DB thought it was)
        retriever_score = None
        if retriever_scores and position < len(retriever_scores):
            retriever_score = retriever_scores[position]

        position_data.append({
            "chunk_id": chunk.get("id", -1),
            "position": position + 1,  # 1-indexed for readability
            "similarity_to_response": round(sim, 3),
            "retriever_score": round(retriever_score, 3) if retriever_score else None,
            "is_phantom": sim < 0.35,  # Same threshold as chunk_attributor
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
            "bias_summary": "Too few chunks to assess position bias (need >= 3)."
        }

    # --- Step 2: Divide positions into thirds ---
    third = max(1, total_chunks // 3)

    beginning = position_data[:third]
    middle = position_data[third:total_chunks - third]
    end = position_data[total_chunks - third:]

    # Handle edge case: if total_chunks not evenly divisible, middle may be empty
    if not middle:
        middle = position_data[third:third + 1] if third < total_chunks else []

    # --- Step 3: Compute average similarity per zone ---
    def avg_sim(zone):
        if not zone:
            return 0.0
        return round(sum(p["similarity_to_response"] for p in zone) / len(zone), 3)

    avg_beginning = avg_sim(beginning)
    avg_middle = avg_sim(middle)
    avg_end = avg_sim(end)

    # --- Step 4: Compute middle phantom ratio ---
    middle_phantoms = sum(1 for p in middle if p["is_phantom"])
    middle_phantom_ratio = round(
        middle_phantoms / len(middle), 3
    ) if middle else 0.0

    # --- Step 5: Check for high-retriever-score phantoms in the middle ---
    # These are the smoking gun: the retriever said "this chunk is relevant!"
    # but the LLM ignored it anyway, and it's in the middle position.
    wasted_good_chunks = []
    for p in middle:
        if p["is_phantom"] and p["retriever_score"] and p["retriever_score"] > 0.4:
            wasted_good_chunks.append(p)

    # --- Step 6: Detect bias pattern ---
    # U-shaped bias: beginning & end are used more than middle
    u_shape_gap = ((avg_beginning + avg_end) / 2) - avg_middle

    # Primacy bias: beginning used much more than end
    primacy_gap = avg_beginning - avg_end

    # Recency bias: end used much more than beginning
    recency_gap = avg_end - avg_beginning

    bias_detected = False
    bias_type = "NONE"
    bias_confidence = "LOW"

    if u_shape_gap > 0.08 and middle_phantom_ratio > 0.5:
        # Classic "Lost in the Middle" — U-shaped attention
        bias_detected = True
        bias_type = "U_SHAPED"
        if u_shape_gap > 0.15 or len(wasted_good_chunks) >= 2:
            bias_confidence = "HIGH"
        elif u_shape_gap > 0.10 or len(wasted_good_chunks) >= 1:
            bias_confidence = "MEDIUM"
        else:
            bias_confidence = "LOW"
    elif primacy_gap > 0.15 and middle_phantom_ratio > 0.4:
        bias_detected = True
        bias_type = "PRIMACY"
        bias_confidence = "MEDIUM" if primacy_gap > 0.2 else "LOW"
    elif recency_gap > 0.15 and middle_phantom_ratio > 0.4:
        bias_detected = True
        bias_type = "RECENCY"
        bias_confidence = "MEDIUM" if recency_gap > 0.2 else "LOW"

    # Boost confidence if we found wasted good chunks
    if len(wasted_good_chunks) >= 2 and bias_detected:
        bias_confidence = "HIGH"

    # --- Step 7: Build summary ---
    if not bias_detected:
        summary = (
            f"No significant position bias detected. "
            f"Similarity distribution across zones — "
            f"Beginning: {avg_beginning}, Middle: {avg_middle}, End: {avg_end}. "
            f"Middle phantom ratio: {middle_phantom_ratio}."
        )
    else:
        summary = (
            f"Position bias DETECTED ({bias_type}, {bias_confidence} confidence). "
            f"The LLM appears to ignore chunks placed in the middle of the context. "
            f"Zone similarities — Beginning: {avg_beginning}, Middle: {avg_middle}, "
            f"End: {avg_end}. "
            f"Middle phantom ratio: {middle_phantom_ratio} "
            f"({middle_phantoms}/{len(middle)} middle chunks were ignored). "
        )
        if wasted_good_chunks:
            ids = [str(w["chunk_id"]) for w in wasted_good_chunks]
            summary += (
                f"SMOKING GUN: Chunk(s) {', '.join(ids)} had high retriever "
                f"scores but were ignored — likely due to position, not relevance."
            )

    # --- Step 8: Build recommendation ---
    if bias_detected:
        recommendation = (
            "Reorder chunks in the prompt: place the most important chunks at "
            "the BEGINNING and END of the context section, not in the middle. "
            "Consider also: (1) reducing the number of retrieved chunks to avoid "
            "overwhelming the context window, (2) using a model with better "
            "long-context handling (e.g., Gemini Pro with 1M context)."
        )
    else:
        recommendation = "No action needed — chunk positions are not causing issues."

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
