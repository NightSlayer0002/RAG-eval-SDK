"""
failure_classifier.py
---------------------
NOVEL FEATURE #2: Failure Mode Classifier

When a RAG pipeline gives a bad answer, this module automatically diagnoses
WHY it failed. It separates the system into components and blames the right one:

  1. RETRIEVER: The vector DB that fetches relevant chunks from memory.
     Measured by: the cosine similarity scores from the vector search
     (already present in context.json as "vectors_info[].score").
     If these scores are low, the retriever fetched irrelevant chunks.

  2. GENERATOR (the LLM): Given the retrieved chunks, did the LLM actually
     USE them? Or did it ignore them and make things up (hallucinate)?
     Measured by: the hallucination_score from evaluators.py.

  3. CONTEXT CONFLICTS (Novel Feature #3): Did the retrieved chunks contradict
     each other? If so, the retriever gave the LLM mixed signals.

  4. POSITION BIAS (Novel Feature #4): Did the LLM ignore good chunks because
     they were placed in the middle of the context window?

Failure Modes:
  - HEALTHY:            All components working well.
  - RETRIEVER_FAILURE:  Bad chunks fetched → generator had nothing to work with.
  - GENERATOR_FAILURE:  Good chunks fetched → LLM ignored them and hallucinated.
  - CONTEXT_CONFLICT:   Retrieved chunks contradict each other → confusing the LLM.
  - POSITION_BIAS:      Good chunks ignored due to position in prompt ("Lost in the Middle").
  - BOTH_FAILED:        Multiple components failed independently.
"""

# Thresholds (tunable)
RETRIEVER_GOOD_THRESHOLD = 0.40   # avg top-3 retriever score to be "good"
HALLUCINATION_GOOD_THRESHOLD = 0.40  # hallucination score below this = "good"


def classify_failure_mode(
    relevance_score: float,
    hallucination_score: float,
    retriever_scores: list,
    attribution_map: dict,
    conflict_analysis: dict = None,
    position_bias: dict = None
) -> dict:
    """
    Diagnoses which component of the RAG pipeline failed.

    Now integrates signals from ALL four novel features:
      - Chunk Attribution Map (which chunks were used)
      - Conflict Detector (did chunks contradict each other?)
      - Position Bias Detector (did chunk position cause ignoring?)

    Args:
        relevance_score:    How relevant the response is to the overall context (0-1).
        hallucination_score: 1 - max_chunk_similarity. High = hallucinated (0-1).
        retriever_scores:   List of cosine similarity scores from the vector DB search.
        attribution_map:    Output of score_attribution_map() for extra context.
        conflict_analysis:  Output of detect_conflicts() — inter-chunk contradictions.
        position_bias:      Output of detect_position_bias() — positional attention bias.

    Returns:
        A dict with diagnosis, confidence, and recommended action.
    """
    if conflict_analysis is None:
        conflict_analysis = {}
    if position_bias is None:
        position_bias = {}

    # --- Retriever Quality ---
    top_3 = sorted(retriever_scores, reverse=True)[:3]
    avg_retriever_score = round(sum(top_3) / len(top_3), 3) if top_3 else 0.0
    retriever_ok = avg_retriever_score >= RETRIEVER_GOOD_THRESHOLD

    # --- Generator Quality ---
    generator_ok = hallucination_score < HALLUCINATION_GOOD_THRESHOLD

    # --- New signals from Novel Features #3 and #4 ---
    has_conflicts = conflict_analysis.get("conflict_count", 0) > 0
    conflict_severity = conflict_analysis.get("conflict_severity", "NONE")
    has_position_bias = position_bias.get("bias_detected", False)
    bias_type = position_bias.get("bias_type", "NONE")

    # --- Enhanced Classification ---
    # Priority order: specific novel diagnoses first, then classic modes
    if retriever_ok and generator_ok and not has_conflicts and not has_position_bias:
        mode = "HEALTHY"
    elif has_conflicts and conflict_severity in ("MODERATE", "HIGH"):
        # Conflicts are a retriever-side problem — it fetched contradictory chunks
        mode = "CONTEXT_CONFLICT"
    elif has_position_bias and retriever_ok:
        # Good chunks fetched, but LLM ignored middle ones due to position
        mode = "POSITION_BIAS"
    elif not retriever_ok and generator_ok:
        mode = "RETRIEVER_FAILURE"
    elif retriever_ok and not generator_ok:
        mode = "GENERATOR_FAILURE"
    else:
        mode = "BOTH_FAILED"

    diagnosis = _get_diagnosis(
        mode, avg_retriever_score, hallucination_score,
        attribution_map, conflict_analysis, position_bias
    )
    recommendation = _get_recommendation(mode)
    confidence = _get_confidence(avg_retriever_score, hallucination_score)

    return {
        "failure_mode": mode,
        "confidence": confidence,
        "retriever_diagnosis": {
            "avg_top3_score": avg_retriever_score,
            "is_healthy": retriever_ok,
            "threshold_used": RETRIEVER_GOOD_THRESHOLD
        },
        "generator_diagnosis": {
            "hallucination_score": hallucination_score,
            "is_healthy": generator_ok,
            "threshold_used": HALLUCINATION_GOOD_THRESHOLD,
            "attributed_chunks_used": attribution_map.get("attributed_count", 0),
            "phantom_chunks": attribution_map.get("phantom_count", 0)
        },
        "conflict_diagnosis": {
            "has_conflicts": has_conflicts,
            "conflict_count": conflict_analysis.get("conflict_count", 0),
            "conflict_severity": conflict_severity
        },
        "position_bias_diagnosis": {
            "bias_detected": has_position_bias,
            "bias_type": bias_type,
            "bias_confidence": position_bias.get("bias_confidence", "LOW")
        },
        "diagnosis": diagnosis,
        "recommended_action": recommendation
    }


def _get_diagnosis(mode, retriever_score, hallucination_score,
                   attribution_map, conflict_analysis=None, position_bias=None):
    attr_count = attribution_map.get("attributed_count", 0)
    total = attribution_map.get("total_chunks_retrieved", 0)
    gap = attribution_map.get("coverage_gap", 0)

    if mode == "HEALTHY":
        return (
            f"System is healthy. Retriever fetched relevant chunks "
            f"(avg score: {retriever_score}) and the generator used them "
            f"(hallucination score: {hallucination_score}, "
            f"grounded in {attr_count}/{total} chunks). "
            f"No conflicts or position bias detected."
        )
    elif mode == "CONTEXT_CONFLICT":
        conflict_count = conflict_analysis.get("conflict_count", 0) if conflict_analysis else 0
        severity = conflict_analysis.get("conflict_severity", "UNKNOWN") if conflict_analysis else "UNKNOWN"
        return (
            f"Retrieved chunks contain {conflict_count} contradictions "
            f"(severity: {severity}). The retriever fetched chunks that disagree "
            f"with each other, forcing the LLM to reconcile conflicting information. "
            f"This is a retriever-side problem — it should not fetch contradictory sources."
        )
    elif mode == "POSITION_BIAS":
        bias_type = position_bias.get("bias_type", "UNKNOWN") if position_bias else "UNKNOWN"
        middle_ratio = position_bias.get("middle_phantom_ratio", 0) if position_bias else 0
        wasted = position_bias.get("wasted_good_chunks", 0) if position_bias else 0
        return (
            f"Position bias detected ({bias_type}). The LLM ignored chunks in the "
            f"middle of the context window — {round(middle_ratio * 100)}% of middle "
            f"chunks were phantom. {wasted} high-quality chunks were wasted due to "
            f"position, not relevance. This is the 'Lost in the Middle' phenomenon."
        )
    elif mode == "RETRIEVER_FAILURE":
        return (
            f"Retriever returned low-quality chunks (avg score: {retriever_score}, "
            f"threshold: {RETRIEVER_GOOD_THRESHOLD}). The generator had poor source "
            f"material to work with. Only {attr_count}/{total} chunks were attributed."
        )
    elif mode == "GENERATOR_FAILURE":
        return (
            f"Retriever fetched relevant chunks (avg score: {retriever_score}), "
            f"but the LLM largely ignored them. Hallucination score: "
            f"{hallucination_score} (threshold: {HALLUCINATION_GOOD_THRESHOLD}). "
            f"Coverage gap: {gap} — the response strays significantly from the context."
        )
    else:
        return (
            f"Both components failed. Retriever score: {retriever_score} "
            f"(threshold: {RETRIEVER_GOOD_THRESHOLD}). "
            f"Hallucination score: {hallucination_score} "
            f"(threshold: {HALLUCINATION_GOOD_THRESHOLD}). "
            f"Full pipeline review required."
        )


def _get_recommendation(mode: str) -> str:
    recommendations = {
        "HEALTHY": (
            "No action needed. Monitor over more queries to ensure consistency."
        ),
        "CONTEXT_CONFLICT": (
            "Fix the retriever's deduplication: (1) add a conflict-aware reranker that "
            "detects contradictory chunks before passing to the LLM, "
            "(2) implement source-date prioritization (prefer newer documents), "
            "(3) add metadata filtering to exclude outdated or superseded documents, "
            "(4) consider a 'single source of truth' rule for factual claims."
        ),
        "POSITION_BIAS": (
            "Reorder chunks in the prompt: (1) place the most relevant chunks at the "
            "BEGINNING and END of the context (not the middle), "
            "(2) reduce total chunk count to avoid overwhelming the context window, "
            "(3) try a model with better long-context handling, "
            "(4) consider splitting the query into sub-queries with fewer chunks each."
        ),
        "RETRIEVER_FAILURE": (
            "Fix the retriever: (1) try a stronger embedding model for indexing, "
            "(2) increase top-k retrieved chunks, "
            "(3) improve document chunking strategy (smaller/larger chunks), "
            "(4) check if the knowledge base is up to date."
        ),
        "GENERATOR_FAILURE": (
            "Fix the generator: (1) strengthen the system prompt — explicitly instruct "
            "the LLM to ONLY use the provided context, "
            "(2) lower the temperature (e.g., temperature=0.2), "
            "(3) try a larger/more capable model, "
            "(4) consider adding retrieved context directly into every prompt turn."
        ),
        "BOTH_FAILED": (
            "Full pipeline review needed. Start with retriever (fix chunk quality first), "
            "then re-evaluate generation. Check if the knowledge base covers the query domain."
        )
    }
    return recommendations.get(mode, "Unknown mode — manual review required.")


def _get_confidence(retriever_score: float, hallucination_score: float) -> str:
    """
    Returns a human-readable confidence label for the diagnosis.
    Scores far from thresholds = high confidence in the classification.
    """
    retriever_margin = abs(retriever_score - RETRIEVER_GOOD_THRESHOLD)
    hallucination_margin = abs(hallucination_score - HALLUCINATION_GOOD_THRESHOLD)
    avg_margin = (retriever_margin + hallucination_margin) / 2

    if avg_margin > 0.15:
        return "HIGH"
    elif avg_margin > 0.07:
        return "MEDIUM"
    else:
        return "LOW"
