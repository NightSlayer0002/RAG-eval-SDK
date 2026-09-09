"""
failure_classifier.py
---------------------
NOVEL FEATURE #2: Failure Mode Classifier

Diagnoses WHY a RAG pipeline gave a bad answer. Six failure modes:
  HEALTHY, RETRIEVER_FAILURE, GENERATOR_FAILURE,
  CONTEXT_CONFLICT, POSITION_BIAS, BOTH_FAILED.
"""

RETRIEVER_GOOD_THRESHOLD = 0.40
HALLUCINATION_GOOD_THRESHOLD = 0.40


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

    Args:
        relevance_score:    How relevant the response is to the context (0-1).
        hallucination_score: 1 - max_chunk_similarity. High = hallucinated.
        retriever_scores:   List of cosine similarity scores from vector DB.
        attribution_map:    Output of score_attribution_map().
        conflict_analysis:  Output of detect_conflicts().
        position_bias:      Output of detect_position_bias().

    Returns:
        failure_mode, confidence, diagnosis, recommended_action + sub-diagnoses.
    """
    if conflict_analysis is None:
        conflict_analysis = {}
    if position_bias is None:
        position_bias = {}

    top_3 = sorted(retriever_scores, reverse=True)[:3]
    avg_retriever_score = round(sum(top_3) / len(top_3), 3) if top_3 else 0.0
    retriever_ok = avg_retriever_score >= RETRIEVER_GOOD_THRESHOLD
    generator_ok = hallucination_score < HALLUCINATION_GOOD_THRESHOLD

    has_conflicts = conflict_analysis.get("conflict_count", 0) > 0
    conflict_severity = conflict_analysis.get("conflict_severity", "NONE")
    has_position_bias = position_bias.get("bias_detected", False)
    bias_type = position_bias.get("bias_type", "NONE")

    # Priority order: novel diagnoses first, then classic modes
    if retriever_ok and generator_ok and not has_conflicts and not has_position_bias:
        mode = "HEALTHY"
    elif has_conflicts and conflict_severity in ("MODERATE", "HIGH"):
        mode = "CONTEXT_CONFLICT"
    elif has_position_bias and retriever_ok:
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
        count = conflict_analysis.get("conflict_count", 0) if conflict_analysis else 0
        severity = conflict_analysis.get("conflict_severity", "UNKNOWN") if conflict_analysis else "UNKNOWN"
        return (
            f"Retrieved chunks contain {count} contradictions "
            f"(severity: {severity}). The retriever fetched chunks that disagree "
            f"with each other, forcing the LLM to reconcile conflicting information."
        )
    elif mode == "POSITION_BIAS":
        b_type = position_bias.get("bias_type", "UNKNOWN") if position_bias else "UNKNOWN"
        middle_ratio = position_bias.get("middle_phantom_ratio", 0) if position_bias else 0
        wasted = position_bias.get("wasted_good_chunks", 0) if position_bias else 0
        return (
            f"Position bias detected ({b_type}). The LLM ignored chunks in the "
            f"middle of the context window -- {round(middle_ratio * 100)}% of "
            f"middle chunks were phantom. {wasted} high-quality chunks wasted."
        )
    elif mode == "RETRIEVER_FAILURE":
        return (
            f"Retriever returned low-quality chunks (avg score: {retriever_score}, "
            f"threshold: {RETRIEVER_GOOD_THRESHOLD}). Only {attr_count}/{total} chunks attributed."
        )
    elif mode == "GENERATOR_FAILURE":
        return (
            f"Retriever fetched relevant chunks (avg score: {retriever_score}), "
            f"but the LLM largely ignored them. Hallucination score: "
            f"{hallucination_score}. Coverage gap: {gap}."
        )
    else:
        return (
            f"Both components failed. Retriever score: {retriever_score}, "
            f"Hallucination score: {hallucination_score}. Full pipeline review required."
        )


def _get_recommendation(mode: str) -> str:
    recommendations = {
        "HEALTHY": "No action needed. Monitor over more queries.",
        "CONTEXT_CONFLICT": (
            "Fix the retriever: add a conflict-aware reranker, implement "
            "source-date prioritization, add metadata filtering."
        ),
        "POSITION_BIAS": (
            "Reorder chunks: place the most relevant chunks at the BEGINNING "
            "and END of the context, not the middle. No retraining needed."
        ),
        "RETRIEVER_FAILURE": (
            "Fix the retriever: try a stronger embedding model, increase top-k, "
            "improve chunking strategy, check knowledge base is up to date."
        ),
        "GENERATOR_FAILURE": (
            "Fix the generator: strengthen the system prompt, lower temperature, "
            "try a larger model, instruct LLM to ONLY use provided context."
        ),
        "BOTH_FAILED": (
            "Full pipeline review needed. Fix retriever first (chunk quality), "
            "then re-evaluate generation."
        )
    }
    return recommendations.get(mode, "Unknown mode -- manual review required.")


def _get_confidence(retriever_score: float, hallucination_score: float) -> str:
    retriever_margin = abs(retriever_score - RETRIEVER_GOOD_THRESHOLD)
    hallucination_margin = abs(hallucination_score - HALLUCINATION_GOOD_THRESHOLD)
    avg_margin = (retriever_margin + hallucination_margin) / 2

    if avg_margin > 0.15:
        return "HIGH"
    elif avg_margin > 0.07:
        return "MEDIUM"
    else:
        return "LOW"
