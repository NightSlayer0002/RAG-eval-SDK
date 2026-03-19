"""Unit tests for failure_classifier.py — all 6 failure modes."""
import pytest
from rag_eval_sdk.failure_classifier import classify_failure_mode


def _make_attribution(attributed=3, phantom=2, gap=0.2):
    return {"attributed_count": attributed, "phantom_count": phantom,
            "total_chunks_retrieved": attributed + phantom, "coverage_gap": gap}

def _make_conflict(count=0, severity="NONE"):
    return {"conflict_count": count, "conflict_severity": severity, "conflict_pairs": []}

def _make_bias(detected=False, bias_type="NONE", confidence="LOW", phantom_ratio=0.0, wasted=0):
    return {"bias_detected": detected, "bias_type": bias_type, "bias_confidence": confidence,
            "middle_phantom_ratio": phantom_ratio, "wasted_good_chunks": wasted}


class TestFailureModeClassifier:
    def test_healthy_system(self):
        result = classify_failure_mode(
            relevance_score=0.8, hallucination_score=0.2,
            retriever_scores=[0.9, 0.85, 0.8],
            attribution_map=_make_attribution(gap=0.1),
            conflict_analysis=_make_conflict(), position_bias=_make_bias()
        )
        assert result["failure_mode"] == "HEALTHY"

    def test_retriever_failure(self):
        result = classify_failure_mode(
            relevance_score=0.2, hallucination_score=0.2,
            retriever_scores=[0.1, 0.15, 0.1],
            attribution_map=_make_attribution(attributed=1, phantom=4, gap=0.5),
            conflict_analysis=_make_conflict(), position_bias=_make_bias()
        )
        assert result["failure_mode"] == "RETRIEVER_FAILURE"

    def test_generator_failure(self):
        result = classify_failure_mode(
            relevance_score=0.3, hallucination_score=0.75,
            retriever_scores=[0.9, 0.85, 0.8],
            attribution_map=_make_attribution(attributed=0, phantom=5, gap=0.8),
            conflict_analysis=_make_conflict(), position_bias=_make_bias()
        )
        assert result["failure_mode"] == "GENERATOR_FAILURE"

    def test_context_conflict(self):
        result = classify_failure_mode(
            relevance_score=0.6, hallucination_score=0.3,
            retriever_scores=[0.9, 0.85, 0.8],
            attribution_map=_make_attribution(gap=0.2),
            conflict_analysis=_make_conflict(count=3, severity="MODERATE"),
            position_bias=_make_bias()
        )
        assert result["failure_mode"] == "CONTEXT_CONFLICT"

    def test_position_bias(self):
        result = classify_failure_mode(
            relevance_score=0.5, hallucination_score=0.3,
            retriever_scores=[0.85, 0.9, 0.8],
            attribution_map=_make_attribution(gap=0.3),
            conflict_analysis=_make_conflict(),
            position_bias=_make_bias(detected=True, bias_type="U_SHAPED", confidence="HIGH",
                                     phantom_ratio=0.6, wasted=2)
        )
        assert result["failure_mode"] == "POSITION_BIAS"

    def test_both_failed(self):
        result = classify_failure_mode(
            relevance_score=0.1, hallucination_score=0.8,
            retriever_scores=[0.1, 0.1, 0.1],
            attribution_map=_make_attribution(attributed=0, phantom=5, gap=0.9),
            conflict_analysis=_make_conflict(), position_bias=_make_bias()
        )
        assert result["failure_mode"] == "BOTH_FAILED"

    def test_output_structure(self):
        result = classify_failure_mode(
            relevance_score=0.7, hallucination_score=0.2,
            retriever_scores=[0.8, 0.75, 0.7],
            attribution_map=_make_attribution(),
            conflict_analysis=_make_conflict(), position_bias=_make_bias()
        )
        for key in ["failure_mode", "confidence", "diagnosis", "recommended_action",
                    "retriever_diagnosis", "generator_diagnosis",
                    "conflict_diagnosis", "position_bias_diagnosis"]:
            assert key in result

    def test_confidence_high_when_clear(self):
        result = classify_failure_mode(
            relevance_score=0.9, hallucination_score=0.05,
            retriever_scores=[0.95, 0.9, 0.88],
            attribution_map=_make_attribution(gap=0.05),
            conflict_analysis=_make_conflict(), position_bias=_make_bias()
        )
        assert result["confidence"] in ("HIGH", "MEDIUM")

    def test_conflict_takes_priority(self):
        result = classify_failure_mode(
            relevance_score=0.5, hallucination_score=0.6,
            retriever_scores=[0.85, 0.8, 0.75],
            attribution_map=_make_attribution(gap=0.4),
            conflict_analysis=_make_conflict(count=4, severity="HIGH"),
            position_bias=_make_bias(detected=True, bias_type="U_SHAPED")
        )
        assert result["failure_mode"] == "CONTEXT_CONFLICT"
