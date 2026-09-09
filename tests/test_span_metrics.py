import pytest
from dataclasses import replace

from benchmarks.detectors import SDKDetector
from benchmarks.schema import BenchmarkExample, ScoreRecord
from benchmarks.span_evaluation import summarize_span_localization
from benchmarks.span_metrics import (
    Span,
    aggregate_character_span_metrics,
    character_span_metrics,
    merge_spans,
    predicted_claim_spans,
)
from rag_eval_sdk.schema import (
    ClaimResult,
    Decision,
    PairScore,
    ResourceUsage,
    Severity,
    VerificationReport,
)
from rag_eval_sdk import HashingSimilarityBackend, RAGEvaluator, VerificationConfig


def _claim(identifier, decision, risk, start, end):
    return ClaimResult(
        claim_id=identifier,
        text=f"claim {identifier}",
        decision=decision,
        severity=Severity.LOW,
        risk_score=risk,
        pair_score=PairScore(1.0 - risk, 0.0, 1.0, "test"),
        evidence_ids=(),
        evidence_scores=(),
        span_start=start,
        span_end=end,
    )


def _report(claims):
    return VerificationReport(
        decision=Decision.ESCALATE,
        risk_score=max(claim.risk_score for claim in claims),
        grounding_risk=0.5,
        faithfulness=0.5,
        claims=claims,
        usage=ResourceUsage(),
        calibration_id="test",
        verifier="test",
    )


def test_merge_spans_unions_overlaps_and_adjacency():
    assert merge_spans([Span(8, 12), Span(0, 10), Span(12, 14)]) == (
        Span(0, 14),
    )


def test_character_metrics_count_interval_union_without_double_counting():
    metrics = character_span_metrics(
        [Span(0, 10), Span(8, 15)],
        [Span(5, 12)],
        text_length=20,
    )

    assert metrics.true_positive_characters == 7
    assert metrics.false_positive_characters == 8
    assert metrics.false_negative_characters == 0
    assert metrics.precision == pytest.approx(7 / 15)
    assert metrics.recall == 1.0
    assert metrics.f1 == pytest.approx(14 / 22)
    assert metrics.intersection_over_union == pytest.approx(7 / 15)


def test_all_negative_span_example_is_an_exact_match():
    metrics = character_span_metrics([], [], text_length=10)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.intersection_over_union == 1.0


def test_aggregate_span_metrics_reports_micro_and_macro_separately():
    result = aggregate_character_span_metrics(
        [
            ([Span(0, 5)], [Span(0, 5)], 10),
            ([Span(0, 5)], [], 10),
        ]
    )

    assert result["micro"]["precision"] == 0.5
    assert result["micro"]["recall"] == 1.0
    assert result["macro"]["f1"] == 0.5


def test_claim_projection_supports_decision_policy_and_fit_threshold():
    report = _report(
        (
            _claim("safe", Decision.SUPPORTED, 0.4, 0, 10),
            _claim("review", Decision.ESCALATE, 0.6, 10, 20),
        )
    )

    assert predicted_claim_spans(report) == (Span(10, 20),)
    assert predicted_claim_spans(report, claim_risk_threshold=0.3) == (
        Span(0, 20),
    )


def test_span_metrics_reject_out_of_bounds_predictions():
    with pytest.raises(ValueError, match="exceeds text length"):
        character_span_metrics([Span(0, 11)], [], text_length=10)


def _score_record(identifier, group, split, *, gold, claim_risk):
    return ScoreRecord(
        detector="sdk",
        example_id=identifier,
        group_id=group,
        split=split,
        hallucinated=bool(gold),
        score=claim_risk,
        elapsed_ms=1.0,
        usage={
            "risk_features": {"global_pair_risk": claim_risk},
            "claim_localization_schema": "claim-risk-spans-v1",
            "claim_predictions": [
                {
                    "claim_id": "claim-1",
                    "start": 0,
                    "end": 5,
                    "decision": "supported",
                    "risk_score": claim_risk,
                    "pair_consistency": 1.0 - claim_risk,
                    "pair_contradiction": 0.0,
                    "evidence_ids": [],
                }
            ],
            "response_characters_received": 10,
        },
        metadata={"gold_spans": gold},
    )


def test_span_summary_calibrates_only_on_non_test_records():
    records = [
        _score_record("cal-neg", "cal-g0", "train", gold=[], claim_risk=0.1),
        _score_record(
            "cal-pos",
            "cal-g1",
            "train",
            gold=[{"start": 0, "end": 5}],
            claim_risk=0.8,
        ),
        _score_record(
            "test-pos",
            "test-g0",
            "test",
            gold=[{"start": 0, "end": 5}],
            claim_risk=0.9,
        ),
        _score_record("test-neg", "test-g1", "test", gold=[], claim_risk=0.2),
    ]

    summary = summarize_span_localization(records)
    changed_test_gold = [
        replace(
            record,
            hallucinated=not record.hallucinated,
            metadata={"gold_spans": []},
        )
        if record.split == "test"
        else record
        for record in records
    ]
    changed = summarize_span_localization(changed_test_gold)

    assert summary["calibration"] == changed["calibration"]
    assert summary["calibration"]["threshold"] == 0.8
    assert summary["held_out"]["claim_risk_threshold"]["micro"]["f1"] == 1.0


def test_sdk_checkpoint_contains_compact_localization_signals():
    detector = SDKDetector(
        RAGEvaluator(
            config=VerificationConfig(audit_enabled=False),
            similarity_backend=HashingSimilarityBackend(128),
        ),
        name="sdk",
    )
    example = BenchmarkExample(
        id="example",
        query="Where is Paris?",
        response="Paris is in France.",
        contexts=("Paris is in France.",),
        hallucinated=False,
        split="test",
        group_id="source",
        metadata={"gold_spans": ()},
    )

    record = detector.score(example)
    prediction = record.usage["claim_predictions"][0]
    assert record.error is None
    assert record.usage["claim_localization_schema"] == "claim-risk-spans-v1"
    assert prediction["start"] == 0
    assert prediction["end"] == len(example.response)
    assert "pair_consistency" in prediction
    assert "pair_contradiction" in prediction
