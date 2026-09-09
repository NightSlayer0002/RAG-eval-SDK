"""Tests for v2 decisions, intervention budgets, and no-evidence safety."""

import numpy as np

from rag_eval_sdk import (
    HHEMPairVerifier,
    HashingSimilarityBackend,
    RAGEvaluator,
    VerificationConfig,
)
from rag_eval_sdk.schema import Decision, EvidenceChunk, PairScore
from rag_eval_sdk.text_features import split_claim_spans, split_claims


class ReactiveVerifier:
    name = "test-reactive"

    def score(self, premise: str, hypothesis: str) -> PairScore:
        if not premise:
            return PairScore(0.0, 0.0, 1.0, self.name)
        if "[COUNTERFACTUAL_ENTITY]" in premise:
            return PairScore(0.1, 0.9, 1.0, self.name)
        return PairScore(0.92, 0.02, 0.9, self.name)


class InsensitiveVerifier:
    name = "test-insensitive"

    def score(self, premise: str, hypothesis: str) -> PairScore:
        return PairScore(0.92, 0.01, 0.9, self.name)


class BatchOnlyVerifier:
    name = "test-batch-only"

    def __init__(self):
        self.batch_calls = 0

    def score(self, premise: str, hypothesis: str) -> PairScore:
        raise AssertionError("the scalar path should not be used")

    def score_many(self, pairs):
        self.batch_calls += 1
        return [PairScore(0.8, 0.0, 0.8, self.name) for _ in pairs]


def _evaluator(verifier, **overrides):
    config = VerificationConfig(**overrides)
    return RAGEvaluator(
        config=config,
        similarity_backend=HashingSimilarityBackend(256),
        pair_verifier=verifier,
    )


def test_missing_evidence_cannot_pass():
    report = _evaluator(ReactiveVerifier()).evaluate(
        "Paris is the capital of France.", []
    )
    assert report.decision is Decision.INSUFFICIENT
    assert report.grounding_risk == 1.0
    assert report.claims[0].evidence_ids == ()


def test_reactive_support_passes_intervention_audit():
    report = _evaluator(ReactiveVerifier()).evaluate(
        "Paris is the capital of France.",
        [{"id": "source", "text": "Paris is the capital of France."}],
    )
    claim = report.claims[0]
    assert claim.decision is Decision.SUPPORTED
    assert claim.audit is not None
    assert claim.audit.applied is True
    assert claim.audit.audit_failed is False
    assert report.usage.audit_probes == 2


def test_insensitive_high_support_is_escalated():
    report = _evaluator(InsensitiveVerifier()).evaluate(
        "Paris is the capital of France.",
        [{"id": "source", "text": "Paris is the capital of France."}],
    )
    claim = report.claims[0]
    assert claim.audit is not None and claim.audit.audit_failed
    assert claim.decision is Decision.ESCALATE
    assert report.grounding_risk >= 0.9


def test_audit_budget_is_enforced():
    report = _evaluator(ReactiveVerifier(), max_audit_claims=1).evaluate(
        "Paris is in France. Berlin is in Germany. Rome is in Italy.",
        [
            "Paris is in France.",
            "Berlin is in Germany.",
            "Rome is in Italy.",
        ],
    )
    assert sum(claim.audit is not None for claim in report.claims) == 1
    assert report.usage.audit_probes <= 3


def test_audits_for_multiple_claims_share_one_batch():
    verifier = BatchOnlyVerifier()
    report = _evaluator(verifier, max_audit_claims=2).evaluate(
        "Paris is in France. Berlin is in Germany.",
        ["Paris is in France.", "Berlin is in Germany."],
    )
    assert sum(claim.audit is not None for claim in report.claims) == 2
    assert report.usage.verifier_batches == 2
    assert verifier.batch_calls == 2


def test_evidence_order_instability_is_exposed_and_escalated():
    class OrderSensitiveVerifier:
        name = "order-sensitive"

        def score(self, premise, hypothesis):
            consistency = 0.9 if premise.startswith("first") else 0.5
            return PairScore(consistency, 0.0, 0.9, self.name)

    report = _evaluator(
        OrderSensitiveVerifier(),
        evidence_per_claim=2,
        evidence_strategy="semantic",
        max_order_shift=0.12,
    ).evaluate(
        "The first report is authoritative.",
        [
            {"id": "first", "text": "first report is authoritative."},
            {"id": "second", "text": "second background passage."},
        ],
    )
    audit = report.claims[0].audit
    assert audit is not None and audit.order_unstable is True
    assert audit.order_shift_score > 0.12
    assert report.claims[0].decision is Decision.ESCALATE


def test_response_and_claim_budgets_are_reported():
    evaluator = _evaluator(
        ReactiveVerifier(), max_claims=1, max_response_characters=128
    )
    response = ("Paris is in France. " * 20).strip()
    report = evaluator.evaluate(response, ["Paris is in France."])
    assert report.usage.response_characters_received == len(response)
    assert report.usage.response_characters_used == 128
    assert report.usage.truncated_claims > 0
    assert report.decision is Decision.ABSTAIN
    assert report.grounding_risk == 1.0
    assert any("partial" in warning for warning in report.warnings)


def test_pipeline_uses_optional_batch_verifier_path():
    verifier = BatchOnlyVerifier()
    report = _evaluator(verifier, audit_enabled=False).evaluate(
        "Paris is in France. Berlin is in Germany.",
        ["Paris is in France. Berlin is in Germany."],
    )
    assert len(report.claims) == 2
    assert verifier.batch_calls == 1
    assert report.usage.evidence_pairs_scored == 3
    assert report.usage.verifier_batches == 1
    assert report.global_pair_score is not None


def test_hhem_adapter_batches_classifier_inputs_without_loading_a_model():
    calls = []

    def fake_classifier(prompts, *, top_k, batch_size):
        calls.append((prompts, top_k, batch_size))
        return [
            [
                {"label": "hallucinated", "score": 0.1},
                {"label": "consistent", "score": 0.9},
            ]
            for _ in prompts
        ]

    verifier = HHEMPairVerifier(batch_size=4)
    verifier._classifier = fake_classifier
    results = verifier.score_many([("a", "b"), ("c", "d")])
    assert [result.consistency for result in results] == [0.9, 0.9]
    assert len(calls) == 1
    assert calls[0][2] == 4


def test_hhem_name_discloses_whether_remote_code_is_pinned():
    assert "@unpinned" in HHEMPairVerifier().name
    assert "@abc123" in HHEMPairVerifier(model_revision="abc123").name


def test_clause_granularity_separates_independent_contrastive_claims():
    text = "Paris is in France, but Berlin is in Germany."
    assert split_claims(text, granularity="sentence") == [text]
    assert split_claims(text, granularity="clause") == [
        "Paris is in France",
        "Berlin is in Germany.",
    ]


def test_claim_spans_map_normalized_units_to_original_response():
    text = "  • Paris   is in France, but Berlin is in Germany.  "
    spans = split_claim_spans(text, granularity="clause")

    assert [span.text for span in spans] == [
        "Paris is in France",
        "Berlin is in Germany.",
    ]
    assert text[spans[0].start : spans[0].end] == "Paris   is in France"
    assert text[spans[1].start : spans[1].end] == "Berlin is in Germany."


def test_pipeline_exposes_label_free_response_offsets():
    response = "\nParis is in France.  Berlin is in Germany.\n"
    report = _evaluator(ReactiveVerifier(), audit_enabled=False).evaluate(
        response,
        ["Paris is in France. Berlin is in Germany."],
    )

    assert [
        response[claim.span_start : claim.span_end]
        for claim in report.claims
    ] == ["Paris is in France.", "Berlin is in Germany."]


def test_coverage_aware_selection_keeps_critical_fact_evidence():
    evaluator = _evaluator(
        ReactiveVerifier(),
        audit_enabled=False,
        evidence_per_claim=2,
        evidence_strategy="coverage_aware",
    )
    chunks = [
        EvidenceChunk("semantic", "General policy information."),
        EvidenceChunk("critical", "The 2026 limit is 47 percent."),
        EvidenceChunk("other", "Unrelated operational details."),
    ]
    selected, _ = evaluator._select_evidence(
        "The 2026 limit is 47 percent.", chunks, np.array([0.9, 0.3, 0.2])
    )
    assert [chunk.id for chunk in selected] == ["semantic", "critical"]
