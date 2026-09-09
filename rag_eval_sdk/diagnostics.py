"""Stage-gated failure hypotheses from explicit observability signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Sequence

from .position_experiment import PositionEffect
from .schema import Decision, VerificationReport


class FailureHypothesis(str, Enum):
    HEALTHY = "healthy"
    KNOWLEDGE_GAP = "knowledge_gap"
    RETRIEVAL_COVERAGE = "retrieval_coverage_failure"
    RANKING_OR_PACKING = "ranking_or_context_packing_failure"
    CONFLICTING_CONTEXT = "conflicting_context"
    POSITION_SENSITIVITY = "position_sensitivity"
    GENERATION_GROUNDING = "generation_grounding_failure"
    EVALUATOR_INSENSITIVITY = "evaluator_insensitivity"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class StageObservations:
    answer_supported_in_corpus: bool | None = None
    answer_supported_in_retrieved_set: bool | None = None
    answer_supported_in_prompt: bool | None = None
    conflicting_evidence_observed: bool | None = None
    position_effect: PositionEffect | None = None


@dataclass(frozen=True)
class DiagnosticReport:
    primary: FailureHypothesis
    hypotheses: Sequence[FailureHypothesis]
    confidence: float
    evidence: Sequence[str]
    missing_observations: Sequence[str]
    next_experiment: str
    caveat: str = (
        "This is stage localization from observed interventions, not proof of a unique root cause."
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_failure(
    verification: VerificationReport,
    observations: StageObservations | None = None,
) -> DiagnosticReport:
    """Localize a failure without treating embedding scores as ground truth."""

    observed = observations or StageObservations()
    hypotheses: list[FailureHypothesis] = []
    evidence: list[str] = []
    missing: list[str] = []
    unsupported = any(
        claim.decision in {Decision.CONTRADICTED, Decision.INSUFFICIENT, Decision.ESCALATE}
        for claim in verification.claims
    )
    audit_failures = sum(
        bool(claim.audit and claim.audit.audit_failed) for claim in verification.claims
    )

    if audit_failures:
        hypotheses.append(FailureHypothesis.EVALUATOR_INSENSITIVITY)
        evidence.append(f"{audit_failures} claim verifier result(s) failed evidence interventions.")
    if observed.conflicting_evidence_observed is True:
        hypotheses.append(FailureHypothesis.CONFLICTING_CONTEXT)
        evidence.append("A separate contradiction check observed incompatible evidence.")
    if observed.position_effect and observed.position_effect.position_sensitive:
        hypotheses.append(FailureHypothesis.POSITION_SENSITIVITY)
        evidence.append(
            f"Paired chunk-order treatments changed the score by "
            f"{observed.position_effect.maximum_gap:.3f}."
        )

    if unsupported:
        if observed.answer_supported_in_corpus is False:
            hypotheses.append(FailureHypothesis.KNOWLEDGE_GAP)
            evidence.append("The required answer was absent from the evaluated corpus.")
        elif (
            observed.answer_supported_in_corpus is True
            and observed.answer_supported_in_retrieved_set is False
        ):
            hypotheses.append(FailureHypothesis.RETRIEVAL_COVERAGE)
            evidence.append("Relevant evidence existed in the corpus but was not retrieved.")
        elif (
            observed.answer_supported_in_retrieved_set is True
            and observed.answer_supported_in_prompt is False
        ):
            hypotheses.append(FailureHypothesis.RANKING_OR_PACKING)
            evidence.append("Relevant retrieved evidence was dropped before generation.")
        elif observed.answer_supported_in_prompt is True:
            hypotheses.append(FailureHypothesis.GENERATION_GROUNDING)
            evidence.append("Relevant evidence reached the prompt, but output claims remained unsupported.")
    elif not hypotheses and verification.decision is Decision.SUPPORTED:
        hypotheses.append(FailureHypothesis.HEALTHY)
        evidence.append("All verifiable claims passed the configured support checks.")

    for name, value in (
        ("answer_supported_in_corpus", observed.answer_supported_in_corpus),
        ("answer_supported_in_retrieved_set", observed.answer_supported_in_retrieved_set),
        ("answer_supported_in_prompt", observed.answer_supported_in_prompt),
        ("conflicting_evidence_observed", observed.conflicting_evidence_observed),
    ):
        if value is None:
            missing.append(name)
    if observed.position_effect is None:
        missing.append("position_effect")

    if not hypotheses:
        hypotheses.append(FailureHypothesis.UNDETERMINED)
        evidence.append("Current telemetry cannot separate retrieval, packing, and generation causes.")
    priority = (
        FailureHypothesis.EVALUATOR_INSENSITIVITY,
        FailureHypothesis.KNOWLEDGE_GAP,
        FailureHypothesis.RETRIEVAL_COVERAGE,
        FailureHypothesis.RANKING_OR_PACKING,
        FailureHypothesis.CONFLICTING_CONTEXT,
        FailureHypothesis.POSITION_SENSITIVITY,
        FailureHypothesis.GENERATION_GROUNDING,
        FailureHypothesis.HEALTHY,
        FailureHypothesis.UNDETERMINED,
    )
    primary = next(item for item in priority if item in hypotheses)
    direct_observations = 5 - len(missing)
    confidence = min(1.0, 0.30 + 0.12 * direct_observations + 0.08 * bool(evidence))
    experiment = {
        FailureHypothesis.EVALUATOR_INSENSITIVITY: (
            "Compare the verifier with a stronger independent NLI or human label before changing the RAG system."
        ),
        FailureHypothesis.KNOWLEDGE_GAP: "Add or update a known relevant source, then repeat retrieval and generation.",
        FailureHypothesis.RETRIEVAL_COVERAGE: "Run a labelled retrieval recall test while holding generation fixed.",
        FailureHypothesis.RANKING_OR_PACKING: "Force the known relevant chunk into the final prompt and repeat generation.",
        FailureHypothesis.CONFLICTING_CONTEXT: "Remove each conflicting source in paired runs and compare the answer.",
        FailureHypothesis.POSITION_SENSITIVITY: "Repeat the paired beginning/middle/end treatment across several seeds.",
        FailureHypothesis.GENERATION_GROUNDING: "Hold prompt evidence fixed and compare constrained decoding or another generator.",
        FailureHypothesis.HEALTHY: "Continue monitoring on a larger, domain-stratified sample.",
        FailureHypothesis.UNDETERMINED: "Capture corpus, retrieval, packed-prompt, and paired-generation observations.",
    }[primary]
    return DiagnosticReport(
        primary=primary,
        hypotheses=tuple(hypotheses),
        confidence=confidence,
        evidence=tuple(evidence),
        missing_observations=tuple(missing),
        next_experiment=experiment,
    )

