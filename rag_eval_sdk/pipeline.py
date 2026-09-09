"""Budgeted, claim-level RAG verification pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .config import VerificationConfig
from .embeddings import SimilarityBackend, get_default_backend, similarity_matrix
from .evidence_audit import EvidenceSensitivityAuditor
from .schema import (
    ClaimResult,
    Decision,
    EvidenceChunk,
    PairScore,
    ResourceUsage,
    Severity,
    VerificationReport,
    coerce_chunks,
)
from .text_features import (
    critical_coverage,
    infer_severity,
    lexical_coverage,
    pack_evidence,
    split_claim_spans,
)
from .verifiers import HeuristicPairVerifier, PairVerifier, score_pairs


@dataclass
class _Draft:
    claim_id: str
    text: str
    span_start: int
    span_end: int
    severity: Severity
    evidence: list[EvidenceChunk]
    evidence_scores: list[float]
    packed_evidence: str
    pair_score: PairScore


class RAGEvaluator:
    """Evaluate a generated response against retrieved evidence under a budget.

    The evaluator never treats missing evidence as a safe answer.  It returns an
    explicit insufficient-evidence or abstain state and preserves per-claim
    evidence, features, interventions, and resource use for later review.
    """

    def __init__(
        self,
        *,
        config: VerificationConfig | None = None,
        similarity_backend: SimilarityBackend | None = None,
        pair_verifier: PairVerifier | None = None,
        escalation_verifier: PairVerifier | None = None,
    ) -> None:
        self.config = config or VerificationConfig()
        self.similarity_backend = similarity_backend or get_default_backend()
        self.pair_verifier = pair_verifier or HeuristicPairVerifier(
            self.similarity_backend
        )
        self.escalation_verifier = escalation_verifier
        self.auditor = EvidenceSensitivityAuditor(
            self.pair_verifier,
            max_evidence_characters=self.config.max_evidence_chars_per_claim,
            min_necessity_drop=self.config.min_necessity_drop,
            min_mutation_drop=self.config.min_mutation_drop,
            max_order_shift=self.config.max_order_shift,
        )

    def _severity_for(
        self, claim: str, index: int, overrides: Mapping[int | str, str | Severity] | None
    ) -> Severity:
        override = None
        if overrides:
            override = overrides.get(index, overrides.get(str(index)))
        if isinstance(override, Severity):
            return override
        return Severity(str(override or infer_severity(claim)).casefold())

    def _select_evidence(
        self,
        claim: str,
        chunks: Sequence[EvidenceChunk],
        row: np.ndarray,
    ) -> tuple[list[EvidenceChunk], list[float]]:
        if not chunks:
            return [], []
        semantic_order = [int(index) for index in np.argsort(-row, kind="stable")]
        if self.config.evidence_strategy == "semantic":
            order = semantic_order[: self.config.evidence_per_claim]
        else:
            order: list[int] = []
            if semantic_order:
                order.append(semantic_order[0])
            feature_views = (
                [critical_coverage(claim, chunk.text) for chunk in chunks],
                [lexical_coverage(claim, chunk.text) for chunk in chunks],
            )
            for feature_values in feature_views:
                ranking = sorted(
                    range(len(chunks)),
                    key=lambda index: (-feature_values[index], index),
                )
                if not ranking or feature_values[ranking[0]] <= 0.0:
                    continue
                for candidate in ranking:
                    if candidate not in order:
                        order.append(candidate)
                        break
                if len(order) >= self.config.evidence_per_claim:
                    break
            for candidate in semantic_order:
                if len(order) >= self.config.evidence_per_claim:
                    break
                if candidate not in order:
                    order.append(candidate)
        return (
            [chunks[index] for index in order],
            [float(row[index]) for index in order],
        )

    def _decide(self, score: PairScore, audit_failed: bool) -> tuple[Decision, list[str]]:
        reasons: list[str] = []
        if score.contradiction >= self.config.contradiction_threshold:
            reasons.append("Contradiction signal exceeded the configured threshold.")
            return Decision.CONTRADICTED, reasons
        if score.consistency >= self.config.support_threshold:
            if audit_failed:
                reasons.append("Support score failed the evidence-sensitivity audit.")
                return Decision.ESCALATE, reasons
            reasons.append("Support signal exceeded the configured threshold.")
            return Decision.SUPPORTED, reasons
        near_support = score.consistency >= (
            self.config.support_threshold - self.config.uncertainty_margin
        )
        near_contradiction = score.contradiction >= (
            self.config.contradiction_threshold - self.config.uncertainty_margin
        )
        if near_support or near_contradiction:
            reasons.append("Signals fall inside the configured uncertainty band.")
            return Decision.ESCALATE, reasons
        reasons.append("The selected evidence does not establish the claim.")
        return Decision.INSUFFICIENT, reasons

    def _risk(self, decision: Decision, score: PairScore, severity: Severity) -> float:
        if decision is Decision.SUPPORTED:
            base = 1.0 - score.consistency
        elif decision is Decision.CONTRADICTED:
            base = max(0.85, score.contradiction)
        elif decision is Decision.ESCALATE:
            base = max(0.65, 1.0 - score.consistency)
        else:
            base = max(0.60, 1.0 - score.consistency)
        severity_weight = {
            Severity.LOW: 1.0,
            Severity.MEDIUM: 1.05,
            Severity.HIGH: self.config.high_claim_multiplier,
            Severity.CRITICAL: self.config.critical_claim_multiplier,
        }[severity]
        return min(1.0, base * severity_weight)

    def evaluate(
        self,
        response: str,
        chunks: Sequence[EvidenceChunk | Mapping[str, object] | str],
        *,
        severity_overrides: Mapping[int | str, str | Severity] | None = None,
    ) -> VerificationReport:
        started = time.perf_counter()
        usage = ResourceUsage()
        usage.input_chunks_received = len(chunks)
        bounded_chunks = chunks[: self.config.max_input_chunks]
        supplied_chunks = coerce_chunks(bounded_chunks)
        normalized_chunks: list[EvidenceChunk] = []
        for chunk in supplied_chunks:
            clipped = chunk.text[: self.config.max_chunk_characters]
            if len(clipped) < len(chunk.text):
                usage.truncated_chunks += 1
            normalized_chunks.append(
                EvidenceChunk(
                    id=chunk.id,
                    text=clipped,
                    retrieval_score=chunk.retrieval_score,
                    metadata=chunk.metadata,
                )
            )
        if len(chunks) > len(bounded_chunks):
            usage.truncated_chunks += len(chunks) - len(bounded_chunks)
        usage.input_chunks_used = len(normalized_chunks)
        response_text = str(response or "")
        usage.response_characters_received = len(response_text)
        bounded_response = response_text[: self.config.max_response_characters]
        usage.response_characters_used = len(bounded_response)
        all_claim_spans = split_claim_spans(
            bounded_response,
            max_claims=None,
            granularity=self.config.claim_granularity,
        )
        claim_spans = all_claim_spans[: self.config.max_claims]
        claims = [span.text for span in claim_spans]
        usage.truncated_claims = max(0, len(all_claim_spans) - len(claim_spans))
        usage.claims = len(claims)

        warnings: list[str] = []
        if usage.truncated_chunks:
            warnings.append(
                f"Input budget truncated or dropped {usage.truncated_chunks} evidence chunk(s)."
            )
        if len(bounded_response) < len(response_text):
            warnings.append(
                "The response exceeded the character budget; the verification report is partial."
            )
        if usage.truncated_claims:
            warnings.append(
                f"The claim budget dropped {usage.truncated_claims} verification unit(s)."
            )
        if self.similarity_backend.name.startswith("hashing-"):
            warnings.append(
                "Hashing similarity is a lexical baseline; use a validated learned backend for production."
            )
        if not claims:
            usage.elapsed_ms = (time.perf_counter() - started) * 1000.0
            warnings.append("The response contained no verifiable sentence-level claims.")
            return VerificationReport(
                decision=Decision.ABSTAIN,
                risk_score=1.0,
                grounding_risk=1.0,
                faithfulness=0.0,
                claims=(),
                usage=usage,
                calibration_id=self.config.calibration_id,
                verifier=self.pair_verifier.name,
                warnings=tuple(warnings),
            )

        matrix = similarity_matrix(
            claims, [chunk.text for chunk in normalized_chunks], self.similarity_backend
        )
        usage.similarity_matrix_bytes = int(matrix.nbytes)
        prepared: list[
            tuple[str, str, int, int, Severity, list[EvidenceChunk], list[float], str]
        ] = []
        for index, claim_span in enumerate(claim_spans):
            claim = claim_span.text
            selected, selected_scores = self._select_evidence(
                claim, normalized_chunks, matrix[index]
            )
            packed = pack_evidence(
                [chunk.text for chunk in selected],
                self.config.max_evidence_chars_per_claim,
            )
            usage.input_characters += len(packed) + len(claim)
            usage.peak_evidence_characters = max(
                usage.peak_evidence_characters, len(packed)
            )
            prepared.append(
                (
                    f"claim-{index + 1}",
                    claim,
                    claim_span.start,
                    claim_span.end,
                    self._severity_for(claim, index, severity_overrides),
                    selected,
                    selected_scores,
                    packed,
                )
            )
        verification_pairs = [
            (packed, claim)
            for _, claim, _, _, _, _, _, packed in prepared
        ]
        global_pair_score = None
        if self.config.global_pair_enabled and normalized_chunks:
            global_evidence = pack_evidence(
                [chunk.text for chunk in normalized_chunks],
                self.config.max_evidence_chars_per_claim,
            )
            verification_pairs.append((global_evidence, bounded_response))
            usage.input_characters += len(global_evidence) + len(bounded_response)
            usage.peak_evidence_characters = max(
                usage.peak_evidence_characters, len(global_evidence)
            )
        pair_scores = score_pairs(self.pair_verifier, verification_pairs)
        if len(pair_scores) > len(prepared):
            global_pair_score = pair_scores[-1]
            pair_scores = pair_scores[:-1]
        usage.evidence_pairs_scored += len(pair_scores)
        if global_pair_score is not None:
            usage.evidence_pairs_scored += 1
        usage.verifier_batches += 1
        drafts: list[_Draft] = []
        for values, pair_score in zip(prepared, pair_scores):
            (
                claim_id,
                claim,
                span_start,
                span_end,
                severity,
                selected,
                selected_scores,
                packed,
            ) = values
            drafts.append(
                _Draft(
                    claim_id=claim_id,
                    text=claim,
                    span_start=span_start,
                    span_end=span_end,
                    severity=severity,
                    evidence=selected,
                    evidence_scores=selected_scores,
                    packed_evidence=packed,
                    pair_score=pair_score,
                )
            )

        if not normalized_chunks:
            warnings.append("No non-empty evidence chunks were supplied.")

        severity_priority = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        audit_candidates = [
            index
            for index, draft in enumerate(drafts)
            if self.config.audit_enabled
            and draft.evidence
            and (
                self.config.audit_high_confidence_claims
                or draft.severity in {Severity.HIGH, Severity.CRITICAL}
            )
            and draft.pair_score.consistency
            >= self.config.support_threshold - self.config.uncertainty_margin
        ]
        audit_candidates.sort(
            key=lambda index: (
                severity_priority[drafts[index].severity],
                -drafts[index].pair_score.consistency,
                index,
            )
        )
        audit_set = set(audit_candidates[: self.config.max_audit_claims])

        audits = {}
        if audit_set:
            audited_indices = sorted(audit_set)
            audit_results = self.auditor.audit_many(
                [
                    (
                        drafts[index].text,
                        drafts[index].evidence,
                        drafts[index].pair_score,
                    )
                    for index in audited_indices
                ]
            )
            audits = dict(zip(audited_indices, audit_results))
            usage.audit_probes += sum(result.probes for result in audit_results)
            usage.evidence_pairs_scored += sum(
                result.probes for result in audit_results
            )
            usage.verifier_batches += 1
            usage.input_characters += sum(
                result.input_characters for result in audit_results
            )

        results: list[ClaimResult] = []
        escalations_used = 0
        for index, draft in enumerate(drafts):
            audit = audits.get(index)
            decision, reasons = self._decide(
                draft.pair_score, bool(audit and audit.audit_failed)
            )
            final_pair_score = draft.pair_score
            if (
                decision is Decision.ESCALATE
                and self.escalation_verifier is not None
                and escalations_used < self.config.max_escalations
            ):
                escalated = self.escalation_verifier.score(
                    draft.packed_evidence, draft.text
                )
                escalations_used += 1
                usage.remote_calls += 1
                usage.evidence_pairs_scored += 1
                usage.verifier_batches += 1
                usage.input_characters += len(draft.packed_evidence) + len(draft.text)
                final_pair_score = PairScore(
                    consistency=escalated.consistency,
                    contradiction=escalated.contradiction,
                    confidence=escalated.confidence,
                    backend=f"{draft.pair_score.backend}->{escalated.backend}",
                    features={
                        **{f"local_{key}": value for key, value in draft.pair_score.features.items()},
                        "local_consistency": draft.pair_score.consistency,
                        "local_contradiction": draft.pair_score.contradiction,
                        **{f"escalated_{key}": value for key, value in escalated.features.items()},
                    },
                )
                decision, escalation_reasons = self._decide(
                    final_pair_score, bool(audit and audit.audit_failed)
                )
                reasons.append("A bounded escalation verifier evaluated this claim.")
                reasons.extend(escalation_reasons)
            if not draft.evidence:
                decision = Decision.INSUFFICIENT
                reasons = ["No evidence was available for this claim."]
            if audit is not None:
                reasons.append(audit.reason)
            results.append(
                ClaimResult(
                    claim_id=draft.claim_id,
                    text=draft.text,
                    decision=decision,
                    severity=draft.severity,
                    risk_score=self._risk(decision, final_pair_score, draft.severity),
                    pair_score=final_pair_score,
                    evidence_ids=tuple(chunk.id for chunk in draft.evidence),
                    evidence_scores=tuple(draft.evidence_scores),
                    audit=audit,
                    reasons=tuple(reasons),
                    span_start=draft.span_start,
                    span_end=draft.span_end,
                )
            )

        decisions = {result.decision for result in results}
        partial_report = (
            len(bounded_response) < len(response_text) or usage.truncated_claims > 0
        )
        if partial_report:
            overall = Decision.ABSTAIN
        elif Decision.CONTRADICTED in decisions:
            overall = Decision.CONTRADICTED
        elif Decision.ESCALATE in decisions:
            overall = Decision.ESCALATE
        elif Decision.INSUFFICIENT in decisions and self.config.fail_on_unverifiable:
            overall = Decision.INSUFFICIENT
        else:
            overall = Decision.SUPPORTED

        risk = 1.0 if partial_report else max(result.risk_score for result in results)
        grounding_risks: list[float] = []
        for result in results:
            base_risk = max(
                1.0 - result.pair_score.consistency,
                result.pair_score.contradiction,
            )
            if not result.evidence_ids:
                base_risk = 1.0
            if result.audit and result.audit.audit_failed:
                expected = max(
                    self.config.min_necessity_drop,
                    self.config.min_mutation_drop,
                    1e-9,
                )
                sensitivity_deficit = max(
                    0.0, 1.0 - result.audit.sensitivity_score / expected
                )
                base_risk = max(
                    base_risk,
                    result.pair_score.consistency * sensitivity_deficit,
                )
            grounding_risks.append(min(1.0, base_risk))
        grounding_risk = 1.0 if partial_report else max(grounding_risks)
        faithfulness = sum(result.pair_score.consistency for result in results) / len(results)
        usage.estimated_input_tokens = (usage.input_characters + 3) // 4
        usage.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return VerificationReport(
            decision=overall,
            risk_score=risk,
            grounding_risk=grounding_risk,
            faithfulness=faithfulness,
            claims=tuple(results),
            usage=usage,
            calibration_id=self.config.calibration_id,
            verifier=self.pair_verifier.name,
            global_pair_score=global_pair_score,
            warnings=tuple(warnings),
        )
