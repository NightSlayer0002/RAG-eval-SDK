"""Controlled, batched evidence interventions for verifier meta-auditing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .schema import EvidenceChunk, InterventionResult, PairScore
from .text_features import mutate_critical_fact, pack_evidence
from .verifiers import PairVerifier, score_pairs


@dataclass(frozen=True)
class _AuditPlan:
    claim: str
    base_score: PairScore
    mutation_kind: str
    mutation_changed: bool
    removal_index: int
    mutation_index: int | None
    reorder_index: int | None
    probes: int
    input_characters: int


class EvidenceSensitivityAuditor:
    """Check whether a support score reacts when its evidence is changed.

    The audit is not a correctness oracle.  It detects a narrower failure mode:
    a verifier that reports support but barely reacts when the most relevant
    evidence is removed or a shared critical fact is changed.
    """

    def __init__(
        self,
        verifier: PairVerifier,
        *,
        max_evidence_characters: int = 6000,
        min_necessity_drop: float = 0.08,
        min_mutation_drop: float = 0.06,
        max_order_shift: float = 0.12,
    ) -> None:
        self.verifier = verifier
        self.max_evidence_characters = max_evidence_characters
        self.min_necessity_drop = min_necessity_drop
        self.min_mutation_drop = min_mutation_drop
        self.max_order_shift = max_order_shift

    def audit(
        self,
        claim: str,
        evidence: Sequence[EvidenceChunk],
        base_score: PairScore,
    ) -> InterventionResult:
        return self.audit_many([(claim, evidence, base_score)])[0]

    def audit_many(
        self,
        items: Sequence[tuple[str, Sequence[EvidenceChunk], PairScore]],
    ) -> list[InterventionResult]:
        """Run every intervention for several claims in one verifier batch."""

        probe_inputs: list[tuple[str, str]] = []
        plans: list[_AuditPlan | None] = []
        for claim, evidence, base_score in items:
            if not evidence:
                plans.append(None)
                continue
            remaining = pack_evidence(
                [chunk.text for chunk in evidence[1:]], self.max_evidence_characters
            )
            removal_index = len(probe_inputs)
            probe_inputs.append((remaining, claim))
            mutation = mutate_critical_fact(evidence[0].text, claim)
            mutation_index = None
            if mutation.changed:
                mutation_index = len(probe_inputs)
                mutated_pack = pack_evidence(
                    [mutation.text, *[chunk.text for chunk in evidence[1:]]],
                    self.max_evidence_characters,
                )
                probe_inputs.append((mutated_pack, claim))
            reorder_index = None
            if len(evidence) > 1:
                reorder_index = len(probe_inputs)
                reordered_pack = pack_evidence(
                    [chunk.text for chunk in reversed(evidence)],
                    self.max_evidence_characters,
                )
                probe_inputs.append((reordered_pack, claim))
            item_probes = [
                probe_inputs[index]
                for index in (removal_index, mutation_index, reorder_index)
                if index is not None
            ]
            plans.append(
                _AuditPlan(
                    claim=claim,
                    base_score=base_score,
                    mutation_kind=mutation.kind,
                    mutation_changed=mutation.changed,
                    removal_index=removal_index,
                    mutation_index=mutation_index,
                    reorder_index=reorder_index,
                    probes=len(item_probes),
                    input_characters=sum(
                        len(premise) + len(hypothesis)
                        for premise, hypothesis in item_probes
                    ),
                )
            )
        probe_scores = score_pairs(self.verifier, probe_inputs)
        results: list[InterventionResult] = []
        for item, plan in zip(items, plans):
            _, _, base_score = item
            if plan is None:
                results.append(
                    InterventionResult(
                        applied=False,
                        base_consistency=base_score.consistency,
                        audit_failed=True,
                        reason="No evidence was available for intervention.",
                    )
                )
                continue
            results.append(self._resolve_plan(plan, probe_scores))
        return results

    def _resolve_plan(
        self, plan: _AuditPlan, probe_scores: Sequence[PairScore]
    ) -> InterventionResult:
        base_score = plan.base_score
        removal_score = probe_scores[plan.removal_index]
        necessity_drop = max(0.0, base_score.consistency - removal_score.consistency)

        mutation_score: PairScore | None = None
        mutation_drop: float | None = None
        contradiction_gain: float | None = None
        if plan.mutation_index is not None:
            mutation_score = probe_scores[plan.mutation_index]
            mutation_drop = max(0.0, base_score.consistency - mutation_score.consistency)
            contradiction_gain = max(
                0.0, mutation_score.contradiction - base_score.contradiction
            )

        reorder_score = (
            probe_scores[plan.reorder_index]
            if plan.reorder_index is not None
            else None
        )
        order_shift = (
            max(
                abs(base_score.consistency - reorder_score.consistency),
                abs(base_score.contradiction - reorder_score.contradiction),
            )
            if reorder_score is not None
            else 0.0
        )
        order_unstable = order_shift > self.max_order_shift

        mutation_response = max(mutation_drop or 0.0, contradiction_gain or 0.0)
        removal_responsive = necessity_drop >= self.min_necessity_drop
        mutation_responsive = (
            plan.mutation_changed and mutation_response >= self.min_mutation_drop
        )

        # Removal alone is not expected to matter when evidence is redundant.
        # A failure therefore requires all available probes to be insensitive.
        if plan.mutation_changed:
            evidence_insensitive = not removal_responsive and not mutation_responsive
            reason = (
                "Verifier was insensitive to both evidence removal and a critical-fact mutation."
                if evidence_insensitive
                else f"Verifier reacted to controlled evidence changes ({plan.mutation_kind})."
            )
        else:
            evidence_insensitive = plan.reorder_index is None and not removal_responsive
            reason = (
                "No safe mutation was possible and removing the sole evidence did not change support."
                if evidence_insensitive
                else "Removal probe recorded; no deterministic critical-fact mutation was possible."
            )
        if order_unstable:
            reason += " Evidence order changed the verifier score beyond the stability limit."
        audit_failed = evidence_insensitive or order_unstable

        sensitivity = max(necessity_drop, mutation_response)
        return InterventionResult(
            applied=True,
            base_consistency=base_score.consistency,
            removal_consistency=removal_score.consistency,
            mutation_consistency=(mutation_score.consistency if mutation_score else None),
            mutation_contradiction=(mutation_score.contradiction if mutation_score else None),
            necessity_drop=necessity_drop,
            mutation_drop=mutation_drop,
            contradiction_gain=contradiction_gain,
            reordered_consistency=(
                reorder_score.consistency if reorder_score is not None else None
            ),
            order_shift_score=order_shift,
            order_unstable=order_unstable,
            sensitivity_score=sensitivity,
            audit_failed=audit_failed,
            reason=reason,
            probes=plan.probes,
            input_characters=plan.input_characters,
        )
