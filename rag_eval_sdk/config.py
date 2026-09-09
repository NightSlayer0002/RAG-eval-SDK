"""Configuration for verification and resource budgets.

Defaults are conservative operating points, not benchmark-tuned constants.
Production and benchmark reports must record a calibration identifier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationConfig:
    max_claims: int = 24
    claim_granularity: str = "clause"
    max_response_characters: int = 12000
    max_input_chunks: int = 64
    max_chunk_characters: int = 6000
    evidence_per_claim: int = 3
    evidence_strategy: str = "coverage_aware"
    max_evidence_chars_per_claim: int = 6000
    global_pair_enabled: bool = True
    support_threshold: float = 0.70
    contradiction_threshold: float = 0.55
    uncertainty_margin: float = 0.10
    audit_enabled: bool = True
    audit_high_confidence_claims: bool = True
    max_audit_claims: int = 2
    max_escalations: int = 4
    min_necessity_drop: float = 0.08
    min_mutation_drop: float = 0.06
    max_order_shift: float = 0.12
    critical_claim_multiplier: float = 1.20
    high_claim_multiplier: float = 1.10
    fail_on_unverifiable: bool = True
    calibration_id: str = "uncalibrated-defaults-v1"

    def __post_init__(self) -> None:
        if self.max_claims < 1:
            raise ValueError("max_claims must be positive")
        if self.claim_granularity not in {"sentence", "clause"}:
            raise ValueError("claim_granularity must be 'sentence' or 'clause'")
        if self.max_response_characters < 128:
            raise ValueError("max_response_characters must be at least 128")
        if self.max_input_chunks < 1:
            raise ValueError("max_input_chunks must be positive")
        if self.max_chunk_characters < 128:
            raise ValueError("max_chunk_characters must be at least 128")
        if self.evidence_per_claim < 1:
            raise ValueError("evidence_per_claim must be positive")
        if self.evidence_strategy not in {"semantic", "coverage_aware"}:
            raise ValueError(
                "evidence_strategy must be 'semantic' or 'coverage_aware'"
            )
        if self.max_evidence_chars_per_claim < 128:
            raise ValueError("max_evidence_chars_per_claim must be at least 128")
        if self.max_audit_claims < 0 or self.max_escalations < 0:
            raise ValueError("audit and escalation budgets cannot be negative")
        for name in (
            "support_threshold",
            "contradiction_threshold",
            "uncertainty_margin",
            "min_necessity_drop",
            "min_mutation_drop",
            "max_order_shift",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in ("critical_claim_multiplier", "high_claim_multiplier"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class HeuristicWeights:
    semantic: float = 0.55
    lexical_coverage: float = 0.25
    critical_coverage: float = 0.20
    numeric_mismatch_penalty: float = 0.65
    negation_mismatch_penalty: float = 0.55

    def __post_init__(self) -> None:
        positive = self.semantic + self.lexical_coverage + self.critical_coverage
        if abs(positive - 1.0) > 1e-6:
            raise ValueError("positive heuristic weights must sum to 1")
