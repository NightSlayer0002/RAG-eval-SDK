"""Typed, serializable records for RAG verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence


class Decision(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient_evidence"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class EvidenceChunk:
    id: str
    text: str
    retrieval_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairScore:
    consistency: float
    contradiction: float
    confidence: float
    backend: str
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("consistency", "contradiction", "confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value within [0, 1]")


@dataclass(frozen=True)
class InterventionResult:
    applied: bool
    base_consistency: float
    removal_consistency: float | None = None
    mutation_consistency: float | None = None
    mutation_contradiction: float | None = None
    necessity_drop: float | None = None
    mutation_drop: float | None = None
    contradiction_gain: float | None = None
    reordered_consistency: float | None = None
    order_shift_score: float = 0.0
    order_unstable: bool = False
    sensitivity_score: float = 0.0
    audit_failed: bool = False
    reason: str = ""
    probes: int = 0
    input_characters: int = 0


@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    text: str
    decision: Decision
    severity: Severity
    risk_score: float
    pair_score: PairScore
    evidence_ids: Sequence[str]
    evidence_scores: Sequence[float]
    audit: InterventionResult | None = None
    reasons: Sequence[str] = field(default_factory=tuple)
    span_start: int | None = None
    span_end: int | None = None

    def __post_init__(self) -> None:
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("claim span_start and span_end must both be set or both be None")
        if self.span_start is not None and (
            self.span_start < 0 or self.span_end <= self.span_start
        ):
            raise ValueError("claim offsets must satisfy 0 <= span_start < span_end")


@dataclass
class ResourceUsage:
    elapsed_ms: float = 0.0
    claims: int = 0
    evidence_pairs_scored: int = 0
    verifier_batches: int = 0
    audit_probes: int = 0
    remote_calls: int = 0
    input_characters: int = 0
    estimated_input_tokens: int = 0
    peak_evidence_characters: int = 0
    input_chunks_received: int = 0
    input_chunks_used: int = 0
    truncated_chunks: int = 0
    response_characters_received: int = 0
    response_characters_used: int = 0
    truncated_claims: int = 0
    similarity_matrix_bytes: int = 0


@dataclass(frozen=True)
class VerificationReport:
    decision: Decision
    risk_score: float
    grounding_risk: float
    faithfulness: float
    claims: Sequence[ClaimResult]
    usage: ResourceUsage
    calibration_id: str
    verifier: str
    global_pair_score: PairScore | None = None
    schema_version: str = "2.0"
    warnings: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_chunks(chunks: Sequence[EvidenceChunk | Mapping[str, Any] | str]) -> list[EvidenceChunk]:
    """Normalize supported public input forms without mutating caller data."""

    normalized: list[EvidenceChunk] = []
    for index, chunk in enumerate(chunks):
        if isinstance(chunk, EvidenceChunk):
            candidate = chunk
        elif isinstance(chunk, str):
            candidate = EvidenceChunk(id=str(index), text=chunk)
        elif isinstance(chunk, Mapping):
            text = chunk.get("text", "")
            score = chunk.get("retrieval_score", chunk.get("score"))
            metadata = chunk.get("metadata", {})
            candidate = EvidenceChunk(
                id=str(chunk.get("id", index)),
                text=str(text) if text is not None else "",
                retrieval_score=float(score) if score is not None else None,
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            )
        else:
            raise TypeError(f"Unsupported chunk type at position {index}: {type(chunk)!r}")
        if candidate.text.strip():
            normalized.append(candidate)
    return normalized
