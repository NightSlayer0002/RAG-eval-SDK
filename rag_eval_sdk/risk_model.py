"""Train-only response-risk calibration from auditable verifier features."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .schema import Decision, VerificationReport


RISK_FEATURE_SCHEMA = "multi-resolution-evidence-geometry-v3"
RISK_FEATURE_NAMES = (
    "global_pair_available",
    "global_pair_risk",
    "global_consistency",
    "global_contradiction",
    "claim_global_risk_gap",
    "grounding_risk",
    "business_risk",
    "max_pair_risk",
    "second_pair_risk",
    "mean_pair_risk",
    "unsupported_fraction",
    "contradicted_fraction",
    "uncertain_fraction",
    "mean_consistency",
    "minimum_consistency",
    "maximum_contradiction",
    "mean_best_evidence_similarity",
    "minimum_best_evidence_similarity",
    "no_evidence_fraction",
    "audited_fraction",
    "audit_failure_fraction",
    "mean_audit_sensitivity",
    "mean_order_shift",
    "maximum_order_shift",
    "order_instability_fraction",
    "log_claim_count",
    "partial_report",
)
MREG_COMPACT_FEATURE_NAMES = (
    "global_pair_risk",
    "max_pair_risk",
)


def extract_risk_features(report: VerificationReport) -> dict[str, float]:
    """Extract fixed, label-independent response features from a report."""

    claims = list(report.claims)
    count = len(claims)
    denominator = max(1, count)
    pair_risks = sorted(
        (
            max(1.0 - claim.pair_score.consistency, claim.pair_score.contradiction)
            for claim in claims
        ),
        reverse=True,
    )
    consistencies = [claim.pair_score.consistency for claim in claims]
    contradictions = [claim.pair_score.contradiction for claim in claims]
    best_evidence = [
        max(claim.evidence_scores) for claim in claims if claim.evidence_scores
    ]
    audited = [claim for claim in claims if claim.audit is not None]
    audit_failures = [claim for claim in audited if claim.audit and claim.audit.audit_failed]
    sensitivities = [
        claim.audit.sensitivity_score for claim in audited if claim.audit is not None
    ]
    order_shifts = [
        claim.audit.order_shift_score for claim in audited if claim.audit is not None
    ]
    uncertain = {Decision.INSUFFICIENT, Decision.ESCALATE, Decision.ABSTAIN}
    partial = bool(
        report.usage.truncated_claims
        or report.usage.response_characters_used
        < report.usage.response_characters_received
    )
    if report.global_pair_score is None:
        global_available = 0.0
        global_risk = 1.0
        global_consistency = 0.0
        global_contradiction = 0.0
    else:
        global_available = 1.0
        global_consistency = report.global_pair_score.consistency
        global_contradiction = report.global_pair_score.contradiction
        global_risk = max(1.0 - global_consistency, global_contradiction)
    values = {
        "global_pair_available": global_available,
        "global_pair_risk": global_risk,
        "global_consistency": global_consistency,
        "global_contradiction": global_contradiction,
        "claim_global_risk_gap": (
            (pair_risks[0] if pair_risks else 1.0) - global_risk
        ),
        "grounding_risk": report.grounding_risk,
        "business_risk": report.risk_score,
        "max_pair_risk": pair_risks[0] if pair_risks else 1.0,
        "second_pair_risk": pair_risks[1] if len(pair_risks) > 1 else (
            pair_risks[0] if pair_risks else 1.0
        ),
        "mean_pair_risk": sum(pair_risks) / denominator,
        "unsupported_fraction": sum(
            claim.decision is not Decision.SUPPORTED for claim in claims
        ) / denominator,
        "contradicted_fraction": sum(
            claim.decision is Decision.CONTRADICTED for claim in claims
        ) / denominator,
        "uncertain_fraction": sum(claim.decision in uncertain for claim in claims)
        / denominator,
        "mean_consistency": sum(consistencies) / denominator,
        "minimum_consistency": min(consistencies, default=0.0),
        "maximum_contradiction": max(contradictions, default=0.0),
        "mean_best_evidence_similarity": (
            sum(best_evidence) / len(best_evidence) if best_evidence else 0.0
        ),
        "minimum_best_evidence_similarity": min(best_evidence, default=0.0),
        "no_evidence_fraction": sum(not claim.evidence_ids for claim in claims)
        / denominator,
        "audited_fraction": len(audited) / denominator,
        "audit_failure_fraction": len(audit_failures) / max(1, len(audited)),
        "mean_audit_sensitivity": (
            sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
        ),
        "mean_order_shift": (
            sum(order_shifts) / len(order_shifts) if order_shifts else 0.0
        ),
        "maximum_order_shift": max(order_shifts, default=0.0),
        "order_instability_fraction": sum(
            bool(claim.audit and claim.audit.order_unstable) for claim in audited
        ) / max(1, len(audited)),
        "log_claim_count": math.log1p(count),
        "partial_report": float(partial),
    }
    return {name: float(values[name]) for name in RISK_FEATURE_NAMES}


@dataclass(frozen=True)
class RiskCalibrationModel:
    feature_schema: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    l2: float
    iterations: int
    calibration_id: str

    def predict_proba(self, features: Mapping[str, float]) -> float:
        values = np.asarray(
            [float(features[name]) for name in self.feature_names], dtype=np.float64
        )
        standardized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        logit = float(np.dot(standardized, np.asarray(self.coefficients)) + self.intercept)
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_risk_calibrator(
    feature_rows: Sequence[Mapping[str, float]],
    labels: Sequence[int | bool],
    *,
    l2: float = 0.10,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    feature_names: Sequence[str] = MREG_COMPACT_FEATURE_NAMES,
) -> RiskCalibrationModel:
    """Fit deterministic L2 logistic calibration without test-set access."""

    if len(feature_rows) != len(labels) or not feature_rows:
        raise ValueError("feature rows and labels must be non-empty and have equal length")
    expected = np.asarray([int(bool(label)) for label in labels], dtype=np.float64)
    if len(set(expected.tolist())) < 2:
        raise ValueError("risk calibration requires both positive and negative examples")
    if l2 < 0.0 or max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid risk-calibration optimizer settings")
    selected_features = tuple(feature_names)
    if not selected_features or len(set(selected_features)) != len(selected_features):
        raise ValueError("feature_names must be non-empty and unique")
    unknown = set(selected_features) - set(RISK_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown risk features: {sorted(unknown)}")
    matrix = np.asarray(
        [[float(row[name]) for name in selected_features] for row in feature_rows],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("risk features must be finite")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-9] = 1.0
    standardized = (matrix - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * l2
    regularizer[0, 0] = 0.0
    completed = 0
    for iteration in range(max_iterations):
        logits = np.clip(design @ parameters, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        gradient = design.T @ (probabilities - expected) / len(expected)
        gradient += regularizer @ parameters
        hessian = (design.T * curvature) @ design / len(expected) + regularizer
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        parameters -= update
        completed = iteration + 1
        if float(np.linalg.norm(update)) < tolerance:
            break
    fingerprint = {
        "schema": RISK_FEATURE_SCHEMA,
        "features": selected_features,
        "matrix": np.round(matrix, 10).tolist(),
        "labels": expected.astype(int).tolist(),
        "l2": l2,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return RiskCalibrationModel(
        feature_schema=RISK_FEATURE_SCHEMA,
        feature_names=selected_features,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in parameters[1:]),
        intercept=float(parameters[0]),
        l2=l2,
        iterations=completed,
        calibration_id=f"{RISK_FEATURE_SCHEMA}:logistic:{digest}",
    )
