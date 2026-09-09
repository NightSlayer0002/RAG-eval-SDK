"""Validation-only threshold calibration and dependency-free binary metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    balanced_accuracy: float


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    positive_when: str
    objective: str
    objective_value: float
    metrics: BinaryMetrics
    examples: int
    positives: int
    calibration_id: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["metrics"] = asdict(self.metrics)
        return result


@dataclass(frozen=True)
class RankingMetrics:
    auroc: float
    average_precision: float
    examples: int
    positives: int


def binary_metrics(
    scores: Sequence[float],
    labels: Sequence[int | bool],
    threshold: float,
    *,
    positive_when: Literal["higher", "lower"] = "higher",
) -> BinaryMetrics:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if positive_when not in {"higher", "lower"}:
        raise ValueError("positive_when must be 'higher' or 'lower'")

    predictions = [
        float(score) >= threshold if positive_when == "higher" else float(score) <= threshold
        for score in scores
    ]
    return binary_metrics_from_predictions(predictions, labels)


def binary_metrics_from_predictions(
    predictions: Sequence[int | bool], labels: Sequence[int | bool]
) -> BinaryMetrics:
    """Compute metrics from explicit decisions, including failure policies."""

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    predicted = [bool(value) for value in predictions]
    expected = [bool(label) for label in labels]
    tp = sum(prediction and label for prediction, label in zip(predicted, expected))
    fp = sum(prediction and not label for prediction, label in zip(predicted, expected))
    tn = sum(not prediction and not label for prediction, label in zip(predicted, expected))
    fn = sum(not prediction and label for prediction, label in zip(predicted, expected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(expected) if expected else 0.0
    balanced = (recall + specificity) / 2.0
    return BinaryMetrics(tp, fp, tn, fn, precision, recall, f1, accuracy, balanced)


def select_threshold(
    validation_scores: Sequence[float],
    validation_labels: Sequence[int | bool],
    *,
    positive_when: Literal["higher", "lower"] = "higher",
    objective: Literal["f1", "balanced_accuracy"] = "f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> CalibrationResult:
    """Select an operating point using validation data only.

    The caller is responsible for passing a validation split.  The returned ID
    fingerprints the inputs and policy so a test report can name exactly which
    calibration produced its threshold.
    """

    if len(validation_scores) != len(validation_labels):
        raise ValueError("validation_scores and validation_labels must match")
    if not validation_scores:
        raise ValueError("at least one validation example is required")
    labels = [bool(label) for label in validation_labels]
    if len(set(labels)) < 2:
        raise ValueError("calibration requires both positive and negative examples")
    for name, value in (("min_precision", min_precision), ("min_recall", min_recall)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")

    values = sorted({float(score) for score in validation_scores})
    epsilon = max(1e-12, (values[-1] - values[0]) * 1e-12)
    midpoints = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
    candidates = [values[0] - epsilon, *midpoints, values[-1] + epsilon]
    feasible: list[tuple[float, BinaryMetrics]] = []
    for threshold in candidates:
        metrics = binary_metrics(
            validation_scores,
            validation_labels,
            threshold,
            positive_when=positive_when,
        )
        if min_precision is not None and metrics.precision < min_precision:
            continue
        if min_recall is not None and metrics.recall < min_recall:
            continue
        feasible.append((threshold, metrics))
    if not feasible:
        raise ValueError("no threshold satisfies the requested precision/recall constraints")

    metric_name = "f1" if objective == "f1" else "balanced_accuracy"
    # Prefer fewer false positives, then fewer false negatives, on objective ties.
    threshold, metrics = max(
        feasible,
        key=lambda item: (
            getattr(item[1], metric_name),
            -item[1].false_positive,
            -item[1].false_negative,
        ),
    )
    fingerprint_payload = {
        "scores": [round(float(score), 12) for score in validation_scores],
        "labels": [int(label) for label in labels],
        "positive_when": positive_when,
        "objective": objective,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "algorithm": "threshold-grid-v1",
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return CalibrationResult(
        threshold=threshold,
        positive_when=positive_when,
        objective=objective,
        objective_value=getattr(metrics, metric_name),
        metrics=metrics,
        examples=len(labels),
        positives=sum(labels),
        calibration_id=f"threshold-grid-v1:{digest}",
    )


def ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[int | bool],
    *,
    positive_when: Literal["higher", "lower"] = "higher",
) -> RankingMetrics:
    """Compute tie-aware AUROC and average precision without extra packages."""

    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and have equal length")
    expected = [bool(label) for label in labels]
    positives = sum(expected)
    negatives = len(expected) - positives
    if not positives or not negatives:
        raise ValueError("ranking metrics require both positive and negative examples")
    oriented = [float(score) if positive_when == "higher" else -float(score) for score in scores]

    ascending = sorted(zip(oriented, expected), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ascending):
        end = index + 1
        while end < len(ascending) and ascending[end][0] == ascending[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(label for _, label in ascending[index:end])
        index = end
    auroc = (
        rank_sum_positive - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    descending = list(reversed(ascending))
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    index = 0
    while index < len(descending):
        end = index + 1
        while end < len(descending) and descending[end][0] == descending[index][0]:
            end += 1
        group_positives = sum(label for _, label in descending[index:end])
        true_positive += group_positives
        false_positive += (end - index) - group_positives
        precision = true_positive / (true_positive + false_positive)
        average_precision += precision * group_positives / positives
        index = end
    return RankingMetrics(auroc, average_precision, len(expected), positives)
