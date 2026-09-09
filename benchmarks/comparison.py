"""Paired statistical comparison on a shared held-out example set."""

from __future__ import annotations

import math
import random
from typing import Sequence

from rag_eval_sdk.calibration import binary_metrics, binary_metrics_from_predictions

from .schema import ScoreRecord


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mcnemar_p_value(first_only_correct: int, second_only_correct: int) -> tuple[float, str]:
    discordant = first_only_correct + second_only_correct
    if discordant == 0:
        return 1.0, "exact"
    if discordant <= 1000:
        tail = sum(
            math.comb(discordant, index) * (0.5**discordant)
            for index in range(min(first_only_correct, second_only_correct) + 1)
        )
        return min(1.0, 2.0 * tail), "exact-binomial"
    statistic = ((abs(first_only_correct - second_only_correct) - 1.0) ** 2) / discordant
    return math.erfc(math.sqrt(statistic / 2.0)), "chi-square-continuity-corrected"


def compare_detectors(
    first_records: Sequence[ScoreRecord],
    second_records: Sequence[ScoreRecord],
    *,
    first_threshold: float,
    second_threshold: float,
    split: str = "test",
    bootstrap_samples: int = 1000,
    seed: int = 1701,
) -> dict[str, object]:
    first_all = {
        record.example_id: record for record in first_records if record.split == split
    }
    second_all = {
        record.example_id: record for record in second_records if record.split == split
    }
    if set(first_all) != set(second_all):
        raise ValueError("detectors do not contain the same held-out example IDs")
    all_ids = sorted(first_all)
    for example_id in all_ids:
        if first_all[example_id].hallucinated != second_all[example_id].hallucinated:
            raise ValueError("detector records disagree on held-out labels")
    first = {
        record.example_id: record
        for record in first_records
        if record.split == split and record.score is not None
    }
    second = {
        record.example_id: record
        for record in second_records
        if record.split == split and record.score is not None
    }
    common_ids = sorted(set(first) & set(second))
    if not common_ids:
        raise ValueError("detectors have no commonly scored held-out examples")
    labels = [first[example_id].hallucinated for example_id in common_ids]
    if any(second[example_id].hallucinated != label for example_id, label in zip(common_ids, labels)):
        raise ValueError("detector records disagree on held-out labels")
    first_scores = [float(first[example_id].score) for example_id in common_ids]
    second_scores = [float(second[example_id].score) for example_id in common_ids]
    first_metrics = binary_metrics(first_scores, labels, first_threshold)
    second_metrics = binary_metrics(second_scores, labels, second_threshold)

    first_predictions = [score >= first_threshold for score in first_scores]
    second_predictions = [score >= second_threshold for score in second_scores]
    first_only_correct = sum(
        a == label and b != label
        for a, b, label in zip(first_predictions, second_predictions, labels)
    )
    second_only_correct = sum(
        b == label and a != label
        for a, b, label in zip(first_predictions, second_predictions, labels)
    )
    p_value, p_method = _mcnemar_p_value(first_only_correct, second_only_correct)

    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(max(0, bootstrap_samples)):
        indices = [rng.randrange(len(common_ids)) for _ in common_ids]
        sampled_labels = [labels[index] for index in indices]
        if len(set(sampled_labels)) < 2:
            continue
        first_sample = binary_metrics(
            [first_scores[index] for index in indices], sampled_labels, first_threshold
        )
        second_sample = binary_metrics(
            [second_scores[index] for index in indices], sampled_labels, second_threshold
        )
        differences.append(first_sample.f1 - second_sample.f1)
    interval = (
        [_quantile(differences, 0.025), _quantile(differences, 0.975)]
        if differences
        else [math.nan, math.nan]
    )

    all_labels = [first_all[example_id].hallucinated for example_id in all_ids]
    first_all_predictions = [
        (float(first_all[example_id].score) >= first_threshold)
        if first_all[example_id].score is not None
        else not first_all[example_id].hallucinated
        for example_id in all_ids
    ]
    second_all_predictions = [
        (float(second_all[example_id].score) >= second_threshold)
        if second_all[example_id].score is not None
        else not second_all[example_id].hallucinated
        for example_id in all_ids
    ]
    first_failure_metrics = binary_metrics_from_predictions(
        first_all_predictions, all_labels
    )
    second_failure_metrics = binary_metrics_from_predictions(
        second_all_predictions, all_labels
    )
    first_failure_only_correct = sum(
        first_prediction == label and second_prediction != label
        for first_prediction, second_prediction, label in zip(
            first_all_predictions, second_all_predictions, all_labels
        )
    )
    second_failure_only_correct = sum(
        second_prediction == label and first_prediction != label
        for first_prediction, second_prediction, label in zip(
            first_all_predictions, second_all_predictions, all_labels
        )
    )
    failure_p_value, failure_p_method = _mcnemar_p_value(
        first_failure_only_correct, second_failure_only_correct
    )
    failure_differences: list[float] = []
    for _ in range(max(0, bootstrap_samples)):
        indices = [rng.randrange(len(all_ids)) for _ in all_ids]
        sampled_labels = [all_labels[index] for index in indices]
        if len(set(sampled_labels)) < 2:
            continue
        first_sample = binary_metrics_from_predictions(
            [first_all_predictions[index] for index in indices], sampled_labels
        )
        second_sample = binary_metrics_from_predictions(
            [second_all_predictions[index] for index in indices], sampled_labels
        )
        failure_differences.append(first_sample.f1 - second_sample.f1)
    failure_interval = (
        [_quantile(failure_differences, 0.025), _quantile(failure_differences, 0.975)]
        if failure_differences
        else [math.nan, math.nan]
    )
    return {
        "protocol": "paired-held-out-v1",
        "split": split,
        "common_successful_examples": len(common_ids),
        "total_paired_examples": len(all_ids),
        "first_coverage": len(first) / len(all_ids) if all_ids else 0.0,
        "second_coverage": len(second) / len(all_ids) if all_ids else 0.0,
        "first_detector": next(iter({record.detector for record in first_records})),
        "second_detector": next(iter({record.detector for record in second_records})),
        "first_f1": first_metrics.f1,
        "second_f1": second_metrics.f1,
        "f1_difference_first_minus_second": first_metrics.f1 - second_metrics.f1,
        "f1_difference_bootstrap_95_percent_ci": interval,
        "bootstrap_samples_completed": len(differences),
        "bootstrap_seed": seed,
        "mcnemar": {
            "first_only_correct": first_only_correct,
            "second_only_correct": second_only_correct,
            "p_value": p_value,
            "method": p_method,
        },
        "failure_as_incorrect": {
            "first_f1": first_failure_metrics.f1,
            "second_f1": second_failure_metrics.f1,
            "f1_difference_first_minus_second": (
                first_failure_metrics.f1 - second_failure_metrics.f1
            ),
            "f1_difference_bootstrap_95_percent_ci": failure_interval,
            "bootstrap_samples_completed": len(failure_differences),
            "mcnemar": {
                "first_only_correct": first_failure_only_correct,
                "second_only_correct": second_failure_only_correct,
                "p_value": failure_p_value,
                "method": failure_p_method,
            },
            "policy": "Every missing or failed score is assigned an incorrect decision.",
        },
        "warning": "Statistical significance does not establish novelty or production utility.",
    }
