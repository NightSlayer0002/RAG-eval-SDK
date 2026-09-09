"""Leakage-safe evaluation of SDK claim spans against RAGTruth annotations."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .schema import ScoreRecord
from .span_metrics import (
    Span,
    aggregate_character_span_metrics,
    character_span_metrics,
)


_RISK_STRATEGIES = {"claim_risk", "raw_pair_risk", "global_full_response"}


def _gold_spans(record: ScoreRecord) -> tuple[Span, ...]:
    values = record.metadata.get("gold_spans")
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"record {record.example_id} has no validated gold_spans metadata"
        )
    spans: list[Span] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(
                f"record {record.example_id} contains a malformed gold span"
            )
        spans.append(Span.from_mapping(value))
    return tuple(spans)


def _response_length(record: ScoreRecord) -> int:
    value = record.usage.get("response_characters_received")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"record {record.example_id} has no valid response character count"
        )
    return value


def _claim_values(record: ScoreRecord) -> tuple[Mapping[str, object], ...]:
    if record.usage.get("claim_localization_schema") != "claim-risk-spans-v1":
        raise ValueError(
            f"record {record.example_id} has no supported claim localization schema"
        )
    values = record.usage.get("claim_predictions")
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"record {record.example_id} has no claim_predictions sequence"
        )
    if not all(isinstance(value, Mapping) for value in values):
        raise ValueError(
            f"record {record.example_id} contains a malformed claim prediction"
        )
    return tuple(values)  # type: ignore[return-value]


def _predicted_spans(
    record: ScoreRecord,
    *,
    threshold: float | None,
    strategy: str = "claim_risk",
) -> tuple[Span, ...]:
    if strategy == "global_full_response":
        if threshold is None:
            raise ValueError("global_full_response requires a numeric threshold")
        features = record.usage.get("risk_features")
        risk = features.get("global_pair_risk") if isinstance(features, Mapping) else None
        if isinstance(risk, bool) or not isinstance(risk, (int, float)):
            raise ValueError(
                f"record {record.example_id} has no numeric global_pair_risk"
            )
        response_length = _response_length(record)
        return (
            (Span(0, response_length),)
            if float(risk) >= threshold and response_length > 0
            else ()
        )
    if strategy not in {"claim_risk", "raw_pair_risk", "decision"}:
        raise ValueError(f"unsupported localization strategy: {strategy}")
    predictions: list[Span] = []
    unsafe_decisions = {
        "contradicted",
        "insufficient_evidence",
        "escalate",
        "abstain",
    }
    for value in _claim_values(record):
        start = value.get("start")
        end = value.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"record {record.example_id} has a claim without integer offsets"
            )
        if strategy == "decision":
            if threshold is not None:
                raise ValueError("decision strategy does not accept a threshold")
            selected = str(value.get("decision")) in unsafe_decisions
        else:
            if threshold is None:
                raise ValueError(f"{strategy} requires a numeric threshold")
            if strategy == "claim_risk":
                risk = value.get("risk_score")
                if isinstance(risk, bool) or not isinstance(risk, (int, float)):
                    raise ValueError(
                        f"record {record.example_id} has a claim without numeric risk"
                    )
                risk = float(risk)
            else:
                consistency = value.get("pair_consistency")
                contradiction = value.get("pair_contradiction")
                if (
                    isinstance(consistency, bool)
                    or isinstance(contradiction, bool)
                    or not isinstance(consistency, (int, float))
                    or not isinstance(contradiction, (int, float))
                ):
                    raise ValueError(
                        f"record {record.example_id} has no raw pair probabilities"
                    )
                risk = max(1.0 - float(consistency), float(contradiction))
            selected = risk >= threshold
        if selected:
            predictions.append(Span(start, end))
    return tuple(predictions)


def _span_examples(
    records: Sequence[ScoreRecord],
    *,
    threshold: float | None,
    strategy: str = "claim_risk",
    failures_as_empty: bool = False,
) -> list[tuple[tuple[Span, ...], tuple[Span, ...], int]]:
    examples = []
    for record in records:
        if record.error is not None or record.score is None:
            if not failures_as_empty:
                continue
            predicted: tuple[Span, ...] = ()
        else:
            predicted = _predicted_spans(
                record, threshold=threshold, strategy=strategy
            )
        examples.append((predicted, _gold_spans(record), _response_length(record)))
    return examples


def select_claim_risk_threshold(
    records: Sequence[ScoreRecord],
    *,
    strategy: str = "claim_risk",
) -> dict[str, object]:
    """Select one scalar threshold using calibration labels only."""

    if strategy not in _RISK_STRATEGIES:
        raise ValueError(f"unsupported risk strategy: {strategy}")

    successful = [
        record for record in records if record.error is None and record.score is not None
    ]
    if not successful:
        raise ValueError("claim threshold selection requires successful records")
    risks: set[float] = set()
    for record in successful:
        if strategy == "global_full_response":
            features = record.usage.get("risk_features")
            risk = features.get("global_pair_risk") if isinstance(features, Mapping) else None
            if isinstance(risk, bool) or not isinstance(risk, (int, float)):
                raise ValueError(
                    f"record {record.example_id} has no numeric global_pair_risk"
                )
            risks.add(float(risk))
            continue
        for value in _claim_values(record):
            if strategy == "claim_risk":
                risk = value.get("risk_score")
                if isinstance(risk, bool) or not isinstance(risk, (int, float)):
                    raise ValueError(
                        f"record {record.example_id} has a claim without numeric risk"
                    )
                risks.add(float(risk))
            else:
                consistency = value.get("pair_consistency")
                contradiction = value.get("pair_contradiction")
                if (
                    isinstance(consistency, bool)
                    or isinstance(contradiction, bool)
                    or not isinstance(consistency, (int, float))
                    or not isinstance(contradiction, (int, float))
                ):
                    raise ValueError(
                        f"record {record.example_id} has no raw pair probabilities"
                    )
                risks.add(max(1.0 - float(consistency), float(contradiction)))
    if not risks:
        raise ValueError("claim threshold selection requires at least one claim risk")
    if min(risks) < 0.0 or max(risks) > 1.0 or not all(
        math.isfinite(risk) for risk in risks
    ):
        raise ValueError("claim risks must be finite values within [0, 1]")
    # The upper sentinel represents an explicit predict-no-span policy.
    candidates = sorted({0.0, *risks, math.nextafter(1.0, math.inf)})
    best: tuple[tuple[float, float, float], float, dict[str, object]] | None = None
    for threshold in candidates:
        metrics = aggregate_character_span_metrics(
            _span_examples(
                successful, threshold=threshold, strategy=strategy
            )
        )
        micro = metrics["micro"]
        assert isinstance(micro, dict)
        rank = (
            float(micro["f1"]),
            float(micro["precision"]),
            threshold,
        )
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)
    assert best is not None
    return {
        "threshold": best[1],
        "strategy": strategy,
        "objective": "micro_character_f1",
        "tie_break": "higher_precision_then_higher_threshold",
        "candidate_thresholds": len(candidates),
        "successful_examples": len(successful),
        "metrics": best[2],
    }


def _f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_span_bootstrap(
    records: Sequence[ScoreRecord],
    *,
    first_strategy: str,
    first_threshold: float | None,
    second_strategy: str,
    second_threshold: float | None,
    samples: int = 2000,
    seed: int = 1729,
) -> dict[str, object]:
    """Source-group bootstrap for a paired micro-character-F1 difference."""

    if samples < 1:
        raise ValueError("samples must be positive")
    successful = [
        record for record in records if record.error is None and record.score is not None
    ]
    if not successful:
        raise ValueError("paired span bootstrap requires successful records")
    by_group: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = defaultdict(list)
    for record in successful:
        gold = _gold_spans(record)
        text_length = _response_length(record)
        first = character_span_metrics(
            _predicted_spans(
                record, threshold=first_threshold, strategy=first_strategy
            ),
            gold,
            text_length=text_length,
        )
        second = character_span_metrics(
            _predicted_spans(
                record, threshold=second_threshold, strategy=second_strategy
            ),
            gold,
            text_length=text_length,
        )
        by_group[record.group_id].append(
            (
                (
                    first.true_positive_characters,
                    first.false_positive_characters,
                    first.false_negative_characters,
                ),
                (
                    second.true_positive_characters,
                    second.false_positive_characters,
                    second.false_negative_characters,
                ),
            )
        )

    def difference(groups: Sequence[str]) -> float:
        first_counts = [0, 0, 0]
        second_counts = [0, 0, 0]
        for group in groups:
            for first, second in by_group[group]:
                for index in range(3):
                    first_counts[index] += first[index]
                    second_counts[index] += second[index]
        return _f1_from_counts(*first_counts) - _f1_from_counts(*second_counts)

    groups = sorted(by_group)
    observed = difference(groups)
    generator = random.Random(seed)
    bootstrap_deltas = [
        difference([generator.choice(groups) for _ in groups])
        for _ in range(samples)
    ]
    lower = _percentile(bootstrap_deltas, 0.025)
    upper = _percentile(bootstrap_deltas, 0.975)
    return {
        "first_strategy": first_strategy,
        "second_strategy": second_strategy,
        "delta_micro_character_f1": observed,
        "confidence_interval_95": [lower, upper],
        "bootstrap_samples": samples,
        "bootstrap_unit": "source_group",
        "source_groups": len(groups),
        "seed": seed,
        "conclusion": (
            "first_better"
            if lower > 0.0
            else "first_worse"
            if upper < 0.0
            else "inconclusive"
        ),
    }


def summarize_span_localization(
    records: Sequence[ScoreRecord],
    *,
    calibration_split: str = "train",
    test_split: str = "test",
    fixed_threshold: float | None = None,
    bootstrap_samples: int = 2000,
) -> dict[str, object]:
    """Calibrate on one split and report untouched held-out span metrics."""

    if not records:
        raise ValueError("no score records were supplied")
    detector_names = {record.detector for record in records}
    if len(detector_names) != 1:
        raise ValueError("span evaluation accepts records from one detector")
    calibration = [record for record in records if record.split == calibration_split]
    held_out = [record for record in records if record.split == test_split]
    if not calibration or not held_out:
        raise ValueError("calibration and test splits must both be non-empty")
    overlap = {record.group_id for record in calibration} & {
        record.group_id for record in held_out
    }
    if overlap:
        raise ValueError(
            f"group leakage: {len(overlap)} source groups occur in both calibration and test"
        )
    if fixed_threshold is None:
        threshold_report = select_claim_risk_threshold(
            calibration, strategy="claim_risk"
        )
        threshold = float(threshold_report["threshold"])
    else:
        if not math.isfinite(fixed_threshold) or not 0.0 <= fixed_threshold <= math.nextafter(1.0, math.inf):
            raise ValueError("fixed_threshold must be within the supported risk range")
        threshold = fixed_threshold
        threshold_report = {
            "threshold": threshold,
            "objective": "fixed_claim_risk_threshold",
            "successful_examples": sum(
                record.error is None and record.score is not None
                for record in calibration
            ),
            "metrics": aggregate_character_span_metrics(
                _span_examples(
                    calibration, threshold=threshold, strategy="claim_risk"
                )
            ),
        }

    successful_test = [
        record for record in held_out if record.error is None and record.score is not None
    ]
    if not successful_test:
        raise ValueError("test split has no successful localization records")
    primary = aggregate_character_span_metrics(
        _span_examples(
            successful_test, threshold=threshold, strategy="claim_risk"
        )
    )
    failure_sensitivity = aggregate_character_span_metrics(
        _span_examples(
            held_out,
            threshold=threshold,
            strategy="claim_risk",
            failures_as_empty=True,
        )
    )
    decision_policy = aggregate_character_span_metrics(
        _span_examples(
            successful_test, threshold=None, strategy="decision"
        )
    )
    raw_threshold_report = select_claim_risk_threshold(
        calibration, strategy="raw_pair_risk"
    )
    raw_threshold = float(raw_threshold_report["threshold"])
    raw_pair_baseline = aggregate_character_span_metrics(
        _span_examples(
            successful_test,
            threshold=raw_threshold,
            strategy="raw_pair_risk",
        )
    )
    global_threshold_report = select_claim_risk_threshold(
        calibration, strategy="global_full_response"
    )
    global_threshold = float(global_threshold_report["threshold"])
    global_baseline = aggregate_character_span_metrics(
        _span_examples(
            successful_test,
            threshold=global_threshold,
            strategy="global_full_response",
        )
    )
    paired_comparisons = [
        paired_span_bootstrap(
            successful_test,
            first_strategy="claim_risk",
            first_threshold=threshold,
            second_strategy="raw_pair_risk",
            second_threshold=raw_threshold,
            samples=bootstrap_samples,
        ),
        paired_span_bootstrap(
            successful_test,
            first_strategy="claim_risk",
            first_threshold=threshold,
            second_strategy="global_full_response",
            second_threshold=global_threshold,
            samples=bootstrap_samples,
        ),
        paired_span_bootstrap(
            successful_test,
            first_strategy="claim_risk",
            first_threshold=threshold,
            second_strategy="decision",
            second_threshold=None,
            samples=bootstrap_samples,
        ),
    ]
    return {
        "protocol": "ragtruth-character-localization-v1",
        "detector": next(iter(detector_names)),
        "positive_unit": "union of half-open response character spans",
        "calibration": {
            "split": calibration_split,
            **threshold_report,
        },
        "held_out": {
            "split": test_split,
            "successful_examples": len(successful_test),
            "total_examples": len(held_out),
            "coverage": len(successful_test) / len(held_out),
            "claim_risk_threshold": primary,
            "decision_policy_without_threshold": decision_policy,
            "failure_as_empty_prediction": failure_sensitivity,
        },
        "baselines": {
            "raw_pair_risk": {
                "calibration": raw_threshold_report,
                "held_out": raw_pair_baseline,
            },
            "global_pair_full_response_projection": {
                "calibration": global_threshold_report,
                "held_out": global_baseline,
            },
        },
        "paired_comparisons": paired_comparisons,
        "metric_note": (
            "Micro character F1 is primary. Macro includes exact all-negative "
            "examples and is secondary. Failed detector calls are excluded from the "
            "primary metric and treated as empty predictions in the sensitivity result."
        ),
    }


def read_score_records(path: str | Path) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(ScoreRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid score record at {source}:{line_number}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved SDK claim spans against RAGTruth gold annotations"
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    summary = summarize_span_localization(
        read_score_records(args.scores),
        calibration_split=args.calibration_split,
        test_split=args.test_split,
        bootstrap_samples=args.bootstrap_samples,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Span summary saved: {destination}")


if __name__ == "__main__":
    main()
