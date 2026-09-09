"""Create descriptive, non-tuning slices for a completed Step 2 span run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .schema import ScoreRecord
from .span_evaluation import (
    _gold_spans,
    _predicted_spans,
    _response_length,
    _span_examples,
    read_score_records,
)
from .span_metrics import aggregate_character_span_metrics, character_span_metrics


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _strategy_metrics(
    records: Sequence[ScoreRecord],
    *,
    strategy: str,
    threshold: float | None,
) -> dict[str, object]:
    return aggregate_character_span_metrics(
        _span_examples(records, threshold=threshold, strategy=strategy)
    )


def _gold_attribute_recall(
    records: Sequence[ScoreRecord],
    *,
    strategy: str,
    threshold: float | None,
    include: Callable[[Mapping[str, object]], bool],
) -> dict[str, object]:
    true_positive = 0
    false_negative = 0
    annotations = 0
    responses = 0
    for record in records:
        values = record.metadata.get("gold_spans")
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Missing gold spans: {record.example_id}")
        selected = [value for value in values if isinstance(value, Mapping) and include(value)]
        if not selected:
            continue
        responses += 1
        annotations += len(selected)
        # Re-use the strict Span parser after substituting only the requested
        # annotation subset into an otherwise unchanged immutable record.
        subset = ScoreRecord(
            detector=record.detector,
            example_id=record.example_id,
            group_id=record.group_id,
            split=record.split,
            hallucinated=record.hallucinated,
            score=record.score,
            elapsed_ms=record.elapsed_ms,
            error=record.error,
            usage=record.usage,
            metadata={**record.metadata, "gold_spans": selected},
        )
        metrics = character_span_metrics(
            _predicted_spans(record, threshold=threshold, strategy=strategy),
            _gold_spans(subset),
            text_length=_response_length(record),
        )
        true_positive += metrics.true_positive_characters
        false_negative += metrics.false_negative_characters
    denominator = true_positive + false_negative
    return {
        "annotations": annotations,
        "responses": responses,
        "gold_characters": denominator,
        "overlapped_characters": true_positive,
        "character_recall": true_positive / denominator if denominator else None,
        "note": (
            "Descriptive recall only. Prediction precision is not attributed to a "
            "gold subtype, and overlapping labels are unioned within each response/subtype."
        ),
    }


def analyze(records: Sequence[ScoreRecord], frozen: Mapping[str, Any]) -> dict[str, Any]:
    test = [
        record
        for record in records
        if record.split == "test" and record.error is None and record.score is not None
    ]
    if not test:
        raise ValueError("No successful test records")
    thresholds: dict[str, float | None] = {
        "claim_risk": float(frozen["calibration"]["threshold"]),
        "raw_pair_risk": float(
            frozen["baselines"]["raw_pair_risk"]["calibration"]["threshold"]
        ),
        "global_full_response": float(
            frozen["baselines"]["global_pair_full_response_projection"]["calibration"]["threshold"]
        ),
        "decision": None,
    }

    by_task: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in test:
        by_task[str(record.metadata.get("task_type", "unknown"))].append(record)
    task_slices = {
        task: {
            "examples": len(task_records),
            "strategies": {
                strategy: _strategy_metrics(
                    task_records,
                    strategy=strategy,
                    threshold=threshold,
                )
                for strategy, threshold in thresholds.items()
            },
        }
        for task, task_records in sorted(by_task.items())
    }

    label_types = sorted(
        {
            str(value.get("label_type", "unknown"))
            for record in test
            for value in record.metadata.get("gold_spans", ())
            if isinstance(value, Mapping)
        }
    )
    annotation_slices = {
        strategy: {
            "label_type": {
                label_type: _gold_attribute_recall(
                    test,
                    strategy=strategy,
                    threshold=threshold,
                    include=lambda value, expected=label_type: str(
                        value.get("label_type", "unknown")
                    )
                    == expected,
                )
                for label_type in label_types
            },
            "implicit_true": _gold_attribute_recall(
                test,
                strategy=strategy,
                threshold=threshold,
                include=lambda value: bool(value.get("implicit_true", False)),
            ),
            "due_to_null": _gold_attribute_recall(
                test,
                strategy=strategy,
                threshold=threshold,
                include=lambda value: bool(value.get("due_to_null", False)),
            ),
        }
        for strategy, threshold in thresholds.items()
    }

    gold_lengths = [
        int(value["end"]) - int(value["start"])
        for record in test
        for value in record.metadata.get("gold_spans", ())
        if isinstance(value, Mapping)
    ]
    claim_lengths = [
        int(value["end"]) - int(value["start"])
        for record in test
        for value in record.usage["claim_predictions"]
    ]
    primary_micro = frozen["held_out"]["claim_risk_threshold"]["micro"]
    raw_micro = frozen["baselines"]["raw_pair_risk"]["held_out"]["micro"]
    return {
        "protocol": "ragtruth-step2-descriptive-slices-v1",
        "status": "post-hoc descriptive only; no threshold or model selection",
        "examples": len(test),
        "thresholds_copied_from_frozen_summary": thresholds,
        "task_slices": task_slices,
        "annotation_recall_slices": annotation_slices,
        "boundary_scale": {
            "gold_annotations": len(gold_lengths),
            "gold_annotation_length_mean": statistics.fmean(gold_lengths),
            "gold_annotation_length_median": statistics.median(gold_lengths),
            "predicted_claim_units": len(claim_lengths),
            "claim_unit_length_mean": statistics.fmean(claim_lengths),
            "claim_unit_length_median": statistics.median(claim_lengths),
        },
        "claim_risk_minus_raw_pair_character_counts": {
            "true_positive": int(primary_micro["true_positive_characters"])
            - int(raw_micro["true_positive_characters"]),
            "false_positive": int(primary_micro["false_positive_characters"])
            - int(raw_micro["false_positive_characters"]),
            "false_negative": int(primary_micro["false_negative_characters"])
            - int(raw_micro["false_negative_characters"]),
            "predicted": int(primary_micro["predicted_characters"])
            - int(raw_micro["predicted_characters"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--span-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(read_score_records(args.scores), _read_json(args.span_summary))
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Exploratory slices saved: {args.output}")


if __name__ == "__main__":
    main()
