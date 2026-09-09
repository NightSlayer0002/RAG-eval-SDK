"""Checkpointed, held-out benchmark execution and reporting."""

from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from rag_eval_sdk.calibration import (
    CalibrationResult,
    binary_metrics,
    binary_metrics_from_predictions,
    ranking_metrics,
    select_threshold,
)
from rag_eval_sdk.risk_model import MREG_COMPACT_FEATURE_NAMES, fit_risk_calibrator

from .detectors import Detector
from .schema import BenchmarkExample, ScoreRecord


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:120]


def _read_records(path: Path) -> list[ScoreRecord]:
    records: dict[str, ScoreRecord] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = ScoreRecord(**json.loads(line))
                records[record.example_id] = record
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid checkpoint at {path}:{line_number}") from exc
    return list(records.values())


def score_examples(
    examples: Sequence[BenchmarkExample],
    detector: Detector,
    output_directory: str | Path,
    *,
    resume: bool = True,
    progress_every: int = 25,
    retry_failures: bool = True,
) -> tuple[list[ScoreRecord], Path]:
    """Score sequentially and durably append every success or failure."""

    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"scores_{_safe_name(detector.name)}.jsonl"
    existing = _read_records(path) if resume else []
    completed = {
        record.example_id
        for record in existing
        if record.error is None or not retry_failures
    }
    mode = "a" if resume else "w"
    records_by_id = {record.example_id: record for record in existing}
    with path.open(mode, encoding="utf-8") as handle:
        pending_total = sum(example.id not in completed for example in examples)
        processed = 0
        for example in examples:
            if example.id in completed:
                continue
            record = detector.score(example)
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            records_by_id[record.example_id] = record
            processed += 1
            if progress_every > 0 and (
                processed % progress_every == 0 or processed == pending_total
            ):
                failures = sum(item.error is not None for item in records_by_id.values())
                print(
                    f"[{detector.name}] {processed}/{pending_total} new examples; "
                    f"{failures} total failures"
                )
    return list(records_by_id.values()), path


def apply_train_only_risk_calibration(
    records: Sequence[ScoreRecord],
    *,
    fit_split: str = "fit",
    l2: float = 0.10,
    feature_names: Sequence[str] = MREG_COMPACT_FEATURE_NAMES,
    oof_folds: int = 5,
    objective: str = "f1",
) -> tuple[list[ScoreRecord], dict[str, object]]:
    """Learn a multivariate SDK risk score from a non-test fitting split."""

    fitting = [
        record
        for record in records
        if record.split == fit_split
        and record.score is not None
        and isinstance(record.usage.get("risk_features"), dict)
    ]
    if not fitting:
        raise ValueError("no successful fitting records contain risk features")
    model = fit_risk_calibrator(
        [record.usage["risk_features"] for record in fitting],  # type: ignore[list-item]
        [record.hallucinated for record in fitting],
        l2=l2,
        feature_names=feature_names,
    )
    groups = sorted(
        {record.group_id for record in fitting},
        key=lambda group: hashlib.sha256(
            f"mreg-oof-v1:{group}".encode("utf-8")
        ).digest(),
    )
    fold_count = min(max(2, oof_folds), len(groups))
    oof_scores: list[float] = []
    oof_labels: list[bool] = []
    oof_valid = len(groups) >= 4
    if oof_valid:
        group_fold = {group: index % fold_count for index, group in enumerate(groups)}
        for fold in range(fold_count):
            fold_train = [
                record
                for record in fitting
                if group_fold[record.group_id] != fold
            ]
            fold_validation = [
                record
                for record in fitting
                if group_fold[record.group_id] == fold
            ]
            if (
                not fold_validation
                or len({record.hallucinated for record in fold_train}) < 2
            ):
                oof_valid = False
                break
            fold_model = fit_risk_calibrator(
                [record.usage["risk_features"] for record in fold_train],  # type: ignore[list-item]
                [record.hallucinated for record in fold_train],
                l2=l2,
                feature_names=feature_names,
            )
            oof_scores.extend(
                fold_model.predict_proba(record.usage["risk_features"])  # type: ignore[arg-type]
                for record in fold_validation
            )
            oof_labels.extend(record.hallucinated for record in fold_validation)
    if oof_valid:
        threshold_selection = select_threshold(
            oof_scores,
            oof_labels,
            objective=objective,  # type: ignore[arg-type]
        ).to_dict()
        threshold_selection.update(
            {
                "strategy": "grouped-out-of-fold",
                "folds": fold_count,
                "grouping": "group_id",
            }
        )
    else:
        threshold_selection = {
            "strategy": "fixed-probability-fallback",
            "threshold": 0.5,
            "folds": 0,
            "reason": "too few groups or a one-class training fold",
        }
    detector_name = f"{records[0].detector}:learned-risk:{model.calibration_id}"
    calibrated: list[ScoreRecord] = []
    for record in records:
        features = record.usage.get("risk_features")
        if record.score is not None and not isinstance(features, dict):
            raise ValueError(
                f"successful record {record.example_id} has no risk feature mapping"
            )
        score = model.predict_proba(features) if isinstance(features, dict) else None
        calibrated.append(
            replace(
                record,
                detector=detector_name,
                score=score,
                usage={
                    **record.usage,
                    "raw_grounding_risk": record.score,
                    "risk_calibration_id": model.calibration_id,
                },
            )
        )
    manifest = {
        **model.to_dict(),
        "fit_split": fit_split,
        "fit_successful_examples": len(fitting),
        "fit_total_examples": sum(record.split == fit_split for record in records),
        "fit_groups": len({record.group_id for record in fitting}),
        "decision_threshold": float(threshold_selection["threshold"]),
        "threshold_selection": threshold_selection,
    }
    return calibrated, manifest


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _slice_summary(
    records: Sequence[ScoreRecord], threshold: float
) -> dict[str, object]:
    successful = [record for record in records if record.score is not None]
    result: dict[str, object] = {
        "total_examples": len(records),
        "successful_examples": len(successful),
        "coverage": len(successful) / len(records) if records else 0.0,
    }
    if successful:
        labels = [record.hallucinated for record in successful]
        result["successful_only"] = asdict(
            binary_metrics(
                [float(record.score) for record in successful], labels, threshold
            )
        )
        if len(set(labels)) == 2:
            result["ranking"] = asdict(
                ranking_metrics(
                    [float(record.score) for record in successful], labels
                )
            )
    result["failure_as_incorrect"] = asdict(
        binary_metrics_from_predictions(
            [
                (float(record.score) >= threshold)
                if record.score is not None
                else not record.hallucinated
                for record in records
            ],
            [record.hallucinated for record in records],
        )
    )
    return result


def _metadata_slices(
    records: Sequence[ScoreRecord], field: str, threshold: float
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[ScoreRecord]] = {}
    for record in records:
        value = record.metadata.get(field)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(record)
    return {
        value: _slice_summary(group_records, threshold)
        for value, group_records in sorted(groups.items())
    }


def _macro_slice_metrics(
    slices: dict[str, dict[str, object]],
) -> dict[str, object]:
    metric_rows = [
        row["successful_only"]
        for row in slices.values()
        if isinstance(row.get("successful_only"), dict)
    ]
    fields = ("f1", "precision", "recall", "balanced_accuracy")
    if not metric_rows:
        return {
            "groups": 0,
            **{field: None for field in fields},
            "weighting": "equal weight per non-empty slice",
        }
    return {
        "groups": len(metric_rows),
        **{
            field: statistics.fmean(float(row[field]) for row in metric_rows)
            for field in fields
        },
        "weighting": "equal weight per non-empty slice",
    }


def summarize_detector(
    records: Sequence[ScoreRecord],
    *,
    validation_split: str = "train",
    test_split: str = "test",
    objective: str = "f1",
    fixed_threshold: float | None = None,
    fixed_threshold_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("no score records were supplied")
    detector_names = {record.detector for record in records}
    if len(detector_names) != 1:
        raise ValueError("summarize_detector accepts records from one detector")

    validation_all = [record for record in records if record.split == validation_split]
    test_all = [record for record in records if record.split == test_split]
    validation_groups = {record.group_id for record in validation_all}
    test_groups = {record.group_id for record in test_all}
    overlap = validation_groups & test_groups
    if overlap:
        raise ValueError(
            f"group leakage: {len(overlap)} source groups occur in both validation and test"
        )
    validation = [record for record in validation_all if record.score is not None]
    test = [record for record in test_all if record.score is not None]
    if not validation or not test:
        raise ValueError("both validation and test need at least one successful score")

    validation_scores = [float(record.score) for record in validation]
    validation_labels = [record.hallucinated for record in validation]
    if fixed_threshold is None:
        calibration = select_threshold(
            validation_scores,
            validation_labels,
            objective=objective,  # type: ignore[arg-type]
        )
    else:
        if not 0.0 <= fixed_threshold <= 1.0:
            raise ValueError("fixed_threshold must be within [0, 1]")
        fixed_metrics = binary_metrics(
            validation_scores, validation_labels, fixed_threshold
        )
        calibration = CalibrationResult(
            threshold=fixed_threshold,
            positive_when="higher",
            objective="fixed_probability_threshold",
            objective_value=fixed_metrics.f1,
            metrics=fixed_metrics,
            examples=len(validation_labels),
            positives=sum(validation_labels),
            calibration_id=f"fixed-probability-threshold-v1:{fixed_threshold:g}",
        )
    held_out = binary_metrics(
        [float(record.score) for record in test],
        [record.hallucinated for record in test],
        calibration.threshold,
    )
    held_out_failure_as_incorrect = binary_metrics_from_predictions(
        [
            (float(record.score) >= calibration.threshold)
            if record.score is not None
            else not record.hallucinated
            for record in test_all
        ],
        [record.hallucinated for record in test_all],
    )
    ranking = ranking_metrics(
        [float(record.score) for record in test],
        [record.hallucinated for record in test],
    )
    errors = [record for record in records if record.error]
    error_types = Counter(str(record.error).split(":", 1)[0] for record in errors)
    resource_records = test_all
    latencies = [record.elapsed_ms for record in resource_records]
    remote_calls = sum(
        int(record.usage.get("remote_calls", 0)) for record in resource_records
    )
    verifier_batches = sum(
        int(record.usage.get("verifier_batches", 0)) for record in resource_records
    )
    estimated_tokens = sum(
        int(record.usage.get("estimated_input_tokens", 0))
        for record in resource_records
    )
    similarity_bytes = [
        int(record.usage.get("similarity_matrix_bytes", 0))
        for record in resource_records
    ]
    truncated_chunks = sum(
        int(record.usage.get("truncated_chunks", 0)) for record in resource_records
    )
    rss_after = [
        int(record.usage.get("process_rss_after_bytes", 0))
        for record in resource_records
    ]
    rss_growth = [
        max(
            0,
            int(record.usage.get("process_rss_after_bytes", 0))
            - int(record.usage.get("process_rss_before_bytes", 0)),
        )
        for record in resource_records
    ]
    task_slices = _metadata_slices(test_all, "task_type", calibration.threshold)
    source_model_slices = _metadata_slices(
        test_all, "model", calibration.threshold
    )
    calibration_report = calibration.to_dict()
    if fixed_threshold_metadata is not None:
        calibration_report["selection"] = fixed_threshold_metadata
    return {
        "protocol": "held-out-threshold-v1",
        "detector": next(iter(detector_names)),
        "calibration": calibration_report,
        "held_out": {
            **asdict(held_out),
            **asdict(ranking),
            "split": test_split,
            "successful_examples": len(test),
            "total_examples": len(test_all),
            "coverage": len(test) / len(test_all) if test_all else 0.0,
            "failure_as_incorrect": asdict(held_out_failure_as_incorrect),
        },
        "validation": {
            "split": validation_split,
            "successful_examples": len(validation),
            "total_examples": len(validation_all),
            "coverage": len(validation) / len(validation_all) if validation_all else 0.0,
        },
        "slices": {
            "task_type": task_slices,
            "source_model": source_model_slices,
            "macro_task": _macro_slice_metrics(task_slices),
            "macro_source_model": _macro_slice_metrics(source_model_slices),
        },
        "failures": {
            "count": len(errors),
            "rate": len(errors) / len(records),
            "by_type": dict(error_types),
            "policy": (
                "Primary score metrics are successful-only. held_out.failure_as_incorrect "
                "assigns every failed example an incorrect decision as a sensitivity analysis."
            ),
        },
        "resources": {
            "scope": test_split,
            "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "remote_calls": remote_calls,
            "verifier_batches": verifier_batches,
            "claims": sum(
                int(record.usage.get("claims", 0)) for record in resource_records
            ),
            "evidence_pairs_scored": sum(
                int(record.usage.get("evidence_pairs_scored", 0))
                for record in resource_records
            ),
            "audit_probes": sum(
                int(record.usage.get("audit_probes", 0))
                for record in resource_records
            ),
            "estimated_input_tokens": estimated_tokens,
            "maximum_similarity_matrix_bytes": max(similarity_bytes, default=0),
            "truncated_or_dropped_chunks": truncated_chunks,
            "truncated_claims": sum(
                int(record.usage.get("truncated_claims", 0))
                for record in resource_records
            ),
            "truncated_response_characters": sum(
                max(
                    0,
                    int(record.usage.get("response_characters_received", 0))
                    - int(record.usage.get("response_characters_used", 0)),
                )
                for record in resource_records
            ),
            "maximum_observed_process_rss_bytes": max(rss_after, default=0),
            "maximum_single_example_rss_growth_bytes": max(rss_growth, default=0),
            "memory_note": "RSS is process-wide and includes retained model/cache memory; it is not a per-call peak allocator trace.",
        },
    }


def save_summary(summary: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
