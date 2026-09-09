"""Replay and integrity-audit a completed RAGTruth benchmark without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag_eval_sdk.risk_model import MREG_COMPACT_FEATURE_NAMES, RISK_FEATURE_NAMES

from .comparison import compare_detectors
from .datasets import load_ragtruth
from .ragtruth_benchmark import (
    _partition_fit_split,
    _safe_file_name,
    _sha256_file,
    _stable_limit,
)
from .runner import (
    _safe_name,
    apply_train_only_risk_calibration,
    summarize_detector,
)
from .schema import BenchmarkExample, ScoreRecord


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_unique_records(path: Path) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = ScoreRecord(**json.loads(line))
            if record.example_id in ids:
                raise ValueError(f"duplicate example ID at {path}:{line_number}")
            ids.add(record.example_id)
            records.append(record)
    return records


def _first_difference(actual: Any, expected: Any, path: str = "root") -> str:
    if type(actual) is not type(expected):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return f"{path}: keys {set(actual) ^ set(expected)} differ"
        for key in sorted(actual):
            if actual[key] != expected[key]:
                return _first_difference(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}: lengths {len(actual)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            if left != right:
                return _first_difference(left, right, f"{path}[{index}]")
    return f"{path}: {actual!r} != {expected!r}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_equal(actual: Any, expected: Any, description: str) -> None:
    normalized_actual = json.loads(json.dumps(_jsonable(actual), sort_keys=True))
    normalized_expected = json.loads(json.dumps(_jsonable(expected), sort_keys=True))
    if normalized_actual != normalized_expected:
        raise ValueError(
            f"replay mismatch: {description}; "
            f"{_first_difference(normalized_actual, normalized_expected)}"
        )


def _verify_records(
    records: list[ScoreRecord], expected: list[BenchmarkExample], detector: str
) -> None:
    expected_by_id = {example.id: example for example in expected}
    _require_equal(
        {record.example_id for record in records},
        set(expected_by_id),
        f"{detector} example IDs",
    )
    for record in records:
        example = expected_by_id[record.example_id]
        _require_equal(record.group_id, example.group_id, f"{record.example_id} group")
        _require_equal(record.split, example.split, f"{record.example_id} split")
        _require_equal(
            record.hallucinated,
            example.hallucinated,
            f"{record.example_id} label",
        )
        if record.score is not None and not 0.0 <= float(record.score) <= 1.0:
            raise ValueError(f"out-of-range score for {record.example_id}")


def _source_scan(project_root: Path, selected_ids: set[str], metric_values: set[str]) -> None:
    source_paths = list((project_root / "rag_eval_sdk").glob("*.py"))
    source_paths.extend((project_root / "benchmarks").glob("*.py"))
    source_paths = [path for path in source_paths if path.name != Path(__file__).name]
    prediction_paths = set((project_root / "rag_eval_sdk").glob("*.py"))
    prediction_paths.add(project_root / "benchmarks" / "detectors.py")
    forbidden_paths = (
        "benchmarks/results",
        "ragtruth_mreg_hhem_ragas_fresh100",
    )
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        normalized = text.replace("\\", "/").lower()
        if path in prediction_paths and any(
            value in normalized for value in forbidden_paths
        ):
            raise ValueError(f"prediction source references benchmark results: {path}")
        if any(example_id in text for example_id in selected_ids):
            raise ValueError(f"prediction source special-cases a selected ID: {path}")
        if any(value in text for value in metric_values):
            raise ValueError(f"prediction source contains a frozen metric value: {path}")


def audit(output_dir: Path, data_dir: Path) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    arguments = manifest["arguments"]
    checksums = {
        name: _sha256_file(data_dir / name)
        for name in ("source_info.jsonl", "response.jsonl")
    }
    _require_equal(checksums, manifest["dataset_checksums"], "dataset checksums")

    selected = _stable_limit(
        list(load_ragtruth(data_dir)),
        arguments["validation_limit"],
        arguments["test_limit"],
        arguments["validation_offset"],
        arguments["test_offset"],
    )
    if arguments["risk_calibrator"] == "logistic":
        selected = _partition_fit_split(selected, arguments["risk_fit_fraction"])
    selected_digest = hashlib.sha256(
        "\n".join(example.id for example in selected).encode("utf-8")
    ).hexdigest()
    _require_equal(selected_digest, manifest["example_ids_sha256"], "selected ID digest")

    records_by_detector: dict[str, list[ScoreRecord]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    learned_records: list[ScoreRecord] | None = None
    learned_summary: dict[str, Any] | None = None

    for detector in manifest["detectors"]:
        score_path = output_dir / f"scores_{_safe_name(detector)}.jsonl"
        records = _read_unique_records(score_path)
        expected = (
            selected
            if detector.startswith("rag-eval-sdk-")
            else [example for example in selected if example.split != "fit"]
        )
        _verify_records(records, expected, detector)
        records_by_detector[detector] = records
        stem = score_path.stem.removeprefix("scores_")

        if detector.startswith("rag-eval-sdk-") and arguments["risk_calibrator"] == "logistic":
            raw_summary = summarize_detector(records, objective=arguments["objective"])
            _require_equal(
                raw_summary,
                _read_json(output_dir / f"summary_{stem}_raw.json"),
                "raw SDK summary",
            )
            learned_records, risk_manifest = apply_train_only_risk_calibration(
                records,
                l2=arguments["risk_l2"],
                feature_names=(
                    MREG_COMPACT_FEATURE_NAMES
                    if arguments["risk_features"] == "compact"
                    else RISK_FEATURE_NAMES
                ),
                oof_folds=arguments["risk_oof_folds"],
                objective=arguments["objective"],
            )
            _require_equal(
                risk_manifest,
                _read_json(output_dir / f"risk_model_{stem}.json"),
                "learned risk model",
            )
            learned_summary = summarize_detector(
                learned_records,
                objective=arguments["objective"],
                fixed_threshold=float(risk_manifest["decision_threshold"]),
                fixed_threshold_metadata=risk_manifest["threshold_selection"],
            )
            _require_equal(
                learned_summary,
                _read_json(output_dir / f"summary_{stem}_learned.json"),
                "learned SDK summary",
            )
            summaries["sdk_raw"] = raw_summary
            summaries["sdk_learned"] = learned_summary
        else:
            summary = summarize_detector(records, objective=arguments["objective"])
            _require_equal(
                summary,
                _read_json(output_dir / f"summary_{stem}.json"),
                f"{detector} summary",
            )
            summaries[detector] = summary

    if learned_records is None or learned_summary is None:
        raise ValueError("completed run has no learned SDK records")
    paired_verified = 0
    for detector in manifest["detectors"]:
        second_records = records_by_detector[detector]
        second_summary = (
            summaries["sdk_raw"]
            if detector.startswith("rag-eval-sdk-")
            else summaries[detector]
        )
        comparison = compare_detectors(
            learned_records,
            second_records,
            first_threshold=float(learned_summary["calibration"]["threshold"]),
            second_threshold=float(second_summary["calibration"]["threshold"]),
            bootstrap_samples=arguments["bootstrap_samples"],
        )
        comparison_path = output_dir / (
            f"paired_{_safe_file_name(str(learned_summary['detector']))}_vs_"
            f"{_safe_file_name(str(second_summary['detector']))}.json"
        )
        _require_equal(comparison, _read_json(comparison_path), comparison_path.name)
        paired_verified += 1

    metric_values = {
        value
        for summary in summaries.values()
        for field in ("f1", "auroc", "average_precision")
        for value in (repr(float(summary["held_out"][field])),)
        if len(value) >= 10
    }
    _source_scan(Path(__file__).resolve().parents[1], {example.id for example in selected}, metric_values)

    test_sets = [
        {
            record.example_id
            for record in records
            if record.split == "test"
        }
        for records in records_by_detector.values()
    ]
    if not test_sets or any(ids != test_sets[0] for ids in test_sets[1:]):
        raise ValueError("detectors do not share exactly the same held-out IDs")

    return {
        "status": "pass",
        "protocol": "frozen-ragtruth-replay-audit-v1",
        "checks": {
            "dataset_checksums_match": True,
            "selected_id_digest_matches": True,
            "checkpoint_ids_unique_and_complete": True,
            "checkpoint_labels_groups_and_splits_match_dataset": True,
            "all_scores_within_unit_interval": True,
            "test_ids_identical_across_detectors": True,
            "risk_model_reproduced_exactly": True,
            "summaries_reproduced_exactly": True,
            "paired_statistics_reproduced_exactly": True,
            "no_selected_ids_or_frozen_metrics_in_prediction_source": True,
            "no_prediction_source_reference_to_result_directory": True,
        },
        "examples": {
            "fit": sum(example.split == "fit" for example in selected),
            "calibration": sum(example.split == "train" for example in selected),
            "test": sum(example.split == "test" for example in selected),
        },
        "detectors": len(records_by_detector),
        "paired_comparisons": paired_verified,
        "artifact_sha256": {
            path.name: _sha256_file(path)
            for path in sorted(output_dir.glob("scores_*.jsonl"))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.output_dir, args.data_dir)
    report_path = args.output_dir / "integrity_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Audit saved: {report_path}")


if __name__ == "__main__":
    main()
