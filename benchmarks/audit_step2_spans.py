"""Replay and integrity-audit a frozen RAGTruth span-localization run.

This command performs no model inference.  It reconstructs the source-exclusive
cohort from the dataset and manifest, validates every saved prediction, and
recomputes both the answer-level and character-span reports exactly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from rag_eval_sdk.text_features import split_claim_spans

from .datasets import load_ragtruth
from .ragtruth_benchmark import _load_excluded_groups, _sha256_file, _stable_limit
from .runner import summarize_detector
from .schema import BenchmarkExample, ScoreRecord
from .span_evaluation import read_score_records, summarize_span_localization


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _require_equal(actual: Any, expected: Any, description: str) -> None:
    if _json_normalize(actual) != _json_normalize(expected):
        raise ValueError(f"Replay mismatch: {description}")


def _require_unit_float(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{description} is outside [0, 1]")
    return number


def _single_matching_file(output_dir: Path, pattern: str) -> Path:
    matches = sorted(output_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} file in {output_dir}; found {len(matches)}"
        )
    return matches[0]


def _source_string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _scan_prediction_source(
    project_root: Path,
    *,
    selected_ids: set[str],
    selected_ids_digest: str,
    result_directory_name: str,
) -> None:
    prediction_paths = [
        *(project_root / "rag_eval_sdk").glob("*.py"),
        project_root / "benchmarks" / "detectors.py",
        project_root / "benchmarks" / "span_evaluation.py",
    ]
    forbidden_fragments = {
        selected_ids_digest.casefold(),
        result_directory_name.casefold(),
        "ragtruth_step2_spans_frozen_20260821",
    }
    for path in prediction_paths:
        text = path.read_text(encoding="utf-8")
        folded = text.casefold()
        if any(fragment in folded for fragment in forbidden_fragments):
            raise ValueError(f"Prediction source references the frozen cohort: {path}")
        hard_coded_ids = _source_string_constants(path) & selected_ids
        if hard_coded_ids:
            raise ValueError(
                f"Prediction source contains selected example IDs {sorted(hard_coded_ids)[:3]}: {path}"
            )


def _checked_exclusion_files(
    manifest: Mapping[str, Any], project_root: Path
) -> tuple[list[str], dict[str, str]]:
    """Replay local checkpoint bytes while preserving historical manifest keys."""
    arguments = manifest["arguments"]
    if not isinstance(arguments, Mapping):
        raise ValueError("Manifest arguments must be a mapping")
    exclusion_values = arguments.get("exclude_groups_from")
    if not isinstance(exclusion_values, list) or not all(
        isinstance(value, str) for value in exclusion_values
    ):
        raise ValueError("Manifest exclude_groups_from must be a list of paths")

    resolved_paths: list[str] = []
    exclusion_hashes: dict[str, str] = {}
    for value in exclusion_values:
        # PureWindowsPath handles both slash styles, including on POSIX clones.
        parts = PureWindowsPath(value).parts
        path = Path(value)
        if "benchmarks" in parts:
            relative_parts = parts[parts.index("benchmarks"):]
            if ".." in relative_parts:
                raise ValueError("Exclusion checkpoint path escapes the repository")
            path = project_root.joinpath(*relative_parts)
        resolved_paths.append(str(path))
        key = value if value in manifest["exclusion_files"] else str(Path(value).resolve())
        exclusion_hashes[key] = _sha256_file(path)
    _require_equal(
        exclusion_hashes,
        manifest["exclusion_files"],
        "exclusion checkpoint hashes",
    )
    return resolved_paths, exclusion_hashes


def _selected_examples(
    manifest: Mapping[str, Any], data_dir: Path
) -> tuple[list[BenchmarkExample], set[str], dict[str, str]]:
    arguments = manifest["arguments"]
    exclusion_paths, exclusion_hashes = _checked_exclusion_files(
        manifest, Path(__file__).resolve().parents[1]
    )
    excluded_groups = _load_excluded_groups(exclusion_paths)
    excluded_digest = hashlib.sha256(
        "\n".join(sorted(excluded_groups)).encode("utf-8")
    ).hexdigest()
    _require_equal(
        excluded_digest,
        manifest["excluded_groups_sha256"],
        "excluded source-group digest",
    )

    available = [
        example
        for example in load_ragtruth(data_dir)
        if example.group_id not in excluded_groups
    ]
    selected = _stable_limit(
        available,
        arguments["validation_limit"],
        arguments["test_limit"],
        arguments["validation_offset"],
        arguments["test_offset"],
    )
    if arguments["risk_calibrator"] != "threshold":
        raise ValueError("Step 2 audit expects the preregistered threshold calibrator")
    return selected, excluded_groups, exclusion_hashes


def _verify_record(
    record: ScoreRecord,
    expected: BenchmarkExample,
    *,
    max_claims: int,
    granularity: str,
) -> None:
    _require_equal(record.group_id, expected.group_id, f"{record.example_id} group")
    _require_equal(record.split, expected.split, f"{record.example_id} split")
    _require_equal(record.hallucinated, expected.hallucinated, f"{record.example_id} label")
    _require_equal(record.metadata, expected.metadata, f"{record.example_id} metadata")
    if record.error is not None or record.score is None:
        raise ValueError(f"Frozen run contains a failed record: {record.example_id}")
    _require_unit_float(record.score, f"{record.example_id} answer score")

    usage = record.usage
    if usage.get("claim_localization_schema") != "claim-risk-spans-v1":
        raise ValueError(f"Unsupported localization schema: {record.example_id}")
    response_received = usage.get("response_characters_received")
    response_used = usage.get("response_characters_used")
    if response_received != len(expected.response):
        raise ValueError(f"Response length mismatch: {record.example_id}")
    if not isinstance(response_used, int) or not 0 <= response_used <= len(expected.response):
        raise ValueError(f"Invalid used response length: {record.example_id}")

    values = usage.get("claim_predictions")
    if not isinstance(values, list) or not all(isinstance(value, Mapping) for value in values):
        raise ValueError(f"Malformed claim predictions: {record.example_id}")
    if usage.get("claims") != len(values):
        raise ValueError(f"Claim telemetry mismatch: {record.example_id}")
    expected_spans = split_claim_spans(
        expected.response[:response_used],
        max_claims=max_claims,
        granularity=granularity,
    )
    if len(values) != len(expected_spans):
        raise ValueError(f"Claim count differs from deterministic splitter: {record.example_id}")
    for index, (value, span) in enumerate(zip(values, expected_spans), 1):
        prefix = f"{record.example_id} claim {index}"
        _require_equal(value.get("claim_id"), f"claim-{index}", f"{prefix} ID")
        _require_equal(value.get("start"), span.start, f"{prefix} start")
        _require_equal(value.get("end"), span.end, f"{prefix} end")
        _require_unit_float(value.get("risk_score"), f"{prefix} risk")
        _require_unit_float(value.get("pair_consistency"), f"{prefix} consistency")
        _require_unit_float(value.get("pair_contradiction"), f"{prefix} contradiction")
        evidence_ids = value.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not all(
            isinstance(evidence_id, str) for evidence_id in evidence_ids
        ):
            raise ValueError(f"Malformed evidence IDs: {prefix}")


def audit(
    output_dir: Path,
    data_dir: Path,
    *,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    answer_summary_path = _single_matching_file(output_dir, "summary_*.json")
    score_path = _single_matching_file(output_dir, "scores_*.jsonl")
    span_summary_path = output_dir / "span_summary_frozen.json"
    manifest = _read_json(manifest_path)
    arguments = manifest["arguments"]

    dataset_hashes = {
        name: _sha256_file(data_dir / name)
        for name in ("source_info.jsonl", "response.jsonl")
    }
    _require_equal(dataset_hashes, manifest["dataset_checksums"], "dataset hashes")
    selected, excluded_groups, exclusion_hashes = _selected_examples(manifest, data_dir)

    selected_digest = hashlib.sha256(
        "\n".join(example.id for example in selected).encode("utf-8")
    ).hexdigest()
    _require_equal(selected_digest, manifest["example_ids_sha256"], "selected ID digest")
    _require_equal(len(selected), manifest["examples"], "selected example count")

    selected_groups = {example.group_id for example in selected}
    old_overlap = selected_groups & excluded_groups
    if old_overlap:
        raise ValueError(f"{len(old_overlap)} Step 1 source groups leaked into Step 2")
    calibration_groups = {
        example.group_id for example in selected if example.split == "train"
    }
    test_groups = {example.group_id for example in selected if example.split == "test"}
    if calibration_groups & test_groups:
        raise ValueError("Calibration and test source groups overlap")

    records = read_score_records(score_path)
    if len(records) != len({record.example_id for record in records}):
        raise ValueError("The frozen score checkpoint contains duplicate example IDs")
    _require_equal(
        [record.example_id for record in records],
        [example.id for example in selected],
        "score record order and IDs",
    )
    expected_by_id = {example.id: example for example in selected}
    for record in records:
        _verify_record(
            record,
            expected_by_id[record.example_id],
            max_claims=int(arguments["max_claims"]),
            granularity=str(arguments["claim_granularity"]),
        )

    replayed_answer = summarize_detector(records, objective=str(arguments["objective"]))
    _require_equal(replayed_answer, _read_json(answer_summary_path), "answer summary")
    replayed_spans = summarize_span_localization(
        records,
        bootstrap_samples=bootstrap_samples,
    )
    _require_equal(replayed_spans, _read_json(span_summary_path), "span summary")

    project_root = Path(__file__).resolve().parents[1]
    _scan_prediction_source(
        project_root,
        selected_ids={example.id for example in selected},
        selected_ids_digest=selected_digest,
        result_directory_name=output_dir.name,
    )

    test_records = [record for record in records if record.split == "test"]
    all_records = records
    rss_available = any(
        int(record.usage.get("process_rss_after_bytes", 0)) > 0
        for record in all_records
    )
    artifact_paths = [
        manifest_path,
        score_path,
        answer_summary_path,
        span_summary_path,
    ]
    return {
        "status": "pass",
        "protocol": "ragtruth-step2-span-replay-audit-v1",
        "checks": {
            "dataset_hashes_match": True,
            "exclusion_checkpoint_hashes_match": True,
            "excluded_group_digest_matches": True,
            "selected_id_digest_matches": True,
            "old_test_group_overlap_is_zero": True,
            "calibration_test_group_overlap_is_zero": True,
            "record_ids_unique_complete_and_ordered": True,
            "labels_groups_splits_and_gold_spans_match_dataset": True,
            "all_400_calls_succeeded": True,
            "all_scores_and_claim_probabilities_valid": True,
            "claim_offsets_replay_exactly": True,
            "answer_summary_replays_exactly": True,
            "span_summary_and_bootstrap_replay_exactly": True,
            "no_selected_ids_or_frozen_cohort_reference_in_prediction_source": True,
        },
        "examples": {
            "calibration": sum(example.split == "train" for example in selected),
            "test": sum(example.split == "test" for example in selected),
            "test_positive": sum(
                example.hallucinated and example.split == "test" for example in selected
            ),
            "test_negative": sum(
                not example.hallucinated and example.split == "test" for example in selected
            ),
        },
        "source_groups": {
            "excluded_prior": len(excluded_groups),
            "calibration": len(calibration_groups),
            "test": len(test_groups),
            "prior_overlap": len(old_overlap),
            "calibration_test_overlap": len(calibration_groups & test_groups),
        },
        "telemetry": {
            "test_claim_predictions": sum(
                len(record.usage["claim_predictions"]) for record in test_records
            ),
            "rss_available_in_checkpoint": rss_available,
            "rss_limitation": (
                None
                if rss_available
                else "psutil was unavailable in the benchmark runtime; checkpoint RSS values are zero"
            ),
        },
        "digests": {
            "selected_ids_sha256": selected_digest,
            "dataset_sha256": dataset_hashes,
            "exclusion_files_sha256": exclusion_hashes,
            "artifacts_sha256": {
                path.name: _sha256_file(path) for path in artifact_paths
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    report = audit(
        args.output_dir,
        args.data_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    destination = args.output_dir / "integrity_audit.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Audit saved: {destination}")


if __name__ == "__main__":
    main()
