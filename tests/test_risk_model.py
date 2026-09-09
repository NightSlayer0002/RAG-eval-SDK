import math
import json
from dataclasses import replace

from benchmarks.detectors import SDKDetector, WholeResponsePairDetector
from benchmarks.ragtruth_benchmark import (
    _load_excluded_groups,
    _partition_fit_split,
    _sdk_detectors,
    _stable_limit,
    build_parser,
)
from benchmarks.runner import apply_train_only_risk_calibration
from benchmarks.schema import BenchmarkExample, ScoreRecord
from rag_eval_sdk.risk_model import RISK_FEATURE_NAMES, fit_risk_calibrator


def _features(risk: float) -> dict[str, float]:
    return {name: float(risk) for name in RISK_FEATURE_NAMES}


def test_risk_calibrator_is_deterministic_and_orders_separable_examples():
    rows = [_features(0.05), _features(0.15), _features(0.85), _features(0.95)]
    labels = [False, False, True, True]
    first = fit_risk_calibrator(rows, labels)
    second = fit_risk_calibrator(rows, labels)
    assert first.calibration_id == second.calibration_id
    assert len(first.feature_names) == 2
    assert first.predict_proba(_features(0.9)) > first.predict_proba(_features(0.1))
    assert 0.0 <= first.predict_proba(_features(0.5)) <= 1.0


def test_train_only_calibration_preserves_raw_score_and_never_drops_failures():
    records = [
        ScoreRecord(
            "sdk", "fit-0", "fit-g0", "fit", False, 0.1, 1.0,
            usage={"risk_features": _features(0.1)},
        ),
        ScoreRecord(
            "sdk", "fit-1", "fit-g1", "fit", True, 0.9, 1.0,
            usage={"risk_features": _features(0.9)},
        ),
        ScoreRecord(
            "sdk", "train-0", "cal-g0", "train", False, 0.2, 1.0,
            usage={"risk_features": _features(0.2)},
        ),
        ScoreRecord(
            "sdk", "test-0", "test-g0", "test", True, None, 1.0,
            error="quota",
        ),
    ]
    calibrated, manifest = apply_train_only_risk_calibration(records)
    assert manifest["fit_successful_examples"] == 2
    assert manifest["decision_threshold"] == 0.5
    assert manifest["threshold_selection"]["strategy"] == "fixed-probability-fallback"
    assert calibrated[2].usage["raw_grounding_risk"] == 0.2
    assert math.isfinite(calibrated[2].score)
    assert calibrated[3].score is None and calibrated[3].error == "quota"


def test_grouped_oof_threshold_uses_only_fit_groups_and_is_deterministic():
    records = []
    for index in range(10):
        label = index % 2 == 1
        records.append(
            ScoreRecord(
                "sdk",
                f"fit-{index}",
                f"fit-group-{index}",
                "fit",
                label,
                float(label),
                1.0,
                usage={"risk_features": _features(0.9 if label else 0.1)},
            )
        )
    records.extend(
        [
            ScoreRecord(
                "sdk", "train", "train-group", "train", False, 0.2, 1.0,
                usage={"risk_features": _features(0.2)},
            ),
            ScoreRecord(
                "sdk", "test", "test-group", "test", True, 0.8, 1.0,
                usage={"risk_features": _features(0.8)},
            ),
        ]
    )
    first, first_manifest = apply_train_only_risk_calibration(records)
    second, second_manifest = apply_train_only_risk_calibration(records)
    assert first_manifest["threshold_selection"]["strategy"] == "grouped-out-of-fold"
    assert first_manifest["decision_threshold"] == second_manifest["decision_threshold"]
    assert [row.score for row in first] == [row.score for row in second]


def test_non_fit_labels_cannot_change_risk_model_or_predictions():
    records = []
    for index in range(10):
        label = index % 2 == 1
        records.append(
            ScoreRecord(
                "sdk",
                f"fit-{index}",
                f"fit-group-{index}",
                "fit",
                label,
                float(label),
                1.0,
                usage={"risk_features": _features(0.9 if label else 0.1)},
            )
        )
    records.extend(
        [
            ScoreRecord(
                "sdk", "train", "train-group", "train", False, 0.3, 1.0,
                usage={"risk_features": _features(0.3)},
            ),
            ScoreRecord(
                "sdk", "test", "test-group", "test", True, 0.7, 1.0,
                usage={"risk_features": _features(0.7)},
            ),
        ]
    )
    flipped = [
        replace(record, hallucinated=not record.hallucinated)
        if record.split != "fit"
        else record
        for record in records
    ]
    original_predictions, original_manifest = apply_train_only_risk_calibration(records)
    flipped_predictions, flipped_manifest = apply_train_only_risk_calibration(flipped)
    assert original_manifest == flipped_manifest
    assert [row.score for row in original_predictions] == [
        row.score for row in flipped_predictions
    ]


def test_fit_partition_is_group_safe_and_contains_both_labels_on_each_side():
    examples = []
    for group_index in range(6):
        for label in (False, True):
            examples.append(
                BenchmarkExample(
                    id=f"train-{group_index}-{int(label)}",
                    query="q",
                    response="r",
                    contexts=("c",),
                    hallucinated=label,
                    split="train",
                    group_id=f"group-{group_index}",
                )
            )
    examples.append(
        BenchmarkExample(
            id="test", query="q", response="r", contexts=("c",),
            hallucinated=True, split="test", group_id="test-group",
        )
    )
    partitioned = _partition_fit_split(examples, 0.67)
    fit_groups = {row.group_id for row in partitioned if row.split == "fit"}
    calibration_groups = {row.group_id for row in partitioned if row.split == "train"}
    assert fit_groups.isdisjoint(calibration_groups)
    assert {row.hallucinated for row in partitioned if row.split == "fit"} == {
        False,
        True,
    }
    assert {row.hallucinated for row in partitioned if row.split == "train"} == {
        False,
        True,
    }
    assert next(row for row in partitioned if row.id == "test").split == "test"


def test_stable_limit_offsets_create_disjoint_label_independent_windows():
    examples = [
        BenchmarkExample(
            id=f"{split}-{index}",
            query="q",
            response="r",
            contexts=("c",),
            hallucinated=bool(index % 2),
            split=split,
            group_id=f"{split}-group-{index}",
        )
        for split in ("train", "test")
        for index in range(20)
    ]
    first = _stable_limit(examples, 5, 5)
    second = _stable_limit(examples, 5, 5, validation_offset=5, test_offset=5)
    assert {row.id for row in first}.isdisjoint(row.id for row in second)
    assert sum(row.split == "train" for row in second) == 5
    assert sum(row.split == "test" for row in second) == 5


def test_prior_checkpoint_groups_can_be_excluded_without_reading_labels(tmp_path):
    checkpoint = tmp_path / "scores.jsonl"
    checkpoint.write_text(
        "\n".join(
            json.dumps({"example_id": identifier, "group_id": group})
            for identifier, group in (("a", "group-1"), ("b", "group-2"))
        ),
        encoding="utf-8",
    )

    assert _load_excluded_groups([str(checkpoint)]) == {"group-1", "group-2"}


def test_sdk_and_whole_response_ablation_do_not_share_result_cache():
    args = build_parser().parse_args(["--data-dir", "unused"])
    sdk, whole = _sdk_detectors(args, include_sdk=True, include_ablation=True)
    assert isinstance(sdk, SDKDetector)
    assert isinstance(whole, WholeResponsePairDetector)
    assert sdk.evaluator.pair_verifier is not whole.verifier
    assert sdk.evaluator.pair_verifier.verifier is whole.verifier.verifier
