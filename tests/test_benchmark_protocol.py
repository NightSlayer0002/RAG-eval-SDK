import json
from dataclasses import replace

from benchmarks.datasets import load_ragtruth
from benchmarks.comparison import compare_detectors
from benchmarks.runner import summarize_detector
from benchmarks.schema import ScoreRecord


def test_ragtruth_loader_joins_sources_and_preserves_labels(tmp_path):
    sources = [
        {
            "source_id": "train-source",
            "task_type": "QA",
            "source": "MARCO",
            "source_info": {
                "question": "Where is Paris?",
                "passages": "Paris is in France.",
            },
        },
        {
            "source_id": "test-source",
            "task_type": "Summary",
            "source": "news",
            "source_info": "A short source document.",
        },
    ]
    responses = [
        {
            "id": "train-response",
            "source_id": "train-source",
            "model": "model-a",
            "temperature": 0,
            "labels": [],
            "split": "train",
            "quality": "good",
            "response": "Paris is in France.",
        },
        {
            "id": "test-response",
            "source_id": "test-source",
            "model": "model-b",
            "temperature": 0,
            "labels": [
                {
                    "start": 3,
                    "end": 14,
                    "text": "unsupported",
                    "label_type": "Evident Baseless Info",
                    "implicit_true": False,
                    "due_to_null": False,
                }
            ],
            "split": "test",
            "quality": "good",
            "response": "An unsupported response.",
        },
    ]
    (tmp_path / "source_info.jsonl").write_text(
        "\n".join(json.dumps(row) for row in sources), encoding="utf-8"
    )
    (tmp_path / "response.jsonl").write_text(
        "\n".join(json.dumps(row) for row in responses), encoding="utf-8"
    )

    examples = list(load_ragtruth(tmp_path))
    assert len(examples) == 2
    assert examples[0].query == "Where is Paris?"
    assert examples[0].contexts == ("passages: Paris is in France.",)
    assert examples[1].hallucinated is True
    assert examples[1].metadata["gold_spans"] == (
        {
            "start": 3,
            "end": 14,
            "text": "unsupported",
            "label_type": "Evident Baseless Info",
            "implicit_true": False,
            "due_to_null": False,
            "annotation_meta": "",
        },
    )


def test_ragtruth_loader_rejects_annotation_text_offset_mismatch(tmp_path):
    source = {
        "source_id": "source",
        "task_type": "QA",
        "source_info": {"question": "Question?", "passages": "Evidence."},
    }
    response = {
        "id": "response",
        "source_id": "source",
        "labels": [{"start": 0, "end": 4, "text": "wrong"}],
        "split": "test",
        "quality": "good",
        "response": "Right answer.",
    }
    (tmp_path / "source_info.jsonl").write_text(
        json.dumps(source), encoding="utf-8"
    )
    (tmp_path / "response.jsonl").write_text(
        json.dumps(response), encoding="utf-8"
    )

    try:
        list(load_ragtruth(tmp_path))
    except ValueError as exc:
        assert "text does not match" in str(exc)
    else:
        raise AssertionError("invalid RAGTruth annotation should be rejected")


def test_summary_calibrates_on_train_and_scores_test():
    records = [
        ScoreRecord("detector", "tr-0", "g-tr-0", "train", False, 0.1, 1.0),
        ScoreRecord("detector", "tr-1", "g-tr-1", "train", True, 0.9, 1.0),
        ScoreRecord(
            "detector", "te-0", "g-te-0", "test", False, 0.2, 1.0,
            metadata={"task_type": "QA", "model": "generator-a"},
        ),
        ScoreRecord(
            "detector", "te-1", "g-te-1", "test", True, 0.8, 1.0,
            metadata={"task_type": "QA", "model": "generator-b"},
        ),
    ]
    summary = summarize_detector(records)
    assert summary["held_out"]["f1"] == 1.0
    assert summary["held_out"]["coverage"] == 1.0
    assert summary["slices"]["task_type"]["QA"]["successful_only"]["f1"] == 1.0
    assert summary["slices"]["macro_task"]["f1"] == 1.0


def test_test_labels_cannot_change_selected_threshold():
    records = [
        ScoreRecord("detector", "tr-0", "g-tr-0", "train", False, 0.1, 1.0),
        ScoreRecord("detector", "tr-1", "g-tr-1", "train", True, 0.9, 1.0),
        ScoreRecord("detector", "te-0", "g-te-0", "test", False, 0.2, 1.0),
        ScoreRecord("detector", "te-1", "g-te-1", "test", True, 0.8, 1.0),
    ]
    flipped_test_labels = [
        replace(record, hallucinated=not record.hallucinated)
        if record.split == "test"
        else record
        for record in records
    ]
    original = summarize_detector(records)
    flipped = summarize_detector(flipped_test_labels)
    assert original["calibration"] == flipped["calibration"]


def test_summary_counts_failed_test_scores_as_incorrect_in_sensitivity_metric():
    records = [
        ScoreRecord("detector", "tr-0", "g-tr-0", "train", False, 0.1, 1.0),
        ScoreRecord("detector", "tr-1", "g-tr-1", "train", True, 0.9, 1.0),
        ScoreRecord("detector", "te-0", "g-te-0", "test", False, 0.2, 1.0),
        ScoreRecord("detector", "te-2", "g-te-2", "test", True, 0.8, 1.0),
        ScoreRecord(
            "detector", "te-1", "g-te-1", "test", True, None, 1.0, error="quota"
        ),
    ]
    summary = summarize_detector(records)
    assert summary["held_out"]["coverage"] == 2 / 3
    assert summary["held_out"]["failure_as_incorrect"]["false_negative"] == 1


def test_summary_can_use_predeclared_probability_threshold():
    records = [
        ScoreRecord("detector", "tr-0", "g-tr-0", "train", False, 0.45, 1.0),
        ScoreRecord("detector", "tr-1", "g-tr-1", "train", True, 0.46, 1.0),
        ScoreRecord("detector", "te-0", "g-te-0", "test", False, 0.2, 1.0),
        ScoreRecord("detector", "te-1", "g-te-1", "test", True, 0.8, 1.0),
    ]
    summary = summarize_detector(records, fixed_threshold=0.5)
    assert summary["calibration"]["threshold"] == 0.5
    assert summary["calibration"]["objective"] == "fixed_probability_threshold"
    assert summary["held_out"]["f1"] == 1.0


def test_paired_comparison_reports_coverage_and_failure_sensitivity():
    first = [
        ScoreRecord("first", "te-0", "g-0", "test", False, 0.1, 1.0),
        ScoreRecord("first", "te-1", "g-1", "test", True, None, 1.0, error="quota"),
    ]
    second = [
        ScoreRecord("second", "te-0", "g-0", "test", False, 0.1, 1.0),
        ScoreRecord("second", "te-1", "g-1", "test", True, 0.9, 1.0),
    ]
    comparison = compare_detectors(
        first,
        second,
        first_threshold=0.5,
        second_threshold=0.5,
        bootstrap_samples=20,
    )
    assert comparison["first_coverage"] == 0.5
    assert comparison["second_coverage"] == 1.0
    assert comparison["failure_as_incorrect"]["first_f1"] == 0.0
    assert comparison["failure_as_incorrect"]["second_f1"] == 1.0
