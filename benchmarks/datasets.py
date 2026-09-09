"""Strict local dataset adapters.  No download occurs in this module."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .schema import BenchmarkExample


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def _stringify_source(value: object, prefix: str = "") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
        if prefix:
            return [f"{prefix}: {paragraph}" for paragraph in paragraphs]
        return paragraphs or ([value.strip()] if value.strip() else [])
    if isinstance(value, Mapping):
        chunks: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            chunks.extend(_stringify_source(child, child_prefix))
        return chunks
    if isinstance(value, list):
        chunks = []
        for index, child in enumerate(value):
            chunks.extend(_stringify_source(child, f"{prefix}[{index}]"))
        return chunks
    return [f"{prefix}: {value}" if prefix else str(value)]


def _query_from_source(row: Mapping[str, Any]) -> str:
    source_info = row.get("source_info")
    if isinstance(source_info, Mapping):
        for key in ("question", "query", "instruction"):
            if source_info.get(key):
                return str(source_info[key])
    if row.get("prompt"):
        return str(row["prompt"])
    task = str(row.get("task_type", "generation"))
    return f"Complete the {task} task using the supplied source information."


def _validated_ragtruth_spans(
    labels: object,
    response_text: str,
    response_id: object,
) -> tuple[dict[str, Any], ...]:
    """Validate and preserve the official character-level annotations."""

    if not isinstance(labels, list):
        raise ValueError(f"Response {response_id} has a non-list labels field")
    spans: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise ValueError(
                f"Response {response_id} label {label_index} is not an object"
            )
        start = label.get("start")
        end = label.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(response_text)
        ):
            raise ValueError(
                f"Response {response_id} label {label_index} has invalid offsets "
                f"({start!r}, {end!r}) for response length {len(response_text)}"
            )
        annotated_text = label.get("text")
        if not isinstance(annotated_text, str):
            raise ValueError(
                f"Response {response_id} label {label_index} has non-string text"
            )
        if response_text[start:end] != annotated_text:
            raise ValueError(
                f"Response {response_id} label {label_index} text does not match its offsets"
            )
        spans.append(
            {
                "start": start,
                "end": end,
                "text": annotated_text,
                "label_type": str(label.get("label_type", "unknown")),
                "implicit_true": bool(label.get("implicit_true", False)),
                "due_to_null": bool(label.get("due_to_null", False)),
                "annotation_meta": str(label.get("meta", "")),
            }
        )
    return tuple(spans)


def load_ragtruth(
    directory: str | Path,
    *,
    allowed_quality: Iterable[str] = ("good",),
    splits: Iterable[str] = ("train", "test"),
) -> Iterator[BenchmarkExample]:
    """Join the official RAGTruth source and response JSONL files.

    Hallucination is positive when at least one annotated span is present,
    including ``implicit_true`` spans: this benchmark measures support by the
    provided source, not truth in the wider world.
    """

    root = Path(directory)
    source_path = root / "source_info.jsonl"
    response_path = root / "response.jsonl"
    if not source_path.is_file() or not response_path.is_file():
        raise FileNotFoundError(
            "RAGTruth requires source_info.jsonl and response.jsonl in the dataset directory"
        )
    sources = {str(row["source_id"]): row for row in _read_jsonl(source_path)}
    quality_set = set(allowed_quality)
    split_set = set(splits)
    for response in _read_jsonl(response_path):
        if str(response.get("split")) not in split_set:
            continue
        if str(response.get("quality", "good")) not in quality_set:
            continue
        source_id = str(response.get("source_id"))
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"Response {response.get('id')} references missing source {source_id}")
        response_text = str(response.get("response", ""))
        raw_labels = response.get("labels")
        gold_spans = _validated_ragtruth_spans(
            [] if raw_labels is None else raw_labels,
            response_text,
            response.get("id"),
        )
        source_info = source.get("source_info")
        if isinstance(source_info, Mapping):
            source_info = {
                key: value
                for key, value in source_info.items()
                if str(key).casefold() not in {"question", "query", "instruction"}
            }
        contexts = _stringify_source(source_info)
        if not contexts:
            raise ValueError(f"Source {source_id} has no usable source_info")
        yield BenchmarkExample(
            id=str(response["id"]),
            query=_query_from_source(source),
            response=response_text,
            contexts=tuple(contexts),
            hallucinated=bool(gold_spans),
            split=str(response["split"]),
            group_id=source_id,
            metadata={
                "task_type": source.get("task_type"),
                "source": source.get("source"),
                "model": response.get("model"),
                "temperature": response.get("temperature"),
                "quality": response.get("quality"),
                "label_types": sorted(
                    {
                        str(span["label_type"])
                        for span in gold_spans
                    }
                ),
                "gold_spans": gold_spans,
            },
        )
