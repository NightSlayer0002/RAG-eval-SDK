"""Exact, dependency-free metrics for RAGTruth character annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from rag_eval_sdk.schema import Decision, VerificationReport


@dataclass(frozen=True, order=True)
class Span:
    """A half-open character range ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise ValueError("span offsets must be integers, not booleans")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span offsets must satisfy 0 <= start < end")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Span":
        start = value.get("start")
        end = value.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("span mappings require integer start and end fields")
        return cls(start=start, end=end)


@dataclass(frozen=True)
class SpanMetrics:
    precision: float
    recall: float
    f1: float
    intersection_over_union: float
    true_positive_characters: int
    false_positive_characters: int
    false_negative_characters: int
    predicted_characters: int
    gold_characters: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _coerce_span(value: Span | Mapping[str, object]) -> Span:
    return value if isinstance(value, Span) else Span.from_mapping(value)


def merge_spans(
    spans: Iterable[Span | Mapping[str, object]],
    *,
    text_length: int | None = None,
) -> tuple[Span, ...]:
    """Return the interval union so overlaps never double-count characters."""

    if text_length is not None and text_length < 0:
        raise ValueError("text_length must be non-negative or None")
    ordered = sorted(_coerce_span(span) for span in spans)
    for span in ordered:
        if text_length is not None and span.end > text_length:
            raise ValueError(
                f"span end {span.end} exceeds text length {text_length}"
            )
    merged: list[Span] = []
    for span in ordered:
        if not merged or span.start > merged[-1].end:
            merged.append(span)
            continue
        previous = merged[-1]
        merged[-1] = Span(previous.start, max(previous.end, span.end))
    return tuple(merged)


def _total_length(spans: Sequence[Span]) -> int:
    return sum(span.end - span.start for span in spans)


def _intersection_length(first: Sequence[Span], second: Sequence[Span]) -> int:
    first_index = 0
    second_index = 0
    overlap = 0
    while first_index < len(first) and second_index < len(second):
        left = first[first_index]
        right = second[second_index]
        overlap += max(0, min(left.end, right.end) - max(left.start, right.start))
        if left.end <= right.end:
            first_index += 1
        else:
            second_index += 1
    return overlap


def _metrics_from_counts(true_positive: int, false_positive: int, false_negative: int) -> SpanMetrics:
    predicted = true_positive + false_positive
    gold = true_positive + false_negative
    precision = true_positive / predicted if predicted else float(gold == 0)
    recall = true_positive / gold if gold else float(predicted == 0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    union = true_positive + false_positive + false_negative
    iou = true_positive / union if union else 1.0
    return SpanMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        intersection_over_union=iou,
        true_positive_characters=true_positive,
        false_positive_characters=false_positive,
        false_negative_characters=false_negative,
        predicted_characters=predicted,
        gold_characters=gold,
    )


def character_span_metrics(
    predicted: Iterable[Span | Mapping[str, object]],
    gold: Iterable[Span | Mapping[str, object]],
    *,
    text_length: int | None = None,
) -> SpanMetrics:
    """Compute positive-class character overlap using half-open offsets.

    This follows RAGTruth's published span-level protocol: precision, recall,
    and F1 are calculated from character overlap. An all-negative example is a
    perfect per-example match; corpus micro scores instead sum counts first.
    """

    predicted_union = merge_spans(predicted, text_length=text_length)
    gold_union = merge_spans(gold, text_length=text_length)
    predicted_characters = _total_length(predicted_union)
    gold_characters = _total_length(gold_union)
    true_positive = _intersection_length(predicted_union, gold_union)
    return _metrics_from_counts(
        true_positive,
        predicted_characters - true_positive,
        gold_characters - true_positive,
    )


def aggregate_character_span_metrics(
    examples: Iterable[
        tuple[
            Iterable[Span | Mapping[str, object]],
            Iterable[Span | Mapping[str, object]],
            int,
        ]
    ],
) -> dict[str, object]:
    """Return corpus-micro and example-macro metrics without materializing masks."""

    per_example: list[SpanMetrics] = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    for predicted, gold, text_length in examples:
        metrics = character_span_metrics(
            predicted, gold, text_length=text_length
        )
        per_example.append(metrics)
        total_true_positive += metrics.true_positive_characters
        total_false_positive += metrics.false_positive_characters
        total_false_negative += metrics.false_negative_characters
    if not per_example:
        raise ValueError("at least one span example is required")
    micro = _metrics_from_counts(
        total_true_positive, total_false_positive, total_false_negative
    )
    count = len(per_example)
    return {
        "examples": count,
        "micro": micro.to_dict(),
        "macro": {
            "precision": sum(item.precision for item in per_example) / count,
            "recall": sum(item.recall for item in per_example) / count,
            "f1": sum(item.f1 for item in per_example) / count,
            "intersection_over_union": sum(
                item.intersection_over_union for item in per_example
            )
            / count,
        },
    }


def predicted_claim_spans(
    report: VerificationReport,
    *,
    claim_risk_threshold: float | None = None,
) -> tuple[Span, ...]:
    """Project unsafe verification units back to response character spans."""

    if claim_risk_threshold is not None and not 0.0 <= claim_risk_threshold <= 1.0:
        raise ValueError("claim_risk_threshold must be within [0, 1]")
    unsafe_decisions = {
        Decision.CONTRADICTED,
        Decision.INSUFFICIENT,
        Decision.ESCALATE,
        Decision.ABSTAIN,
    }
    predicted: list[Span] = []
    for claim in report.claims:
        is_unsafe = (
            claim.risk_score >= claim_risk_threshold
            if claim_risk_threshold is not None
            else claim.decision in unsafe_decisions
        )
        if not is_unsafe:
            continue
        if claim.span_start is None or claim.span_end is None:
            raise ValueError(
                f"Claim {claim.claim_id!r} does not contain response offsets"
            )
        predicted.append(Span(claim.span_start, claim.span_end))
    return merge_spans(predicted)
