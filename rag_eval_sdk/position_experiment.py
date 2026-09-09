"""Paired position interventions for measuring generator sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import EvidenceChunk, coerce_chunks


@dataclass(frozen=True)
class PositionVariant:
    treatment: str
    target_chunk_id: str
    chunks: tuple[EvidenceChunk, ...]
    target_index: int


@dataclass(frozen=True)
class PositionEffect:
    beginning_score: float
    middle_score: float
    end_score: float
    maximum_gap: float
    middle_drop: float
    end_drop: float
    preferred_position: str
    position_sensitive: bool
    threshold: float


def create_position_variants(
    chunks: Sequence[EvidenceChunk | Mapping[str, object] | str],
    target_chunk_id: str,
) -> tuple[PositionVariant, PositionVariant, PositionVariant]:
    """Move one target chunk while preserving every other chunk's order."""

    normalized = coerce_chunks(chunks)
    target_matches = [chunk for chunk in normalized if chunk.id == str(target_chunk_id)]
    if len(target_matches) != 1:
        raise ValueError("target_chunk_id must identify exactly one non-empty chunk")
    target = target_matches[0]
    others = [chunk for chunk in normalized if chunk.id != target.id]
    middle_index = len(others) // 2
    arrangements = {
        "beginning": [target, *others],
        "middle": [*others[:middle_index], target, *others[middle_index:]],
        "end": [*others, target],
    }
    return tuple(
        PositionVariant(name, target.id, tuple(order), order.index(target))
        for name, order in arrangements.items()
    )  # type: ignore[return-value]


def analyse_position_effect(
    treatment_scores: Mapping[str, float],
    *,
    sensitivity_threshold: float = 0.10,
) -> PositionEffect:
    """Summarize paired scores produced under the three position treatments."""

    missing = {"beginning", "middle", "end"} - set(treatment_scores)
    if missing:
        raise ValueError(f"missing treatment scores: {sorted(missing)}")
    values = {name: float(treatment_scores[name]) for name in ("beginning", "middle", "end")}
    if any(not 0.0 <= score <= 1.0 for score in values.values()):
        raise ValueError("treatment scores must be within [0, 1]")
    maximum_gap = max(values.values()) - min(values.values())
    preferred = max(values, key=values.get)  # type: ignore[arg-type]
    return PositionEffect(
        beginning_score=values["beginning"],
        middle_score=values["middle"],
        end_score=values["end"],
        maximum_gap=maximum_gap,
        middle_drop=max(0.0, values["beginning"] - values["middle"]),
        end_drop=max(0.0, values["beginning"] - values["end"]),
        preferred_position=preferred,
        position_sensitive=maximum_gap >= sensitivity_threshold,
        threshold=sensitivity_threshold,
    )

