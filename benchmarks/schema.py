"""Shared records for reproducible detector benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    query: str
    response: str
    contexts: Sequence[str]
    hallucinated: bool
    split: str
    group_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreRecord:
    detector: str
    example_id: str
    group_id: str
    split: str
    hallucinated: bool
    score: float | None
    elapsed_ms: float
    error: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
