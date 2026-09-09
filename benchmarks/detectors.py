"""Adapters that expose comparable hallucination-risk scores."""

from __future__ import annotations

import asyncio
import importlib.metadata
import time
from typing import Any, Protocol

from rag_eval_sdk import RAGEvaluator
from rag_eval_sdk.risk_model import extract_risk_features
from rag_eval_sdk.text_features import pack_evidence
from rag_eval_sdk.verifiers import PairVerifier

from .schema import BenchmarkExample, ScoreRecord


def _memory_snapshot() -> dict[str, int]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        result = {"process_rss_bytes": int(info.rss)}
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            result["process_peak_working_set_bytes"] = int(peak)
        return result
    except (ImportError, OSError):
        return {}


class Detector(Protocol):
    name: str

    def score(self, example: BenchmarkExample) -> ScoreRecord:
        """Return a high-is-more-hallucinated score or an explicit error."""


class SDKDetector:
    def __init__(self, evaluator: RAGEvaluator, name: str = "rag-eval-sdk-v2") -> None:
        self.evaluator = evaluator
        self.name = name

    def score(self, example: BenchmarkExample) -> ScoreRecord:
        started = time.perf_counter()
        memory_before = _memory_snapshot()
        try:
            report = self.evaluator.evaluate(example.response, example.contexts)
            return ScoreRecord(
                detector=self.name,
                example_id=example.id,
                group_id=example.group_id,
                split=example.split,
                hallucinated=example.hallucinated,
                score=report.grounding_risk,
                elapsed_ms=report.usage.elapsed_ms,
                usage={
                    "risk_features": extract_risk_features(report),
                    "claim_localization_schema": "claim-risk-spans-v1",
                    "claim_predictions": [
                        {
                            "claim_id": claim.claim_id,
                            "start": claim.span_start,
                            "end": claim.span_end,
                            "decision": claim.decision.value,
                            "risk_score": claim.risk_score,
                            "pair_consistency": claim.pair_score.consistency,
                            "pair_contradiction": claim.pair_score.contradiction,
                            "evidence_ids": list(claim.evidence_ids),
                        }
                        for claim in report.claims
                    ],
                    "claims": report.usage.claims,
                    "evidence_pairs_scored": report.usage.evidence_pairs_scored,
                    "verifier_batches": report.usage.verifier_batches,
                    "audit_probes": report.usage.audit_probes,
                    "remote_calls": report.usage.remote_calls,
                    "estimated_input_tokens": report.usage.estimated_input_tokens,
                    "input_chunks_received": report.usage.input_chunks_received,
                    "input_chunks_used": report.usage.input_chunks_used,
                    "truncated_chunks": report.usage.truncated_chunks,
                    "response_characters_received": report.usage.response_characters_received,
                    "response_characters_used": report.usage.response_characters_used,
                    "truncated_claims": report.usage.truncated_claims,
                    "similarity_matrix_bytes": report.usage.similarity_matrix_bytes,
                    "process_rss_before_bytes": memory_before.get("process_rss_bytes", 0),
                    "process_rss_after_bytes": _memory_snapshot().get("process_rss_bytes", 0),
                },
                metadata=example.metadata,
            )
        except Exception as exc:
            return ScoreRecord(
                detector=self.name,
                example_id=example.id,
                group_id=example.group_id,
                split=example.split,
                hallucinated=example.hallucinated,
                score=None,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
                usage={
                    "claim_localization_schema": "claim-risk-spans-v1",
                    "claim_predictions": [],
                    "response_characters_received": len(example.response),
                },
                metadata=example.metadata,
            )


class WholeResponsePairDetector:
    """Ablation: one verifier call on the packed context and whole response."""

    def __init__(
        self,
        verifier: PairVerifier,
        *,
        max_evidence_characters: int = 12_000,
        name: str | None = None,
    ) -> None:
        self.verifier = verifier
        self.max_evidence_characters = max_evidence_characters
        self.name = name or f"whole-response:{verifier.name}"

    def score(self, example: BenchmarkExample) -> ScoreRecord:
        started = time.perf_counter()
        memory_before = _memory_snapshot()
        premise = pack_evidence(example.contexts, self.max_evidence_characters)
        try:
            pair = self.verifier.score(premise, example.response)
            return ScoreRecord(
                detector=self.name,
                example_id=example.id,
                group_id=example.group_id,
                split=example.split,
                hallucinated=example.hallucinated,
                score=max(1.0 - pair.consistency, pair.contradiction),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                usage={
                    "claims": 1,
                    "evidence_pairs_scored": 1,
                    "verifier_batches": 1,
                    "audit_probes": 0,
                    "remote_calls": 0,
                    "estimated_input_tokens": (len(premise) + len(example.response) + 3) // 4,
                    "process_rss_before_bytes": memory_before.get("process_rss_bytes", 0),
                    "process_rss_after_bytes": _memory_snapshot().get("process_rss_bytes", 0),
                },
                metadata=example.metadata,
            )
        except Exception as exc:
            return ScoreRecord(
                detector=self.name,
                example_id=example.id,
                group_id=example.group_id,
                split=example.split,
                hallucinated=example.hallucinated,
                score=None,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
                metadata=example.metadata,
            )


class RagasFaithfulnessDetector:
    """Real Ragas Faithfulness, not a locally reimplemented approximation."""

    def __init__(self, llm: Any) -> None:
        try:
            self.version = importlib.metadata.version("ragas")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Install the benchmark extra to use the Ragas baseline") from exc
        self.llm = llm
        try:
            from ragas.metrics.collections import Faithfulness

            self.metric = Faithfulness(llm=llm)
            self.api_generation = "collections"
        except ImportError:
            from ragas.metrics import Faithfulness

            self.metric = Faithfulness(llm=llm)
            self.api_generation = "legacy"
        self.name = f"ragas-faithfulness:{self.version}"

    async def _ascore(self, example: BenchmarkExample) -> float:
        if self.api_generation == "collections":
            result = await self.metric.ascore(
                user_input=example.query,
                response=example.response,
                retrieved_contexts=list(example.contexts),
            )
            return float(result.value)
        from ragas.dataset_schema import SingleTurnSample

        sample = SingleTurnSample(
            user_input=example.query,
            response=example.response,
            retrieved_contexts=list(example.contexts),
        )
        return float(await self.metric.single_turn_ascore(sample))

    def score(self, example: BenchmarkExample) -> ScoreRecord:
        started = time.perf_counter()
        try:
            faithfulness = asyncio.run(self._ascore(example))
            if not 0.0 <= faithfulness <= 1.0:
                raise ValueError(f"Ragas returned out-of-range score {faithfulness}")
            return ScoreRecord(
                detector=self.name,
                example_id=example.id,
                group_id=example.group_id,
                split=example.split,
                hallucinated=example.hallucinated,
                score=1.0 - faithfulness,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                usage={"ragas_api": self.api_generation},
                metadata=example.metadata,
            )
        except Exception as exc:
            return ScoreRecord(
                detector=self.name,
                example_id=example.id,
                group_id=example.group_id,
                split=example.split,
                hallucinated=example.hallucinated,
                score=None,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
                metadata=example.metadata,
            )
