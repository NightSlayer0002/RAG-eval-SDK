"""CLI-facing end-to-end generation and verification pipeline."""

from __future__ import annotations

import os
from typing import Any

from .llm import generate_llm_response
from .pipeline import RAGEvaluator
from .providers import TextGenerator
from .utils import load_json, save_json


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_inputs(chat_path: str, context_path: str) -> tuple[str, list[dict[str, Any]]]:
    chat_data = load_json(chat_path)
    context_data = load_json(context_path)
    try:
        query = str(chat_data["messages"][-1]["content"])
        raw_chunks = context_data["data"]["vector_data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Expected chat.messages[-1].content and context.data.vector_data."
        ) from exc
    if not isinstance(raw_chunks, list):
        raise ValueError("context.data.vector_data must be a list")
    return query, raw_chunks


def run_evaluation_pipeline(
    chat_path: str | None = None,
    context_path: str | None = None,
    output_path: str | None = None,
    *,
    provider: TextGenerator | None = None,
    evaluator: RAGEvaluator | None = None,
) -> dict[str, object]:
    """Generate one response and save a v2 evidence-verification report."""

    chat_path = chat_path or os.path.join(_PROJECT_ROOT, "data", "chat.json")
    context_path = context_path or os.path.join(_PROJECT_ROOT, "data", "context.json")
    output_path = output_path or os.path.join(
        _PROJECT_ROOT, "output", "evaluation_report.json"
    )
    query, chunks = _read_inputs(chat_path, context_path)
    generation = generate_llm_response(query, chunks, provider=provider)
    selected_evaluator = evaluator or RAGEvaluator()
    verification = selected_evaluator.evaluate(str(generation["response"]), chunks)
    report: dict[str, object] = {
        "schema_version": "2.0",
        "query": query,
        "generation": generation,
        "verification": verification.to_dict(),
        "limitations": [
            "Sentence-level units are not guaranteed to be atomic logical claims.",
            "Uncalibrated defaults are operating points, not benchmark-derived guarantees.",
            "Evidence-sensitivity interventions audit verifier behavior; they do not prove truth.",
        ],
    }
    save_json(report, output_path)
    print(
        f"Verification: {verification.decision.value}; risk={verification.risk_score:.3f}; "
        f"claims={len(verification.claims)}; remote_calls={verification.usage.remote_calls}"
    )
    print(f"Report saved: {output_path}")
    return report


if __name__ == "__main__":
    run_evaluation_pipeline()
