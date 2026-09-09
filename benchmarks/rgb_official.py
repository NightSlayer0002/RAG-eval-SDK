"""Reproducible English RGB generation benchmark with cached provider calls.

This measures the answer-generating system under RGB interventions.  It is not
used as the primary labelled benchmark for hallucination-detector accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from rag_eval_sdk.providers import (
    DiskCachedGenerator,
    GoogleGenAIProvider,
    OpenAICompatibleProvider,
    TextGenerator,
)


_SYSTEM_PROMPT = (
    "You are an accurate and reliable AI assistant that can answer questions with "
    "the help of external documents. External documents may contain noisy or "
    "factually incorrect information. If the documents contain the correct answer, "
    "give an accurate answer. If they do not contain the answer, say: 'I can not "
    "answer the question because of the insufficient information in documents.' "
    "If some documents contain factual errors, first say: 'There are factual errors "
    "in the provided documents.' and then provide the correct answer."
)


class InputBudgetExceededError(ValueError):
    """Raised before a provider call when a prompt exceeds the declared budget."""


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _as_text_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"RGB field {field!r} must be a list of strings")
    return list(value)


def build_rgb_documents(
    instance: Mapping[str, Any],
    *,
    dataset_name: str,
    noise_rate: float,
    passage_count: int,
    correct_rate: float,
    seed: int,
) -> list[str]:
    """Reproduce RGB's document construction without mutating dataset rows."""

    if not 0.0 <= noise_rate <= 1.0 or not 0.0 <= correct_rate <= 1.0:
        raise ValueError("noise_rate and correct_rate must be within [0, 1]")
    if dataset_name.endswith("_fact") and noise_rate + correct_rate > 1.0:
        raise ValueError("counterfactual noise_rate + correct_rate cannot exceed 1")
    if passage_count < 0:
        raise ValueError("passage_count cannot be negative")
    rng = random.Random(seed)
    negative = _as_text_list(instance.get("negative", []), "negative")
    negative_count = math.ceil(passage_count * noise_rate)
    positive_count = passage_count - negative_count

    if dataset_name.endswith("_int"):
        raw_groups = instance.get("positive")
        if not isinstance(raw_groups, list) or not all(isinstance(group, list) for group in raw_groups):
            raise ValueError("RGB integration positive must be a list of passage groups")
        groups: list[list[str]] = []
        for group in raw_groups:
            if not all(isinstance(item, str) for item in group):
                raise ValueError("RGB integration passage groups must contain strings")
            copy = list(group)
            rng.shuffle(copy)
            if copy:
                groups.append(copy)
        documents = [group[0] for group in groups]
        if len(documents) < positive_count and groups:
            for offset in range(1, max(map(len, groups))):
                for group in groups:
                    if len(group) > offset:
                        documents.append(group[offset])
                    if len(documents) == positive_count:
                        break
                if len(documents) == positive_count:
                    break
        documents = documents[:positive_count]
        documents.extend(negative[: max(0, passage_count - len(documents))])
    elif dataset_name.endswith("_fact"):
        positive = _as_text_list(instance.get("positive", []), "positive")
        wrong = _as_text_list(instance.get("positive_wrong", []), "positive_wrong")
        if len(wrong) < len(positive):
            raise ValueError("RGB counterfactual positive_wrong must align with positive")
        correct_count = math.ceil(passage_count * correct_rate)
        wrong_count = max(0, passage_count - negative_count - correct_count)
        indices = list(range(len(positive)))
        selected_wrong = rng.sample(indices, min(len(indices), wrong_count))
        documents = [wrong[index] for index in selected_wrong]
        remaining = [index for index in indices if index not in selected_wrong]
        if correct_count and remaining:
            selected_correct = rng.sample(remaining, min(len(remaining), correct_count))
            documents.extend(positive[index] for index in selected_correct)
        documents.extend(negative[:negative_count])
    else:
        positive = _as_text_list(instance.get("positive", []), "positive")
        if noise_rate == 1.0:
            negative_count = passage_count
            positive_count = 0
        elif negative_count > len(negative):
            negative_count = len(negative)
            positive_count = passage_count - negative_count
        elif positive_count > len(positive):
            positive_count = len(positive)
            negative_count = passage_count - positive_count
        documents = positive[:positive_count] + negative[:negative_count]
    rng.shuffle(documents)
    return documents


def _answer_components(answer: object) -> list[list[str]]:
    if isinstance(answer, str):
        return [[answer]]
    if not isinstance(answer, list):
        raise ValueError("RGB answer must be a string or list")
    components: list[list[str]] = []
    for component in answer:
        if isinstance(component, str):
            components.append([component])
        elif isinstance(component, list) and all(isinstance(item, str) for item in component):
            components.append(list(component))
        else:
            raise ValueError("RGB answer alternatives have an unsupported shape")
    return components


def score_rgb_prediction(prediction: str, answer: object) -> dict[str, object]:
    folded = prediction.casefold()
    components = _answer_components(answer)
    component_matches = [
        any(alternative.casefold() in folded for alternative in alternatives)
        for alternatives in components
    ]
    rejected = "insufficient information" in folded
    factual_error_detected = "factual errors" in folded
    return {
        "component_matches": component_matches,
        "answer_correct": bool(component_matches) and all(component_matches) and not rejected,
        "rejected": rejected,
        "factual_error_detected": factual_error_detected,
    }


def _provider(args: argparse.Namespace) -> TextGenerator:
    if args.provider == "google":
        generator: TextGenerator = GoogleGenAIProvider(
            model=args.model,
            api_key_env=args.api_key_env or "GEMINI_API_KEY",
            requests_per_minute=args.requests_per_minute,
        )
    else:
        preset_urls = {
            "nvidia": "https://integrate.api.nvidia.com/v1",
            "groq": "https://api.groq.com/openai/v1",
        }
        base_url = preset_urls.get(args.provider, args.base_url)
        if not base_url:
            raise ValueError("--base-url is required for a generic OpenAI-compatible provider")
        preset_key_environments = {
            "nvidia": "NVIDIA_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        generator = OpenAICompatibleProvider(
            base_url=base_url,
            model=args.model,
            api_key_env=args.api_key_env
            or preset_key_environments.get(args.provider, "OPENAI_API_KEY"),
            requests_per_minute=args.requests_per_minute,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
        )
    return DiskCachedGenerator(generator, args.cache_dir)


def _condition_id(
    args: argparse.Namespace, dataset_name: str, dataset_sha256: str
) -> str:
    payload = {
        "dataset": dataset_name,
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "max_input_characters": args.max_input_characters,
        "max_output_tokens": args.max_output_tokens,
        "noise_rate": args.noise_rate,
        "correct_rate": args.correct_rate,
        "passage_count": args.passage_count,
        "seed": args.seed,
        "prompt_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "dataset_sha256": dataset_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one reproducible English RGB condition")
    parser.add_argument("--dataset", required=True, help="Path to en_refine.json, en_int.json, or en_fact.json")
    parser.add_argument(
        "--provider", choices=("nvidia", "groq", "openai-compatible", "google")
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--requests-per-minute", type=float)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--cache-dir", default="benchmarks/cache/generation")
    parser.add_argument("--output-dir", default="benchmarks/results/rgb_v2")
    parser.add_argument("--noise-rate", type=float, default=0.0)
    parser.add_argument("--correct-rate", type=float, default=0.0)
    parser.add_argument("--passage-count", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-input-characters", type=int, default=12000)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-retry-failures", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate every row and selected document condition without calling a model",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.stem
    if dataset_name not in {"en_refine", "en_int", "en_fact"}:
        raise ValueError("This runner supports en_refine, en_int, and en_fact")
    rows = list(_read_jsonl(dataset_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.validate_only:
        failures: list[str] = []
        for row in rows:
            try:
                _answer_components(row.get("answer"))
                documents = build_rgb_documents(
                    row,
                    dataset_name=dataset_name,
                    noise_rate=args.noise_rate,
                    passage_count=args.passage_count,
                    correct_rate=args.correct_rate,
                    seed=args.seed,
                )
                document_text = "\n".join(documents)
                user_prompt = f"Document:\n{document_text}\n\nQuestion:\n{row['query']}"
                if len(_SYSTEM_PROMPT) + len(user_prompt) > args.max_input_characters:
                    raise InputBudgetExceededError(
                        "constructed prompt exceeds --max-input-characters"
                    )
            except Exception as exc:
                failures.append(f"id={row.get('id')}: {type(exc).__name__}: {exc}")
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "rows_checked": len(rows),
                    "validation_failures": len(failures),
                    "first_failures": failures[:20],
                    "noise_rate": args.noise_rate,
                    "correct_rate": args.correct_rate,
                    "passage_count": args.passage_count,
                    "max_input_characters": args.max_input_characters,
                },
                indent=2,
            )
        )
        if failures:
            raise RuntimeError(f"RGB schema validation failed for {len(failures)} row(s)")
        return
    if not args.provider or not args.model:
        raise ValueError("--provider and --model are required unless --validate-only is used")
    generator = _provider(args)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    condition = _condition_id(args, dataset_name, dataset_sha256)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"rgb_{dataset_name}_{condition}.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists() and not args.no_resume:
        existing = {str(row["id"]): row for row in _read_jsonl(output_path)}
    mode = "a" if existing and not args.no_resume else "w"
    results_by_id = dict(existing)
    with output_path.open(mode, encoding="utf-8") as handle:
        for position, row in enumerate(rows, 1):
            row_id = str(row.get("id"))
            if row_id in existing and (
                not existing[row_id].get("error") or args.no_retry_failures
            ):
                continue
            documents: list[str] = []
            prompt_characters = 0
            attempt_started = time.perf_counter()
            try:
                documents = build_rgb_documents(
                    row,
                    dataset_name=dataset_name,
                    noise_rate=args.noise_rate,
                    passage_count=args.passage_count,
                    correct_rate=args.correct_rate,
                    seed=args.seed,
                )
                document_text = "\n".join(documents)
                user_prompt = f"Document:\n{document_text}\n\nQuestion:\n{row['query']}"
                prompt_characters = len(_SYSTEM_PROMPT) + len(user_prompt)
                if prompt_characters > args.max_input_characters:
                    raise InputBudgetExceededError(
                        f"prompt has {prompt_characters} characters; budget is "
                        f"{args.max_input_characters}"
                    )
                generated = generator.generate(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                )
                scoring = score_rgb_prediction(generated.text, row.get("answer"))
                result = {
                    "id": row_id,
                    "query": row.get("query"),
                    "answer": row.get("answer"),
                    "documents": documents,
                    "prediction": generated.text,
                    "scoring": scoring,
                    "generation": asdict(generated),
                    "run_latency_seconds": time.perf_counter() - attempt_started,
                    "prompt_characters": prompt_characters,
                    "estimated_input_tokens": (prompt_characters + 3) // 4,
                    "error": None,
                }
            except Exception as exc:
                result = {
                    "id": row_id,
                    "query": row.get("query"),
                    "answer": row.get("answer"),
                    "documents": documents,
                    "prediction": None,
                    "scoring": None,
                    "generation": None,
                    "run_latency_seconds": time.perf_counter() - attempt_started,
                    "prompt_characters": prompt_characters,
                    "estimated_input_tokens": (prompt_characters + 3) // 4,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            results_by_id[row_id] = result
            if position % 20 == 0 or position == len(rows):
                print(f"[{dataset_name}:{condition}] {position}/{len(rows)}")

    target_ids = {str(row.get("id")) for row in rows}
    results = [
        result for row_id, result in results_by_id.items() if row_id in target_ids
    ]
    successful = [result for result in results if not result["error"]]
    errors = [result for result in results if result["error"]]
    correct = sum(bool(result["scoring"]["answer_correct"]) for result in successful)
    rejected = sum(bool(result["scoring"]["rejected"]) for result in successful)
    detected = sum(
        bool(result["scoring"]["factual_error_detected"]) for result in successful
    )
    detected_and_correct = sum(
        bool(result["scoring"]["factual_error_detected"])
        and bool(result["scoring"]["answer_correct"])
        for result in successful
    )
    denominator = len(results)
    run_latencies = [float(result.get("run_latency_seconds", 0.0)) for result in results]
    sorted_latencies = sorted(run_latencies)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    generation_rows = [result["generation"] for result in successful]
    reported_input_tokens = [
        int(generation["input_tokens"])
        for generation in generation_rows
        if generation.get("input_tokens") is not None
    ]
    reported_output_tokens = [
        int(generation["output_tokens"])
        for generation in generation_rows
        if generation.get("output_tokens") is not None
    ]
    cache_hits = sum(bool(generation.get("cached")) for generation in generation_rows)
    summary = {
        "protocol": "rgb-official-shape-compatible-v1",
        "condition_id": condition,
        "dataset": dataset_name,
        "model": args.model,
        "provider": args.provider,
        "noise_rate": args.noise_rate,
        "correct_rate_input": args.correct_rate,
        "passage_count": args.passage_count,
        "max_input_characters": args.max_input_characters,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
        "dataset_sha256": dataset_sha256,
        "total_examples": denominator,
        "successful_examples": len(successful),
        "coverage": len(successful) / denominator if denominator else 0.0,
        "generation_failures": len(errors),
        "failure_types": dict(Counter(str(result["error"]).split(":", 1)[0] for result in errors)),
        "accuracy_successful_only": correct / len(successful) if successful else 0.0,
        "accuracy_failure_as_incorrect": correct / denominator if denominator else 0.0,
        "rejection_rate_successful_only": rejected / len(successful) if successful else 0.0,
        "error_detection_rate_successful_only": detected / len(successful) if successful else 0.0,
        "correction_given_detection": detected_and_correct / detected if detected else 0.0,
        "resources": {
            "run_latency_seconds_mean": (
                statistics.fmean(run_latencies) if run_latencies else 0.0
            ),
            "run_latency_seconds_p95": (
                sorted_latencies[p95_index] if sorted_latencies else 0.0
            ),
            "estimated_input_tokens_total": sum(
                int(result.get("estimated_input_tokens", 0)) for result in results
            ),
            "provider_reported_input_tokens_total": sum(reported_input_tokens),
            "provider_reported_input_token_examples": len(reported_input_tokens),
            "provider_reported_output_tokens_total": sum(reported_output_tokens),
            "provider_reported_output_token_examples": len(reported_output_tokens),
            "cache_hits": cache_hits,
            "new_remote_calls": len(successful) - cache_hits,
        },
        "interpretation": (
            "At noise_rate=1, use rejection rate. For en_fact, use error detection and "
            "correction-given-detection. Otherwise use answer accuracy."
        ),
        "method_note": (
            "Document sampling and substring answer matching reproduce the public RGB script's "
            "core protocol. Failures are retained and also reported as incorrect."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
