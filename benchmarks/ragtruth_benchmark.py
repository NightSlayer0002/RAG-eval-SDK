"""Command-line RAGTruth hallucination-detection benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from rag_eval_sdk import (
    CachedPairVerifier,
    HHEMPairVerifier,
    HashingSimilarityBackend,
    HeuristicPairVerifier,
    RAGEvaluator,
    SentenceTransformerBackend,
    VerificationConfig,
    __version__,
)
from rag_eval_sdk.risk_model import (
    MREG_COMPACT_FEATURE_NAMES,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_SCHEMA,
)

from .comparison import compare_detectors
from .datasets import load_ragtruth
from .detectors import RagasFaithfulnessDetector, SDKDetector, WholeResponsePairDetector
from .runner import (
    apply_train_only_risk_calibration,
    save_summary,
    score_examples,
    summarize_detector,
)
from .schema import BenchmarkExample


def _safe_file_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:100]


def _device_argument(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_excluded_groups(paths: list[str]) -> set[str]:
    """Read prior score checkpoints and return their source groups."""

    groups: set[str] = set()
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Excluded-group score file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                group_id = row.get("group_id") if isinstance(row, dict) else None
                if group_id is None:
                    raise ValueError(f"Missing group_id at {path}:{line_number}")
                groups.add(str(group_id))
    return groups


def _stable_limit(
    examples: list[BenchmarkExample],
    validation_limit: int | None,
    test_limit: int | None,
    validation_offset: int = 0,
    test_offset: int = 0,
) -> list[BenchmarkExample]:
    selected: list[BenchmarkExample] = []
    for split, limit, offset in (
        ("train", validation_limit, validation_offset),
        ("test", test_limit, test_offset),
    ):
        candidates = [example for example in examples if example.split == split]
        candidates.sort(
            key=lambda example: hashlib.sha256(example.id.encode("utf-8")).digest()
        )
        remaining = candidates[offset:]
        selected.extend(remaining if limit is None else remaining[:limit])
    return selected


def _partition_fit_split(
    examples: list[BenchmarkExample], fit_fraction: float
) -> list[BenchmarkExample]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("--risk-fit-fraction must be strictly between 0 and 1")
    train_examples = [example for example in examples if example.split == "train"]
    groups = sorted({example.group_id for example in train_examples})
    if len(groups) < 2:
        raise ValueError("learned risk calibration requires at least two train groups")
    fit_count = min(len(groups) - 1, max(1, round(len(groups) * fit_fraction)))
    fit_groups: set[str] | None = None
    # The split is group-safe and train-label-stratified. Trying deterministic
    # salts prevents small pilots from accidentally producing a one-class fit
    # or calibration partition without ever consulting the held-out test data.
    for salt in range(256):
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"risk-fit-v1:{salt}:{group}".encode("utf-8")
            ).digest(),
        )
        candidate = set(ordered[:fit_count])
        fit_labels = {
            example.hallucinated
            for example in train_examples
            if example.group_id in candidate
        }
        calibration_labels = {
            example.hallucinated
            for example in train_examples
            if example.group_id not in candidate
        }
        if len(fit_labels) == 2 and len(calibration_labels) == 2:
            fit_groups = candidate
            break
    if fit_groups is None:
        raise ValueError(
            "could not create group-safe fit and calibration partitions containing "
            "both labels; increase --validation-limit or disable learned risk"
        )
    return [
        replace(example, split="fit")
        if example.split == "train" and example.group_id in fit_groups
        else example
        for example in examples
    ]


def _sdk_detectors(
    args: argparse.Namespace, *, include_sdk: bool, include_ablation: bool
) -> list[object]:
    if args.similarity == "hashing":
        similarity = HashingSimilarityBackend(dimensions=args.hash_dimensions)
    else:
        similarity = SentenceTransformerBackend(
            model_name=args.embedding_model,
            device=args.embedding_device,
            cache_entries=args.embedding_cache_entries,
            batch_size=args.embedding_batch_size,
        )
    if args.verifier == "heuristic":
        base_verifier = HeuristicPairVerifier(similarity)
    else:
        if not args.trust_remote_code:
            raise ValueError(
                "The selected HHEM repository requires custom model code. Review the "
                "repository, then pass --trust-remote-code explicitly."
            )
        base_verifier = HHEMPairVerifier(
            model_name=args.hhem_model,
            tokenizer_name=args.hhem_tokenizer,
            model_revision=args.hhem_revision,
            tokenizer_revision=args.hhem_tokenizer_revision,
            device=args.hhem_device,
            batch_size=args.hhem_batch_size,
            trust_remote_code=args.trust_remote_code,
        )
    sdk_verifier = CachedPairVerifier(
        base_verifier, max_entries=args.verifier_cache_entries
    )
    config = VerificationConfig(
        max_claims=args.max_claims,
        claim_granularity=args.claim_granularity,
        max_response_characters=args.max_response_characters,
        max_input_chunks=args.max_input_chunks,
        max_chunk_characters=args.max_chunk_characters,
        evidence_per_claim=args.evidence_per_claim,
        evidence_strategy=args.evidence_strategy,
        max_evidence_chars_per_claim=args.max_evidence_characters,
        global_pair_enabled=not args.no_global_pair,
        max_audit_claims=args.max_audit_claims,
        audit_enabled=not args.no_audit,
        support_threshold=args.support_threshold,
        contradiction_threshold=args.contradiction_threshold,
        uncertainty_margin=args.uncertainty_margin,
        min_necessity_drop=args.min_necessity_drop,
        min_mutation_drop=args.min_mutation_drop,
        max_order_shift=args.max_order_shift,
        max_escalations=0,
        calibration_id="decision-defaults-not-used-for-benchmark-threshold",
    )
    detectors: list[object] = []
    settings = {
        "similarity": args.similarity,
        "hash_dimensions": args.hash_dimensions,
        "embedding_model": args.embedding_model,
        "verifier": args.verifier,
        "hhem_model": args.hhem_model,
        "hhem_revision": args.hhem_revision,
        "hhem_tokenizer": args.hhem_tokenizer,
        "hhem_tokenizer_revision": args.hhem_tokenizer_revision,
        "hhem_batch_size": args.hhem_batch_size,
        "max_claims": args.max_claims,
        "claim_granularity": args.claim_granularity,
        "max_response_characters": args.max_response_characters,
        "max_input_chunks": args.max_input_chunks,
        "max_chunk_characters": args.max_chunk_characters,
        "evidence_per_claim": args.evidence_per_claim,
        "evidence_strategy": args.evidence_strategy,
        "max_evidence_characters": args.max_evidence_characters,
        "global_pair_enabled": not args.no_global_pair,
        "max_audit_claims": args.max_audit_claims,
        "audit_enabled": not args.no_audit,
        "support_threshold": args.support_threshold,
        "contradiction_threshold": args.contradiction_threshold,
        "uncertainty_margin": args.uncertainty_margin,
        "min_necessity_drop": args.min_necessity_drop,
        "min_mutation_drop": args.min_mutation_drop,
        "max_order_shift": args.max_order_shift,
        "risk_feature_schema": RISK_FEATURE_SCHEMA,
        "risk_calibrator": args.risk_calibrator,
        "risk_fit_fraction": args.risk_fit_fraction,
        "risk_l2": args.risk_l2,
        "risk_features": args.risk_features,
        "risk_oof_folds": args.risk_oof_folds,
    }
    settings_digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    if include_sdk:
        evaluator = RAGEvaluator(
            config=config,
            similarity_backend=similarity,
            pair_verifier=sdk_verifier,
        )
        name = f"rag-eval-sdk-{__version__}:{args.similarity}:{args.verifier}:{settings_digest}"
        detectors.append(SDKDetector(evaluator, name=name))
    if include_ablation:
        detectors.append(
            WholeResponsePairDetector(
                CachedPairVerifier(
                    base_verifier, max_entries=args.verifier_cache_entries
                ),
                max_evidence_characters=args.max_evidence_characters,
                name=f"whole-response-ablation:{args.verifier}:{settings_digest}",
            )
        )
    return detectors


def _ragas_detector(args: argparse.Namespace) -> RagasFaithfulnessDetector:
    if not args.ragas_model:
        raise ValueError("--ragas-model is required for the Ragas detector")
    key = os.getenv(args.ragas_api_key_env)
    if not key:
        raise RuntimeError(f"Set {args.ragas_api_key_env} before running Ragas")
    try:
        from openai import AsyncOpenAI
        from ragas.cache import DiskCacheBackend
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError("Install rag-eval-sdk[benchmark] for the Ragas detector") from exc
    client_kwargs: dict[str, object] = {
        "api_key": key,
        "max_retries": args.ragas_max_retries,
        "timeout": args.ragas_timeout_seconds,
    }
    if args.ragas_base_url:
        client_kwargs["base_url"] = args.ragas_base_url
    client = AsyncOpenAI(**client_kwargs)
    cache = None
    if not args.no_ragas_cache:
        cache = DiskCacheBackend(
            cache_dir=str(Path(args.output_dir) / "ragas_llm_cache")
        )
    llm = llm_factory(
        args.ragas_model,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=args.ragas_max_tokens,
        cache=cache,
    )
    detector = RagasFaithfulnessDetector(llm)
    endpoint_identity = hashlib.sha256(
        (
            f"{args.ragas_model}|{args.ragas_base_url or 'provider-default'}|"
            f"retries={args.ragas_max_retries}|timeout={args.ragas_timeout_seconds}|"
            f"max_tokens={args.ragas_max_tokens}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    detector.name = f"{detector.name}:{args.ragas_model}:{endpoint_identity}"
    return detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Held-out RAGTruth comparison with explicit failures and resource reporting"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="benchmarks/results/ragtruth_v2")
    parser.add_argument(
        "--detector",
        choices=("sdk", "verifier", "local", "ragas", "both", "all"),
        default="sdk",
        help=(
            "sdk or verifier alone; local compares SDK with the identical base "
            "verifier; both compares SDK with Ragas; all runs every detector"
        ),
    )
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--validation-offset", type=int, default=0)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument(
        "--exclude-groups-from",
        action="append",
        default=[],
        metavar="SCORES_JSONL",
        help=(
            "Exclude every source group present in a prior score checkpoint. "
            "Repeat this option to construct a contamination-resistant new cohort."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-retry-failures", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--objective", choices=("f1", "balanced_accuracy"), default="f1")

    parser.add_argument("--similarity", choices=("hashing", "sentence-transformer"), default="hashing")
    parser.add_argument("--hash-dimensions", type=int, default=2048)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device")
    parser.add_argument("--embedding-cache-entries", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--verifier", choices=("heuristic", "hhem"), default="heuristic")
    parser.add_argument("--hhem-model", default="vectara/hallucination_evaluation_model")
    parser.add_argument(
        "--hhem-revision",
        default="8e4a2e6e96c708cc76c2344f7e4757df2515292c",
        help="Reviewed immutable HHEM repository revision",
    )
    parser.add_argument("--hhem-tokenizer", default="google/flan-t5-base")
    parser.add_argument(
        "--hhem-tokenizer-revision",
        default="7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    )
    parser.add_argument("--hhem-device", type=_device_argument, default=-1)
    parser.add_argument("--hhem-batch-size", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verifier-cache-entries", type=int, default=8192)
    parser.add_argument("--max-claims", type=int, default=24)
    parser.add_argument(
        "--claim-granularity", choices=("sentence", "clause"), default="clause"
    )
    parser.add_argument("--max-response-characters", type=int, default=12000)
    parser.add_argument("--max-input-chunks", type=int, default=64)
    parser.add_argument("--max-chunk-characters", type=int, default=6000)
    parser.add_argument("--evidence-per-claim", type=int, default=3)
    parser.add_argument(
        "--evidence-strategy",
        choices=("semantic", "coverage_aware"),
        default="coverage_aware",
    )
    parser.add_argument("--max-evidence-characters", type=int, default=6000)
    parser.add_argument("--no-global-pair", action="store_true")
    parser.add_argument("--max-audit-claims", type=int, default=2)
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--support-threshold", type=float, default=0.70)
    parser.add_argument("--contradiction-threshold", type=float, default=0.55)
    parser.add_argument("--uncertainty-margin", type=float, default=0.10)
    parser.add_argument("--min-necessity-drop", type=float, default=0.08)
    parser.add_argument("--min-mutation-drop", type=float, default=0.06)
    parser.add_argument("--max-order-shift", type=float, default=0.12)
    parser.add_argument(
        "--risk-calibrator", choices=("threshold", "logistic"), default="logistic"
    )
    parser.add_argument("--risk-fit-fraction", type=float, default=0.8)
    parser.add_argument("--risk-l2", type=float, default=0.10)
    parser.add_argument(
        "--risk-features",
        choices=("compact", "all"),
        default="compact",
        help="Compact is the lower-variance two-feature MREG calibrator.",
    )
    parser.add_argument("--risk-oof-folds", type=int, default=5)

    parser.add_argument("--ragas-model")
    parser.add_argument("--ragas-base-url")
    parser.add_argument("--ragas-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--ragas-max-retries", type=int, default=5)
    parser.add_argument("--ragas-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--ragas-max-tokens", type=int, default=4096)
    parser.add_argument("--no-ragas-cache", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_examples = list(load_ragtruth(args.data_dir))
    excluded_groups = _load_excluded_groups(args.exclude_groups_from)
    if excluded_groups:
        all_examples = [
            example for example in all_examples if example.group_id not in excluded_groups
        ]
    examples = _stable_limit(
        all_examples,
        args.validation_limit,
        args.test_limit,
        args.validation_offset,
        args.test_offset,
    )
    if args.risk_calibrator == "logistic":
        examples = _partition_fit_split(examples, args.risk_fit_fraction)
    if not examples:
        raise RuntimeError("No RAGTruth examples matched the requested settings")

    detectors: list[object] = []
    if args.detector in {"sdk", "verifier", "local", "both", "all"}:
        detectors.extend(
            _sdk_detectors(
                args,
                include_sdk=args.detector in {"sdk", "local", "both", "all"},
                include_ablation=args.detector in {"verifier", "local", "all"},
            )
        )
    if args.detector in {"ragas", "both", "all"}:
        detectors.append(_ragas_detector(args))
    data_root = Path(args.data_dir)
    dataset_checksums = {
        name: _sha256_file(data_root / name)
        for name in ("source_info.jsonl", "response.jsonl")
    }
    exclusion_files = {
        str(Path(value).resolve()): _sha256_file(Path(value))
        for value in args.exclude_groups_from
    }
    excluded_groups_digest = hashlib.sha256(
        "\n".join(sorted(excluded_groups)).encode("utf-8")
    ).hexdigest()
    example_ids_digest = hashlib.sha256(
        "\n".join(example.id for example in examples).encode("utf-8")
    ).hexdigest()
    resume_signature = {
        "protocol": "ragtruth-held-out-v1",
        "dataset_checksums": dataset_checksums,
        "excluded_groups_sha256": excluded_groups_digest,
        "example_ids_sha256": example_ids_digest,
        "detectors": [detector.name for detector in detectors],
    }
    manifest = {
        "protocol": "ragtruth-held-out-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sdk_version": __version__,
        "python": platform.python_version(),
        "dataset_directory": str(Path(args.data_dir).resolve()),
        "examples": len(examples),
        "validation_examples": sum(example.split == "train" for example in examples),
        "risk_fit_examples": sum(example.split == "fit" for example in examples),
        "test_examples": sum(example.split == "test" for example in examples),
        "detectors": [detector.name for detector in detectors],
        "dataset_checksums": dataset_checksums,
        "exclusion_files": exclusion_files,
        "excluded_source_groups": len(excluded_groups),
        "excluded_groups_sha256": excluded_groups_digest,
        "example_ids_sha256": example_ids_digest,
        "resume_signature": resume_signature,
        "arguments": vars(args),
        "notes": [
            "Subset selection is a label-independent SHA-256 ordering of example IDs.",
            "Thresholds are selected on train and frozen before test evaluation.",
            "When learned risk is enabled, source groups are deterministically split into fit and calibration partitions before scoring.",
            "RAGTruth source_id is enforced as the leakage group.",
            "Any --exclude-groups-from checkpoints are hashed and excluded before subset selection.",
        ],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.no_resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("resume_signature") != resume_signature:
            raise RuntimeError(
                "Existing output is from a different dataset, subset, or detector configuration. "
                "Use a new --output-dir or pass --no-resume to replace it."
            )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    completed: list[tuple[str, list[object], dict[str, object]]] = []
    for detector in detectors:
        detector_examples = (
            examples
            if isinstance(detector, SDKDetector)
            else [example for example in examples if example.split != "fit"]
        )
        records, score_path = score_examples(
            detector_examples,
            detector,  # type: ignore[arg-type]
            output,
            resume=not args.no_resume,
            progress_every=args.progress_every,
            retry_failures=not args.no_retry_failures,
        )
        result_stem = score_path.stem.removeprefix("scores_")
        if args.risk_calibrator == "logistic" and isinstance(detector, SDKDetector):
            raw_summary = summarize_detector(records, objective=args.objective)
            raw_summary_path = output / f"summary_{result_stem}_raw.json"
            save_summary(raw_summary, raw_summary_path)

            learned_records, risk_manifest = apply_train_only_risk_calibration(
                records,
                l2=args.risk_l2,
                feature_names=(
                    MREG_COMPACT_FEATURE_NAMES
                    if args.risk_features == "compact"
                    else RISK_FEATURE_NAMES
                ),
                oof_folds=args.risk_oof_folds,
                objective=args.objective,
            )
            risk_path = output / f"risk_model_{score_path.stem.removeprefix('scores_')}.json"
            save_summary(risk_manifest, risk_path)
            learned_summary = summarize_detector(
                learned_records,
                objective=args.objective,
                fixed_threshold=float(risk_manifest["decision_threshold"]),
                fixed_threshold_metadata=risk_manifest["threshold_selection"],  # type: ignore[arg-type]
            )
            learned_summary_path = output / f"summary_{result_stem}_learned.json"
            save_summary(learned_summary, learned_summary_path)
            # Learned first makes it the primary system in paired comparisons;
            # raw SDK remains an explicit ablation using identical verifier calls.
            completed.append(
                (str(learned_summary["detector"]), learned_records, learned_summary)
            )
            completed.append((str(raw_summary["detector"]), records, raw_summary))
            print(f"Risk model saved: {risk_path}")
            print(f"Raw SDK summary saved: {raw_summary_path}")
            print(f"Learned SDK summary saved: {learned_summary_path}")
            continue
        summary = summarize_detector(records, objective=args.objective)
        summary_path = output / f"summary_{result_stem}.json"
        save_summary(summary, summary_path)
        completed.append((str(summary["detector"]), records, summary))
        print(f"Summary saved: {summary_path}")

    if len(completed) >= 2:
        first_name, first_records, first_summary = completed[0]
        for second_name, second_records, second_summary in completed[1:]:
            comparison = compare_detectors(
                first_records,  # type: ignore[arg-type]
                second_records,  # type: ignore[arg-type]
                first_threshold=float(first_summary["calibration"]["threshold"]),  # type: ignore[index]
                second_threshold=float(second_summary["calibration"]["threshold"]),  # type: ignore[index]
                bootstrap_samples=args.bootstrap_samples,
            )
            comparison_path = output / (
                f"paired_{_safe_file_name(first_name)}_vs_{_safe_file_name(second_name)}.json"
            )
            save_summary(comparison, comparison_path)
            print(f"Paired comparison saved: {comparison_path}")


if __name__ == "__main__":
    main()
