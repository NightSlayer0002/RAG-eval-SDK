# RAG Evaluation SDK

An experimental, pip-installable Python SDK for evidence-grounded RAG hallucination evaluation. The evaluation core is provider independent and uses observable text, without access to model internals.

The SDK combines offset-traceable claim decomposition, bounded evidence mapping, local HHEM verification, intervention-based audits, and latency/token/truncation telemetry. Precise hallucination boundaries and post-repair certification remain research directions.

## Evaluated results

The frozen Step 1 experiment used 80 fit, 20 calibration, and 100 held-out RAGTruth responses with source-group-disjoint partitions.

| Detector | Held-out F1 | Coverage |
|---|---:|---:|
| Compact multi-resolution SDK + HHEM | 0.7708 | 100% |
| Real Ragas Faithfulness 0.4.3 / Gemini 3.5 Flash-Lite | 0.7473 | 100% |
| Whole-response HHEM control | 0.7089 | 100% |

The SDK-minus-Ragas F1 difference is +0.0236, with paired bootstrap 95% CI [-0.0790, 0.1350]. **The difference is statistically inconclusive.** This is one held-out cohort, not a validated superiority claim. See the [frozen Step 1 report](FROZEN_STEP1_REPORT.md).

Step 2 tested span localization on a separate 100-calibration/300-test cohort, excluding Step 1 source groups. Whole-clause MREG projection scored 0.2602 micro character F1 versus 0.2710 for raw clause HHEM; the paired difference was negative, with 95% CI [-0.0175, -0.0046]. That approach was stopped without test-set retuning. See the [frozen Step 2 report](FROZEN_STEP2_REPORT.md).

The earlier “44% better than RAGAS” claim was invalid: it compared flag counts against a custom similarity heuristic, not the Ragas package. The [historical audit](RESULTS_AUDIT.md) preserves that correction.

## Install

Python 3.10 or newer:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[dev]"
```

The base package requires NumPy. Importing it does not download a model. Optional capabilities are installed separately:

```bash
python -m pip install -e ".[hhem]"       # Local HHEM verification
python -m pip install -e ".[local]"      # Sentence-transformer evidence similarity
python -m pip install -e ".[benchmark]"  # Real Ragas and benchmark dependencies
python -m pip install -e ".[google]"     # Google generation adapter
```

## Offline example

```python
from rag_eval_sdk import HashingSimilarityBackend, RAGEvaluator

evaluator = RAGEvaluator(similarity_backend=HashingSimilarityBackend())
report = evaluator.evaluate(
    response="The library is open Monday through Friday.",
    chunks=[
        {"id": "hours", "text": "The library is open Monday through Friday."},
        {"id": "weekends", "text": "The library is closed on weekends."},
    ],
)

print(report.decision.value)
print(report.grounding_risk)
print(report.usage.estimated_input_tokens)
print(report.to_dict())
```

This default hashing/heuristic configuration is an offline baseline. It is not the HHEM configuration that produced the frozen results.

## How evaluation works

1. Split the response into sentence or clause units with exact character offsets.
2. Map each unit to a bounded number of evidence chunks.
3. Verify whole-response and claim/evidence pairs, retaining support, contradiction, and uncertainty.
4. Audit selected claims by removing evidence or mutating a shared number, entity, or negation.
5. Optionally escalate uncertain claims through an injected verifier within a hard budget.
6. Return decisions, evidence IDs, reasons, intervention outcomes, and resource telemetry.

The compact learned risk model combines whole-response risk and worst-claim risk. Intervention probes are diagnostic outputs; they were not learned features in the frozen compact model.

`HHEMPairVerifier`, `CachedPairVerifier`, and `SentenceTransformerBackend` provide optional local components. `OpenAICompatibleProvider`, `GoogleGenAIProvider`, and `LLMPairVerifier` provide generation and escalation adapters. See [the reproduction guide](REPRODUCING.md) for model pins and commands.

## Benchmarks and tests

```bash
python -m pytest -q
python -m benchmarks.ragtruth_benchmark --help
python -m benchmarks.span_evaluation --help
python -m benchmarks.rgb_official --help
```

RAGTruth supplies human hallucination annotations. The benchmark implements source-group leakage checks, train-only calibration, durable JSONL checkpoints, paired statistics, coverage/failure reporting, and replay audits. RGB is an optional generator robustness experiment, not an equivalent hallucination-detector benchmark.

Frozen predictions, manifests, fitted calibration, summaries, and integrity reports are included under [benchmarks/results](benchmarks/results). Raw datasets, model weights, generation caches, and temporary runs are excluded. **Replay requires the original checksum-matching RAGTruth files, but no model inference or API calls.** Follow [REPRODUCING.md](REPRODUCING.md).

## Repository layout

```text
rag_eval_sdk/       SDK, provider adapters, verification and diagnostics
benchmarks/         RAGTruth/RGB runners, statistics and replay audits
  results/          Frozen Step 1 and Step 2 evidence bundles
tests/              Offline regression tests
data/               Small fictional CLI input examples
FROZEN_STEP*.md      Completed experiments and limitations
RESEARCH_PROTOCOL.md
STEP2_PROTOCOL.md   Preserved research protocols
NOVELTY_LEDGER.md   Unvalidated proposals and prior-art notes
REPRODUCING.md       Setup, replay and packaging instructions
VALIDATION_STATUS.md
```

Legacy API exports remain for compatibility and are covered by tests. The duplicate `src/` implementation, obsolete benchmark scripts/charts, personal study notes, environments, and secrets are excluded from this GitHub edition.

## Limitations and research status

- Deterministic clauses are not guaranteed to be atomic semantic claims.
- Evidence similarity mapping does not establish causal attribution or reveal model attention.
- Intervention-based audits test verifier behavior; they do not certify truth.
- Failure-stage diagnoses require appropriate pipeline observations and are not validated causal guarantees.
- Exact boundary localization is unresolved; the tested whole-clause approach failed.
- Post-repair certification is proposed research, not an implemented or validated capability.
- Remote generation caches store responses in plaintext; use them only with suitable data.

New experiments must use fresh source groups. Frozen cohorts must not be reused to validate changes informed by their test labels. Research hypotheses and remaining checks are recorded in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) and [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

## License and datasets

SDK code is [MIT licensed](LICENSE). Dataset and model licenses are separate. Obtain [RAGTruth](https://github.com/ParticleMedia/RAGTruth) and [RGB](https://github.com/chen700564/RGB) from their original repositories and follow their terms. Dataset/model files are not redistributed here.

