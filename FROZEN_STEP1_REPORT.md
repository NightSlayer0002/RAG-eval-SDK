# Frozen Step-1 report: multi-resolution hallucination detection

Status: completed and frozen on 2026-08-19.

## Result in one sentence

On a fresh 100-example RAGTruth test cohort, the learned multi-resolution system produced the best point estimate of the three tested systems (F1 0.7708 versus real Ragas Faithfulness 0.7473 and whole-response HHEM 0.7089), but neither paired F1 improvement was statistically conclusive. This is promising evidence, not a defensible superiority claim.

## Frozen research question

Does combining a whole-response evidence score with the worst claim-level evidence score improve held-out RAG hallucination detection over the same verifier used once on the whole response, and over real Ragas Faithfulness?

The experimental score is called Multi-Resolution Evidence Geometry (MREG). Its compact learned layer has only two inputs:

1. `global_pair_risk`: HHEM risk for packed evidence versus the complete response.
2. `max_pair_risk`: the greatest HHEM risk among the selected claim/evidence pairs.

A deterministic logistic model combines the two. It has no access to model internals, test labels, dataset IDs, task-specific prediction constants, or result files. Evidence Reaction Audit probes are still produced for diagnostics, but the frozen compact risk model does not use them as learned features.

## Dataset and split

The benchmark used the official RAGTruth `source_info.jsonl` and `response.jsonl` files.

| Partition | Examples | Purpose |
|---|---:|---|
| Fit | 80 | Fit the two-feature logistic model and select its threshold by 5-fold group-safe out-of-fold predictions |
| Calibration | 20 | Report frozen-threshold calibration performance; select thresholds for the non-learned baselines |
| Test | 100 | One held-out comparison only |

The test cohort contains 44 hallucinated and 56 grounded responses: 44 data-to-text, 32 QA, and 24 summarization examples. Selection is a label-independent SHA-256 ordering with offsets of 50 for both train and test. `source_id` is the leakage group; fit, calibration, and test source groups do not overlap where the official split permits enforcement.

Dataset SHA-256 values:

- `source_info.jsonl`: `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b`
- `response.jsonl`: `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`
- Ordered selected-example digest: `709d1435cee55de9ae236e6e6a3aab2a40e2dfe056f08e36e57e12f23316a674`

## Exact systems

### Experimental MREG system

- SDK: `rag-eval-sdk 2.0.0a1`
- Similarity shortlist: deterministic 2,048-dimensional hashing backend
- Claim granularity: clause
- Maximum claims: 24
- Evidence selection: coverage-aware, three candidates per claim
- Whole-response evidence budget: 6,000 characters
- Claim evidence budget: 6,000 characters
- Audit budget: at most two claims
- Verifier: Vectara HHEM, CPU, batch size 8
- HHEM revision: `8e4a2e6e96c708cc76c2344f7e4757df2515292c`
- FLAN-T5 tokenizer revision: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`
- Logistic L2: 0.1
- Learned coefficients: 0.569466 for global risk and 0.491757 for maximum claim risk
- Learned intercept: -0.690209
- Decision threshold: 0.386306, selected on 80 group-safe out-of-fold fit predictions

### Whole-response control

The control uses the same loaded HHEM weights, the same 6,000-character packed evidence budget, and a separate result cache. It performs one evidence/response comparison per example. Its threshold, 0.435896, was selected on the 20-example calibration partition.

### Real Ragas baseline

- Ragas: 0.4.3
- Metric: the official `ragas.metrics.collections.Faithfulness`
- Judge: stable `gemini-3.5-flash-lite`
- Endpoint: Google's OpenAI-compatible Gemini endpoint
- Temperature: 0
- Maximum output tokens: 4,096
- Persistent disk cache: enabled
- Decision threshold: hallucination risk 0.097619, selected on the 20-example calibration partition

Ragas made two cached judge generations per evaluated response: 240 cache entries for 120 calibration-plus-test examples. The adapter did not retain provider token-usage fields, so an exact dollar total cannot be reconstructed. Google's current list price is $0.30 per million input tokens and $2.50 per million output/thinking tokens on the paid standard tier; the free tier is listed as free of charge. This is pricing context, not a measured cost for this run.

## Held-out results

Higher scores mean greater hallucination risk. All systems scored all 100 held-out examples.

| System | F1 | Precision | Recall | Accuracy | Balanced acc. | AUROC | AUPRC | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MREG learned | **0.7708** | 0.7115 | **0.8409** | **0.7800** | **0.7865** | **0.8551** | **0.7796** | 1.000 |
| Ragas Faithfulness | 0.7473 | 0.7234 | 0.7727 | 0.7700 | 0.7703 | 0.8373 | 0.7134 | 1.000 |
| Whole-response HHEM | 0.7089 | **0.8000** | 0.6364 | 0.7700 | 0.7557 | 0.8304 | 0.7653 | 1.000 |
| Raw SDK score | 0.7636 | 0.6364 | 0.9545 | 0.7400 | 0.7630 | 0.7821 | 0.6678 | 1.000 |

MREG is not uniformly better. It trades some precision for recall relative to whole-response HHEM. Ragas has slightly higher precision. The learned layer mainly improves calibration/ranking and reduces raw SDK false positives; its F1 improvement over the raw SDK score is small.

## Paired uncertainty

Paired bootstrap intervals use 5,000 deterministic resamples of the same 100 held-out IDs.

| Comparison | F1 difference | Bootstrap 95% CI | McNemar discordance | Exact p |
|---|---:|---:|---:|---:|
| MREG minus Ragas | +0.0236 | [-0.0790, 0.1350] | 16 MREG-only correct / 15 Ragas-only correct | 1.000 |
| MREG minus whole HHEM | +0.0620 | [-0.0333, 0.1638] | 9 MREG-only correct / 8 HHEM-only correct | 1.000 |
| MREG minus raw SDK | +0.0072 | [-0.0645, 0.0733] | 9 learned-only correct / 5 raw-only correct | 0.424 |

Every interval includes zero. The test therefore does not establish that MREG is better than Ragas or HHEM in the population. It establishes that the point estimate is competitive enough to justify replication.

## Task slices

| Task | n | Positives | MREG F1 | Ragas F1 | Whole HHEM F1 |
|---|---:|---:|---:|---:|---:|
| Data-to-text | 44 | 28 | 0.7500 | **0.7660** | 0.6667 |
| QA | 32 | 9 | **0.8000** | 0.6207 | 0.7368 |
| Summarization | 24 | 7 | 0.8333 | **0.9333** | 0.8333 |
| Macro average | — | — | **0.7944** | 0.7733 | 0.7456 |

The apparent QA advantage and summarization weakness are hypotheses, not domain claims; the slices are small and no task-specific threshold was used.

## Resource measurements

| System | Mean latency | p50 | p95 | Evidence pairs | Est. input tokens | Audit probes |
|---|---:|---:|---:|---:|---:|---:|
| MREG/raw SDK | 4.885 s | 4.498 s | 9.773 s | 1,353 | 442,016 | 503 |
| Whole HHEM | 0.625 s | 0.550 s | 1.412 s | 100 | 74,456 | 0 |
| Ragas/Gemini | 8.233 s | 7.885 s | 12.583 s | not exposed | not captured | 0 |

MREG is about 8.2 times slower at the median and uses about 5.9 times the estimated verifier input of whole-response HHEM. It is about 43% faster at the median than this remote Ragas configuration, but that is not a hardware-equivalent comparison. HHEM ran on CPU; the RTX 4050 remained at 0% utilization. The process was observed at roughly 5.6 GiB working set during inference, but `psutil` was unavailable to the benchmark process, so the report does not claim a controlled peak-RAM measurement.

## Integrity and leakage audit

`benchmarks.audit_frozen_run` replayed the completed artifacts without inference and passed all checks:

- official dataset hashes and selected-ID digest match the manifest;
- every checkpoint ID is unique and complete;
- checkpoint labels, groups, and splits match the official joined dataset;
- all detectors contain exactly the same 100 test IDs;
- every saved score lies in [0, 1];
- the logistic model, thresholds, summaries, slices, and paired bootstraps reproduce exactly;
- selected IDs and distinctive frozen metric values do not occur in prediction source;
- prediction source does not read the benchmark result directory;
- regression tests show that changing non-fit/test labels cannot change learned scores or thresholds;
- SDK and whole-response result caches are distinct, while only immutable model weights are shared.

Saved prediction hashes are recorded in `benchmarks/results/ragtruth_mreg_hhem_ragas_fresh100/integrity_audit.json`.

## Development and negative results

The following are development results and were not used as independent evidence:

- A full heuristic-verifier run showed a sizable improvement over its transparent whole-response heuristic control, but it did not compare with real Ragas.
- A claim-only HHEM pilot lost clearly to whole-response HHEM.
- A 27-feature MREG pilot overfit a 40-example fit set and performed poorly.
- The compact two-feature development pilot nearly tied whole-response HHEM but did not beat it.
- On the fresh run, the learned compact score improves AUROC substantially over the raw SDK risk but improves F1 by only 0.0072.

These negative results are part of the scientific record. No additional Step-1 variant will be fitted after seeing the frozen test set.

## Limitations

1. The held-out set has only 100 examples; paired intervals are wide.
2. The experiment covers one benchmark and one fresh cohort, not “every company/service.”
3. RAGTruth's original labels have documented limitations and newer re-annotations report additional missed hallucinations.
4. MREG uses 80 labeled fit examples plus an out-of-fold threshold; Ragas and whole HHEM use a threshold selected on the remaining 20 calibration examples. This is a frozen protocol difference that must be disclosed.
5. Only Ragas and the identical HHEM control were executed. Closed products such as Cleanlab TLM and Galileo Luna were not available for a fair local run; Patronus Lynx and newer academic detectors remain future baselines.
6. Ragas provider token counts and exact dollars were not captured.
7. Memory telemetry was incomplete.
8. The benchmark measures answer-level hallucination detection. It does not validate claim-span localization, root-cause diagnosis, repair selection, selective blocking, or production utility.

## Defensible claim

The strongest accurate wording is:

> Built a pip-installable, model-agnostic RAG reliability SDK with multi-resolution whole-answer and claim-level evidence verification, counterfactual evidence audits, group-safe calibration, durable benchmark replay, and resource telemetry. On a frozen 100-example RAGTruth test cohort, it achieved F1 0.771 versus 0.747 for real Ragas Faithfulness and 0.709 for the identical whole-response HHEM control, with 100% coverage; paired confidence intervals were inconclusive.

Do not say “outperformed Ragas,” “beat Ragas,” “state of the art,” “44% better,” or “solved hallucinations.”

## Reproduction

Expensive benchmark (requires the pinned HHEM cache and `GEMINI_API_KEY`):

```powershell
python -m benchmarks.ragtruth_benchmark `
  --data-dir benchmarks/data/RAGTruth `
  --output-dir benchmarks/results/ragtruth_mreg_hhem_ragas_fresh100 `
  --detector all `
  --verifier hhem `
  --trust-remote-code `
  --validation-limit 100 `
  --test-limit 100 `
  --validation-offset 50 `
  --test-offset 50 `
  --hhem-batch-size 8 `
  --ragas-model gemini-3.5-flash-lite `
  --ragas-base-url https://generativelanguage.googleapis.com/v1beta/openai/ `
  --ragas-api-key-env GEMINI_API_KEY `
  --ragas-max-tokens 4096 `
  --bootstrap-samples 5000
```

Replay-only audit (no HHEM or API calls):

```powershell
python -m benchmarks.audit_frozen_run `
  --data-dir benchmarks/data/RAGTruth `
  --output-dir benchmarks/results/ragtruth_mreg_hhem_ragas_fresh100
```

## Step-1 stopping rule

Step 1 is closed. Any new detector idea belongs in a preregistered future experiment with a new untouched test set. The next work should validate diagnostic evidence and build a budgeted escalation/repair protocol; it must not tune MREG on this test set.

## Primary references

- [RAGTruth](https://aclanthology.org/2024.acl-long.585/)
- [Ragas Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model)
- [Gemini 3.5 Flash-Lite model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
