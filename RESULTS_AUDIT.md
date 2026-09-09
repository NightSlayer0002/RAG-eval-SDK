# Historical results audit

This file separates historical artifacts from evidence that can support a public
claim. Obsolete result files were removed from the curated GitHub edition;
historical reconstructions below describe the separately retained original project.

## Bottom line

No checked-in historical result demonstrates that this SDK beat the RAGAS
package. The name `RAGAS` in the old scripts referred to a local similarity
heuristic, not an invocation of RAGAS Faithfulness. The old 52 scenarios are
useful regression fixtures, but they are not an independent benchmark.

## The 52 live-generation scenarios

The responses were generated through a live API, but the scenario categories
were authored within this project rather than independently annotated. Treating
those categories as response-level labels yields the following reconstruction:

| Method in old artifact | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 5 | 4 | 31 | 12 | 0.56 | 0.29 | 0.39 | 0.69 |
| Custom `RAGAS`-named heuristic | 8 | 10 | 25 | 9 | 0.44 | 0.47 | 0.46 | 0.64 |
| Historical SDK | 10 | 16 | 19 | 7 | 0.39 | 0.59 | 0.47 | 0.56 |

The historical SDK gained recall at a large precision cost. Its F1 advantage
over the custom comparator is about one point, not 44%. These figures still
overstate certainty because the labels were not independent and no confidence
interval was calculated.

The position-bias output reconstructed against the scenario categories was
approximately precision 0.41, recall 0.53, F1 0.46, and accuracy 0.60. A
single-response correlation between chunk position and similarity cannot prove
position bias; the v2 paired intervention runner replaces that inference.

The historical conflict detector found none of the 18 scenarios categorized as
conflicts. That result does not support a conflict-detection claim.

The historical LLM-judge output was not usable: 43 records were errors, 7 were
parse errors, and only 2 produced a mostly-faithful result.

## The 20 synthetic scenarios

`benchmarks/results/benchmark_results.json` reports a 100% hallucination
reduction for the SDK. That aggregate conflicts with the file's own per-scenario
records: the SDK marks almost every scenario as a hallucination or position-bias
failure, including most clean scenarios. The aggregate is therefore not a valid
performance statistic.

## Historical RGB attempt

The original RGB summary contained errors for every dataset, and the old runner
did not correctly handle all RGB data shapes. There are no historical RGB scores
to report. The v2 loader now validates `en_refine`, `en_int`, and `en_fact`, but
generation has deliberately not been run yet.

## Frozen v2 result (2026-08-19)

A fresh official run has now replaced the earlier “unknown” status. On 100
held-out RAGTruth examples, with thresholds and the compact risk model learned
without test labels:

| Detector | F1 | Precision | Recall | Accuracy | AUROC | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Learned MREG/HHEM | 0.7708 | 0.7115 | 0.8409 | 0.7800 | 0.8551 | 1.000 |
| Real Ragas Faithfulness 0.4.3 / Gemini 3.5 Flash-Lite | 0.7473 | 0.7234 | 0.7727 | 0.7700 | 0.8373 | 1.000 |
| Whole-response HHEM | 0.7089 | 0.8000 | 0.6364 | 0.7700 | 0.8304 | 1.000 |

MREG's F1 differences were +0.0236 versus Ragas (paired bootstrap 95% CI
[-0.0790, 0.1350]) and +0.0620 versus whole-response HHEM (95% CI
[-0.0333, 0.1638]). Both intervals cross zero. The result is competitive and
worth replicating, but statistically inconclusive.

The replay audit reconstructs every model, threshold, summary, and paired
comparison from saved predictions and official dataset hashes. Full details are
in [FROZEN_STEP1_REPORT.md](FROZEN_STEP1_REPORT.md).

## What may be claimed now

- v2 is a pip-package project with clause-level evidence mapping, multi-resolution
  HHEM verification, bounded evidence interventions, explicit uncertainty, and
  resource telemetry.
- A frozen 100-example RAGTruth cohort produced a higher F1 point estimate than
  real Ragas Faithfulness and whole-response HHEM at 100% coverage.
- The difference is not statistically conclusive and is not evidence of state of
  the art, universal superiority, product reliability, or novelty.

Do not use “44% better,” “outperformed Ragas,” “state of the art,” “automated
root-cause classification,” or a position-bias accuracy number in a résumé,
paper, website, or pitch. Use the exact qualified wording in the frozen report.
