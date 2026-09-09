# Validation status

Last updated: 2026-09-09 (GitHub curation).

## Historical experiment validation (2026-08-21)

The following records describe the original experiment environment. Development
pilots and the historical wheel are not distributed in this curated edition.

- Official RAGTruth files downloaded and checksum-pinned.
- RAGTruth loader joined 17,617 responses to sources: 14,942 official train and 2,675 official test examples.
- Vectara HHEM repository and FLAN-T5 tokenizer pinned to immutable revisions.
- HHEM smoke test reproduced the model card's reference scores (0.011062 and 0.647364).
- Full 2,675-example official RAGTruth test run with the transparent heuristic verifier completed with zero failures. This was a development/architecture ablation, not a Ragas comparison.
- Claim-only and high-dimensional MREG/HHEM development pilots completed and were retained as negative results.
- Frozen fresh RAGTruth comparison completed:
  - 80 fit examples;
  - 20 calibration examples;
  - 100 untouched test examples;
  - learned compact MREG/HHEM;
  - identical whole-response HHEM control with an isolated result cache;
  - real Ragas Faithfulness 0.4.3 with Gemini 3.5 Flash-Lite;
  - 100% held-out coverage and zero failures for all detectors;
  - 5,000 paired bootstrap resamples and exact McNemar tests.
- Replay-only integrity audit passed and exactly regenerated the risk model, thresholds, summaries, slices, and paired comparisons from saved predictions.
- Regression tests cover test-label independence, fit-only learning, group-safe splitting, fresh split offsets, failure retention, paired ID alignment, and SDK/control cache isolation.
- Step 2 code foundation preserves and validates all 14,226 gold spans in the 17,617 quality-filtered RAGTruth examples, maps verifier claims to exact response offsets, and evaluates interval-union character metrics without allocating full response masks.
- Step 2 checkpoints retain compact claim risk, raw pair risk, global pair risk, decisions, offsets, and evidence IDs from one verifier run. Thresholds use calibration records only; paired comparisons resample source groups.
- Prior benchmark source groups can be excluded from a new cohort using hashed score checkpoints.
- Frozen Step 2 completed on 100 calibration and 300 held-out RAGTruth responses after excluding all 193 Step 1 source groups. All 400 calls succeeded and calibration/test source-group overlap was zero.
- Step 2 H7 failed: MREG whole-clause localization micro character F1 was 0.2602 versus 0.2710 for raw clause HHEM; paired delta -0.0108, 95% CI [-0.0175, -0.0046]. The current localization branch is stopped without retuning.
- The Step 2 replay audit exactly regenerated the cohort, labels, gold spans, claim boundaries, answer summary, span summary, and 5,000-resample paired bootstrap results. See [FROZEN_STEP2_REPORT.md](FROZEN_STEP2_REPORT.md).
- Original full local suite: 86 passing tests.
- Built `dist/rag_eval_sdk-2.0.0a2-py3-none-any.whl`, installed it into an isolated temporary target, imported version `2.0.0a2` from that target, and verified the `rag-eval`, `rag-eval-ragtruth`, `rag-eval-rgb`, and `rag-eval-spans` entry-point metadata and CLI help paths.
- Wheel SHA-256: `7e362867bdfe2b6817608b49d2894c0a33bac0624e36f7ddb39baf08b3212670`.
- RGB schemas and bounded prompt construction were validated for the supplied English files without generation calls.

The final frozen metrics and limitations are in [FROZEN_STEP1_REPORT.md](FROZEN_STEP1_REPORT.md) and [FROZEN_STEP2_REPORT.md](FROZEN_STEP2_REPORT.md).

## Reproducibility pins

- SDK version used in the frozen Step 1 run: `2.0.0a1`
- SDK version used in the frozen Step 2 run: `2.0.0a2`
- Current working package version: `2.0.0a2`
- Python: 3.12.13
- Ragas: 0.4.3
- Ragas judge: `gemini-3.5-flash-lite`
- HHEM revision: `8e4a2e6e96c708cc76c2344f7e4757df2515292c`
- HHEM weights SHA-256: `634de18a38cf1e991c1acd0f7a9e0d30f7ea187fba42bb4798f862d3edd31e72`
- FLAN-T5 tokenizer revision: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`
- RAGTruth source SHA-256: `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b`
- RAGTruth response SHA-256: `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`

## Remaining validation work

- Capture provider-reported Ragas input/output tokens and exact dollar cost in future runs.
- Capture controlled peak RAM/VRAM with the benchmark process itself; the frozen run's `psutil` fields are zero.
- Replicate on a new preregistered dataset/cohort. Do not tune MREG on the frozen RAGTruth test IDs.
- Develop any boundary-localization replacement on calibration/development data and evaluate it only on a fresh cohort excluding every Step 1 and Step 2 source group. The consumed 300-example Step 2 test set cannot validate a revised projector.
- Evidence attribution still lacks source-side gold evidence labels; do not equate response-span localization with evidence-witness accuracy.
- Validate diagnostic/root-cause labels and budgeted repair selection separately from answer-level detection.
- Run RGB generation experiments only as generator-robustness tests; RGB is not a replacement for a hallucination-detector benchmark.

## Environment note

The original copied `venv` launcher pointed to an unavailable Python executable.
It has been removed. Create a fresh environment using the README instructions.
The frozen run used a bundled Python runtime with an existing site-packages
directory added explicitly.

## GitHub curation validation (2026-09-09)

- Imported the current SDK, tests, modern benchmark runners, protocols and both
  frozen evidence bundles from the newer project into the publication copy.
- Removed duplicate source, obsolete benchmarks, personal notes, secrets,
  environments, build products, development caches and bulk datasets.
- Corrected replay path handling so exclusion checkpoints come from this checkout
  and prediction-source scanning is independent of the dataset location.
- All 89 offline tests passed, including relocation and checkpoint-tampering checks.
- Both frozen replay audits passed, with the original RAGTruth files provided from
  an external directory. No model inference or remote API calls were made.
- Rebuilt the wheel, installed it into a temporary target and validated the offline
  SDK example and all four console entry-point help commands.
- Validation used Python 3.12.14 from the bundled runtime, with pytest loaded from
  the original environment's site-packages. This is not a fresh online installation
  of every optional HHEM/Ragas dependency.

See [REPRODUCING.md](REPRODUCING.md) for portable, non-mutating replay instructions
and the limits of inference reproduction across SDK and dependency versions.

## Claim policy

The old résumé claim “outperforming RAGAS by 44% across 52 live scenarios” is false. The new frozen run has a favorable but statistically inconclusive +0.0236 F1 point difference versus real Ragas on 100 held-out examples. Public wording must include the sample size, exact baselines, full coverage, and inconclusive uncertainty.
