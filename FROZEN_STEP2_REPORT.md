# Frozen Step 2 report: claim/span localization

Date completed: 2026-08-21  
Decision: **H7 failed; stop the current MREG whole-clause localization branch.**

## What was tested

The preregistered question was whether thresholded MREG claim-risk spans localize
RAGTruth hallucinations better than three projections produced from the same
frozen verifier pass:

1. raw per-clause HHEM risk;
2. whole-response HHEM risk projected over the complete response; and
3. the SDK's categorical unsafe-claim decisions.

The cohort contained 100 calibration responses and 300 held-out test responses.
All 193 source groups seen in the Step 1 checkpoint were excluded before the
label-independent SHA-256 selection. Calibration had 98 source groups, test had
204, and their overlap was zero. Thresholds were chosen using calibration labels
only. The primary metric was corpus-micro character F1; paired intervals used
5,000 source-group bootstrap resamples. All 400 local HHEM evaluations succeeded.

The 5+5 smoke test was used only to validate mechanics and was discarded as
accuracy evidence.

## Primary held-out result

| Projection | Precision | Recall | Micro char F1 | IoU | Predicted chars |
|---|---:|---:|---:|---:|---:|
| MREG claim risk | 0.1824 | 0.4534 | **0.2602** | 0.1495 | 25,294 |
| Raw clause HHEM | 0.1943 | 0.4477 | **0.2710** | 0.1567 | 23,452 |
| Global HHEM, full response | 0.2311 | 0.4116 | **0.2960** | 0.1737 | 18,130 |
| Categorical claim decision | 0.0946 | 0.8223 | **0.1697** | 0.0927 | 88,468 |

Gold contained 10,177 unioned annotated characters. Coverage was 300/300, so
the failure-as-empty sensitivity result is identical to the primary result.
Example-macro F1 is not used for the decision because its 0.5983 value is
inflated by exact all-negative examples.

### Paired decision tests

| Comparison (MREG minus baseline) | F1 delta | Paired 95% CI | Conclusion |
|---|---:|---:|---|
| Raw clause HHEM | -0.0108 | [-0.0175, -0.0046] | **MREG is worse** |
| Global full-response projection | -0.0358 | [-0.1898, 0.1043] | Inconclusive |
| Categorical decision | +0.0905 | [0.0221, 0.1522] | MREG is better |

H7 required improvement over every listed baseline, especially raw clause
HHEM. It therefore fails under the preregistered stop rule. The global
projection has the best point estimate, but its interval against MREG is too
wide to claim a difference.

## Why it failed

This is mainly a boundary-precision failure, not an offset or scoring-pipeline
failure.

- RAGTruth's 144 test annotations have median length 41 characters, while the
  2,145 deterministic clause units have median length 104 characters.
- Relative to raw clause HHEM, MREG selected 1,842 additional characters. Those
  added only 58 true-positive characters but 1,784 false-positive characters.
- MREG slightly increased recall, from 0.4477 to 0.4534, but the precision loss
  was larger, from 0.1943 to 0.1824.
- Task-level micro F1 was low for data-to-text (0.1576), moderate for summaries
  (0.3655), and highest for QA (0.4693). Raw HHEM remained slightly better on
  data-to-text and summaries and essentially tied on QA.
- Descriptive claim-risk recall was 0.5157 for Evident Baseless Info, 0.3254 for
  Evident Conflict, and 0.2416 for Subtle Baseless Info. These slices are
  exploratory and were not used for selection.

The categorical policy has high recall because it marks too many entire clauses;
its 80,099 false-positive characters make it unsuitable for precise highlighting.
The MREG risk modifications are useful for answer triage in some settings, but
this run shows that they do not improve exact localization when projected over
whole clauses.

## Separate answer-level result on this cohort

The raw SDK grounding-risk score, thresholded on the 100 calibration responses,
obtained:

- F1 0.6054, precision 0.4788, recall 0.8229;
- balanced accuracy 0.7007;
- AUROC 0.7756 and average precision 0.5697;
- 96 positive and 204 negative test responses; 79 TP, 86 FP, 118 TN, and 17 FN.

This is not the learned compact MREG configuration used in Step 1, so its F1
must not be presented as a direct Step 1 replication. Its task slices expose a
serious operating-threshold problem: all 88 data-to-text responses were flagged
(53 TP, 35 FP), producing balanced accuracy 0.5 despite F1 0.7518. QA F1 was
0.3529 and summary F1 was 0.4928. A single universal threshold is not yet a
credible product policy.

Ragas Faithfulness is not included in the span table because its scalar public
metric does not return response character offsets. The valid Ragas comparison
remains the separate answer-level Step 1 experiment; Step 2 neither beats nor
loses to Ragas on span localization because that comparison is undefined.

## Resource result

The run was local and made zero remote/API calls. Across all 400 responses it
recorded 46.98 minutes of summed evaluation latency, 2,945 claims, and 2,309,474
estimated input tokens. On the 300-response test subset:

- mean latency 7.41 s, p50 5.68 s, p95 20.32 s;
- 2,145 claims, 3,919 evidence pairs, 1,474 audit probes, and 598 verifier batches;
- 1,772,638 estimated input tokens;
- 13 input chunks truncated/dropped, zero truncated claims, and zero truncated
  response characters;
- maximum similarity-matrix storage 3,060 bytes.

The benchmark runtime lacked `psutil`, so the checkpoint's RSS fields are zero.
The external process observation was approximately 5.9 GiB working memory, but
that was not a controlled peak trace and is not treated as a benchmark metric.
PyTorch was CPU-only; no GPU workload was used.

## Integrity and reproducibility

The replay audit reconstructed every ID, label, source group, split, gold span,
claim boundary, threshold, summary, and bootstrap result exactly. It also found
no selected-ID or frozen-cohort reference in prediction source. Audit status is
`pass`.

Frozen SHA-256 values:

- selected ID sequence: `d075f9986b65d8855b54b313206f3798ef87f3100e4e58335ae484efcbaea0fb`
- score checkpoint: `bb89522c0e4b651bc13afc65b394b9c663128585c83547d1e857bbdb6b8ea371`
- span summary: `d3777707b313faf8882d2a73daa7feaf2ac8330a5fdd13bc561fd4d8c34c333b`
- answer summary: `6813a86a900a877ca9d62f6bac0260bf909709cb773afad33e38cc841d0e72ab`
- manifest: `416e93477cfc33d23ca08908f3994ddba4b62a06a347b318a0bacc130305c906`
- exploratory slices: `6e795a1e327e7fd8a00fafd540f31f820059ffe463ec1b1f58644efa4ff09067`

Artifacts are in
[`benchmarks/results/ragtruth_step2_spans_frozen_20260821`](benchmarks/results/ragtruth_step2_spans_frozen_20260821).
The audit can be replayed without model inference using
`python -m benchmarks.audit_step2_spans`.

## Scientific and product decision

Do not tune this method against these 300 test labels and do not run a modified
projector on the same cohort as if it were new evidence. This cohort is now
consumed for model-development purposes.

The next defensible branch is a **boundary projector**, not more answer-level
risk fusion:

1. freeze raw per-clause HHEM as the local risk kernel;
2. develop a label-free minimal-subspan projection method using evidence/claim
   alignment and controlled token or phrase deletions;
3. calibrate it without these 300 test labels;
4. compare it with raw-clause projection and a specialized span detector on a
   fresh cohort excluding all Step 1 and Step 2 source groups;
5. require a significant micro-character-F1 gain plus materially lower false
   positive span length at a measured latency/memory budget.

This is a useful negative result. It removes one overcomplicated component from
the localization path and identifies the actual bottleneck: finding unsupported
boundaries inside a mostly supported clause.
