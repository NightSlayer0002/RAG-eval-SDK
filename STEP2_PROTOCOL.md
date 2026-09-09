# Step 2 preregistration: claim/span localization

Status: **frozen run completed 2026-08-21; H7 failed and this branch is stopped.**

The preregistration below is preserved as written. The completed result is in
[FROZEN_STEP2_REPORT.md](FROZEN_STEP2_REPORT.md). On 300 source-exclusive test
responses, MREG claim-risk projection obtained micro character F1 0.2602 versus
0.2710 for raw clause HHEM. The paired delta was -0.0108 with 95% CI
[-0.0175, -0.0046], so the preregistered stop rule fired. No result-dependent
retuning was performed.

This step asks a harder question than Step 1: not only whether an answer contains
a hallucination, but which exact response characters are unsupported. RAGTruth's
published span protocol uses character-overlap precision, recall, and F1
([ACL 2024 paper, section 5.3](https://aclanthology.org/2024.acl-long.585/)).

## Frozen hypothesis

H7: With one threshold selected only on a source-group-disjoint calibration
split, MREG's claim-risk spans will obtain higher micro character F1 than:

1. the SDK's existing categorical claim-decision policy;
2. a full-response projection of the identical answer-level HHEM control; and
3. raw per-clause HHEM risk using the same claim splitter and evidence budget.

The result is a win only if the paired, source-group bootstrap 95% confidence
interval for the F1 difference excludes zero. A higher point estimate alone is
not a win.

Ragas Faithfulness remains an answer-level comparator. Its public scalar result
does not identify response offsets, so it will not be misrepresented as a span
detector. A specialized token/span model such as LettuceDetect should be
reported separately as a trained ceiling, with its RAGTruth training exposure
made explicit ([paper](https://arxiv.org/abs/2502.17125)). RT4CHART is the most
important recent research comparison because it also maps hierarchical claim
decisions back to spans ([paper](https://arxiv.org/abs/2603.27752)).

## Cohort freeze

- Dataset files and hashes remain the official local RAGTruth snapshot.
- Exclude **every `source_id` group** present in the frozen Step 1 score
  checkpoint before selecting any Step 2 example.
- Select examples by the existing label-independent SHA-256 ID ordering.
- Planned frozen size: 100 calibration responses and 300 held-out test
  responses. No test label may affect a threshold, feature, prompt, splitter,
  or verifier setting.
- Use `--risk-calibrator threshold` for this experiment. Response-level learned
  MREG calibration is not needed to select the per-claim localization threshold.
- A mechanics-only smoke run may use at most 5 calibration and 5 test examples;
  its outputs are discarded and cannot be reported as evidence.

The benchmark runner now supports `--exclude-groups-from SCORES_JSONL`. The
checkpoint hash and excluded-group digest are written into the new manifest.

## Frozen system configuration

- Deterministic clause units with exact response offsets.
- Hashing similarity, 2,048 dimensions.
- Coverage-aware evidence selection, 3 chunks per claim.
- HHEM model and tokenizer at the immutable revisions recorded in
  `FROZEN_STEP1_REPORT.md`.
- CPU, batch size 8, maximum 24 claims, maximum 6,000 evidence characters per
  claim, and at most 2 evidence-intervention audits.
- No remote LLM is required for the SDK localization run.

Settings may change only after a failed mechanics smoke test. Any change must be
written here before the frozen run, and the reason must not use held-out labels.

## Metrics and failure policy

Primary:

- corpus-micro character precision, recall, and F1 over the union of half-open
  predicted and gold intervals;
- paired source-group bootstrap confidence interval for each F1 difference.

Secondary:

- character intersection-over-union;
- example-macro character metrics (all-negative exact matches are counted as
  correct and therefore this is never the primary metric);
- answer-level F1/AUROC/AUPRC on the same cohort;
- task and annotation-type slices, treated as exploratory unless sufficiently
  powered;
- coverage, failure-as-empty-prediction sensitivity, latency, verifier pairs,
  batches, estimated input tokens, and truncation counts.

Overlapping or duplicate annotations are unioned before counting. Failed calls
are excluded from the primary successful-only metric and treated as empty span
predictions in a required sensitivity result. There is no silent retry-based
deletion.

## Implemented safeguards

- `load_ragtruth` preserves every official gold span and rejects invalid
  offsets or text/offset mismatches.
- Claim splitting is label-free and maps normalized verification text back to
  the untouched response.
- Predictions store only offsets, risks, decisions, and evidence IDs; copying
  response text into every claim record is avoided.
- Span unions are computed as intervals, not full-length character masks, so
  metric memory is proportional to span count rather than response length.
- Threshold selection reads calibration records only; a regression test changes
  held-out labels and verifies that the selected threshold is unchanged.

## Stop/go rule after the frozen run

- **Go to Step 3 (selective escalation):** H7 passes, coverage is at least 99%,
  and median latency/resource use stays within the frozen budget.
- **Revise once:** point estimate improves but uncertainty is inconclusive, or
  recall is useful but whole-clause projection causes low precision.
- **Stop this branch:** no improvement over raw clause HHEM, or any apparent
  gain depends on test-label tuning, missing examples, annotation leakage, or a
  materially larger verifier budget.

If whole-clause spans are the main precision bottleneck, the allowed next
research change is a label-free minimal subspan projector. It must be frozen on
calibration data and tested against exact gold characters; it may not search for
gold substrings or use label text at inference time.
