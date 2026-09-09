# Research protocol: auditable, budget-aware RAG verification

Status: Step 1 completed on 2026-08-19. Step 2 completed on 2026-08-21 and its MREG whole-clause localization hypothesis failed, so that branch is closed. New mechanisms require a versioned protocol and new untouched source groups. See [FROZEN_STEP1_REPORT.md](FROZEN_STEP1_REPORT.md), [STEP2_PROTOCOL.md](STEP2_PROTOCOL.md), and [FROZEN_STEP2_REPORT.md](FROZEN_STEP2_REPORT.md).

## Frozen Step-1 disposition

- H1 detection: favorable point estimate versus real Ragas (+0.0236 F1), but not demonstrated because the paired 95% interval crosses zero.
- H2 added value: favorable point estimate versus identical whole-response HHEM (+0.0620 F1), also statistically inconclusive.
- H3 efficiency: full local coverage and faster median latency than the tested remote Ragas configuration, but about 8.2 times slower than one-pass local HHEM.
- H4 audit validity: not established; ERA diagnostic probes were not part of the compact learned decision features.
- H5 diagnosis: not tested by RAGTruth and must use controlled pipeline failures.
- H6 position: not tested by this benchmark and must use actual matched generator interventions.

No Step-1 detector changes may be justified using the frozen 100 test labels.
Post-test ideas require a new versioned protocol and untouched source groups.

## Research question

Can claim-level evidence mapping plus a budgeted Evidence Reaction Audit (ERA) improve held-out hallucination detection or failure localization at a competitive latency/cost, beyond the gain from the underlying verifier alone?

The important comparison is not only against RAGAS. The field already includes compact consistency models, fact-checking cascades, and span detectors. RAGAS is the recognizable product baseline; a bare HHEM verifier and a current token/span detector are stronger scientific ablations.

## Hypotheses

- H1 — Detection: v2 improves response-level F1 and balanced accuracy over real RAGAS Faithfulness on a held-out labelled set.
- H2 — Added-value: v2 improves over the exact same pair verifier applied to the whole response. If this fails, claim decomposition/ERA has not justified its complexity.
- H3 — Efficiency: local v2 achieves high coverage without remote calls for most examples and exposes a usable cost/latency frontier.
- H4 — Audit validity: supported verifier decisions that fail ERA are more likely to be false positives than those that pass ERA.
- H5 — Diagnosis: stage-gated diagnoses identify the manipulated failure stage more accurately than score-threshold rules.
- H6 — Position: paired beginning/middle/end treatments reveal reproducible generator sensitivity that the legacy one-pass position heuristic cannot establish.
- H7 — Localization: calibration-only MREG claim-risk spans improve micro character F1 over the categorical claim policy, raw clause-level HHEM risk, and a full-response projection of identical global HHEM, with a paired source-group bootstrap interval excluding zero.

## Datasets and roles

1. RAGTruth — primary response-level and span-labelled detection benchmark. Calibrate on `train`; evaluate once on `test`; group by `source_id`.
2. RGB — secondary generator/system stress test. Report noise accuracy, all-noise rejection, integration accuracy, counterfactual error detection, and correction. Do not use it as the sole detector benchmark.
3. At least one non-RAGTruth detector dataset — required before a strong generalization claim, especially because HHEM and LettuceDetect publish RAGTruth results and may have been selected or trained with that distribution.
4. A purpose-built causal diagnostic set — inject exactly one known failure at a time: corpus absence, retrieval miss, packing drop, evidence conflict, generator noncompliance, and chunk position. Keep clean controls and mixed-failure cases separate.

Dataset licenses are part of the product decision. RGB is CC BY-NC-SA 4.0 and cannot simply become proprietary commercial evaluation data.

## Compared systems

- Lexical/hash baseline.
- Bare learned pair verifier on packed context and whole response.
- Bare verifier with sentence-level claim units.
- Claim units plus top-k evidence mapping.
- Claim units plus evidence mapping plus ERA.
- Full local cascade plus bounded remote escalation.
- Official RAGAS Faithfulness with disclosed version and evaluator model.
- At least one current compact/span detector such as MiniCheck or LettuceDetect, subject to model and code licenses.

All systems receive identical examples. Any system-specific context limit or truncation is recorded. Failures remain visible in coverage and failure rates.

## Primary endpoints

- Response-level hallucination F1.
- Balanced accuracy.
- AUROC and average precision, so conclusions do not depend only on one threshold.
- Coverage and failure rate.

Secondary endpoints:

- precision and recall;
- task-stratified metrics for QA, summarization, and data-to-text;
- span-level metrics when the detector emits spans;
- mean, p50, and p95 latency;
- peak/estimated working memory, matrix bytes, tokens, calls, and monetary cost;
- calibration error and threshold stability across folds;
- diagnostic-stage accuracy and time-to-fix in a small engineer study.

## Calibration and leakage rules

- The test labels must never select thresholds, weights, prompts, models, evidence count, or audit budget.
- Use the official split where available. Enforce no `source_id` overlap between calibration and test.
- Choose one primary operating objective before test: F1, balanced accuracy, or a precision floor. The current runner defaults to F1 only as an explicit configurable policy.
- Treat unparseable judge output, quota exhaustion, and timeouts as failures. Report successful-only metrics plus coverage and failure-as-incorrect sensitivity.
- Cache exact prompts and responses. Never regenerate only the examples a method got wrong.
- Record package versions, model IDs, endpoint, temperature, seed, prompt version, context limits, and dataset checksum.

## Statistical decision rules

“Beats” requires more than a larger point estimate:

- the paired test-set F1 difference must be positive;
- its 95% paired bootstrap interval should exclude zero;
- McNemar’s paired test should support a difference in correctness;
- coverage should be at least 98% or no worse than the compared method;
- gains must not come entirely from one task type or one source model;
- results must replicate on a second dataset or domain.

If the interval crosses zero, report “no demonstrated difference.” If accuracy improves but latency/cost is much worse, report a trade-off curve rather than a win.

## ERA validation

ERA is the candidate research contribution, but it can fail in predictable ways:

- Removing a chunk may have no effect because evidence is redundant.
- A synthetic mutation may create unnatural text or alter an irrelevant fact.
- A verifier may react to surface changes without reasoning about evidence.
- Claim splitting may combine several facts and obscure which fact changed.
- The same model used for base verification and intervention scoring can share blind spots.

Required tests:

1. Mutation validity: humans judge whether the mutation changes the claim-relevant fact while preserving the rest.
2. Sensitivity discrimination: compare ERA scores for known-correct versus known-wrong verifier decisions.
3. Negative controls: mutate facts unrelated to the claim; the support score should stay stable.
4. Redundancy controls: duplicate supporting evidence; removal of one duplicate should not be labelled a failure.
5. Cross-verifier replication: run interventions through at least two independent verifier families.
6. Ablate removal, number mutation, entity mutation, and negation mutation separately.

If ERA adds no significant value over the bare verifier, remove it from the default product path. It may remain a diagnostic experiment, but not a scoring feature.

## Position-bias validation

The legacy detector inferred position bias from one response and chunk similarities. That is correlational. The v2 experiment keeps query and chunks fixed, moves one target chunk to beginning/middle/end, generates under matched settings, and compares answer/support scores. Repeat over seeds or deterministic decoding, randomize treatment order, and use paired confidence intervals.

## Product gates

Research accuracy is only gate one. A credible company product also needs:

- adapters for common tracing standards and RAG frameworks;
- async batch and streaming APIs with backpressure;
- encrypted or customer-controlled caches and a no-retention mode;
- PII/secrets redaction and tenant isolation;
- versioned policies, calibrations, prompts, and model artifacts;
- a review queue for `escalate` and `insufficient_evidence` decisions;
- drift monitoring by domain, model, language, and document type;
- evidence that the diagnosis recommends the correct engineering experiment;
- a clear license inventory for every model and dataset.

The likely product wedge is not “another faithfulness number.” It is a low-cost verification and incident-localization layer that shows which claim failed, which evidence was used, whether the verifier reacted to evidence, what stage remains unobserved, and what experiment should be run next.

## Stop conditions

Stop adding mechanisms and simplify if any of these occur:

- v2 does not beat its bare-verifier ablation;
- gains vanish outside RAGTruth;
- ERA mostly fires on redundant evidence or invalid mutations;
- remote escalation is required for most claims;
- p95 latency or memory makes online use impractical;
- diagnoses remain `undetermined` because normal production traces lack the required observations.

Those are useful scientific results. They prevent a complicated prototype from being mistaken for a product.
