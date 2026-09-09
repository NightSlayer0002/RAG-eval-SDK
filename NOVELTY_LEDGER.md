# Novelty ledger

Last literature check: 2026-08-19.

This document separates implemented mechanisms from defensible novelty. “No result found” is not proof that no prior art exists. A paper, patent, or sales claim requires a broader search and experiments against the closest work.

## What is already crowded

| Project idea | Closest known work | What already exists | Decision |
|---|---|---|---|
| Claim-level plus whole-answer verification | RT4CHART (2026), RAGChecker, MiniCheck | Claim decomposition, local-to-global verification, evidence-linked diagnostics, answer-level aggregation | Keep as useful engineering; do not claim the category is novel |
| Evidence removal/mutation/order probes | CUE-R (2026), MetaRAG (2025), EAEV (2026), RC-RAG (2024) | Evidence interventions, counterfactual stability, metamorphic relations, confidence/risk response | ERA is not standalone novelty |
| Minimal sufficient evidence | Minimal Evidence Groups (2025), evidence-ranking/MSR work (2026), S2G-RAG (2026) | Set-cover-like minimal evidence groups and structured evidence sufficiency | Do not claim this alone |
| Budget-aware escalation | SATER (2025), routing/cascade literature | Confidence-aware rejection, pre-generation routing, cascades, cost/latency frontiers | Product requirement, not novelty |
| Diagnose and choose a repair under resource limits | D2R-RAG (2026), RePAIR (2026), RAG error-taxonomy work (2026) | Observable failure signatures, adaptive repair, response-to-action learning, latency/VRAM constraints | The original BVER proposal is no longer novel by itself |
| Proof/evidence-carrying output | Proof-Carrying Numbers (2025), claim-selective certification (2026), proof-carrying-answer prototypes | Per-output provenance, claim/evidence selection, empirical certificate language | “Proof-carrying RAG” alone is not available as a novelty claim |
| Position robustness | Lost in the Middle, query/document perturbation studies, 2026 corpus-evolution metamorphic testing | Position/order/noise perturbations and robustness measurement | Our old one-pass position heuristic is not credible novelty |

## Implemented Step-1 mechanism: MREG

Multi-Resolution Evidence Geometry combines a whole-response evidence risk and a worst-claim evidence risk with a compact, train-only logistic calibrator.

What may differ from common baselines:

- the exact two-feature black-box fusion;
- equivalent whole-response context budget plus bounded claim-conditioned evidence;
- one batched verifier interface with explicit resource accounting;
- diagnostic evidence-reaction outputs kept separate from the compact decision model.

What prevents a novelty claim today:

- RT4CHART already describes hierarchical local-to-global verification;
- the fresh result is only one 100-example cohort and is statistically inconclusive;
- no head-to-head experiment with RT4CHART, EAEV, MiniCheck/LettuceDetect, Lynx, TLM, or Luna has been completed;
- the learned layer's F1 gain over the raw SDK score is only 0.0072 on the frozen cohort.

Decision: freeze as the Step-1 detector kernel. Treat it as an engineering/research baseline for later work, not the final invention.

## Candidate invention: BIRC

Working name: **BIRC — Bidirectional Intervention Repair Certificate**.

### Core object

A repaired answer is not accepted merely because its faithfulness score rises. It must carry a replayable empirical certificate showing two opposite behaviors:

1. **Resist:** the answer and verifier decision remain stable under transformations that should be irrelevant—reordering, duplicate evidence, harmless formatting, paraphrase, and removal of non-witness chunks.
2. **Revise:** the answer or release decision changes in the pre-registered direction when licensed decisive evidence is removed, contradicted, superseded, or replaced.

For each claim, the certificate contains:

```text
claim and character span
hashed evidence witness set
source/authority/time policy
pre-registered nuisance interventions and expected invariances
pre-registered decisive interventions and expected directional changes
observed answer/verifier deltas
independent verifier identity
repair action and before/after response hashes
tokens, calls, latency, memory and dollar budget
PASS / ABSTAIN with reason
```

This is an empirical test certificate, not a mathematical proof that the answer is true.

### Precise research hypothesis

> Under a fixed evaluation-and-repair budget, selecting probes that jointly test nuisance invariance and decisive-evidence responsiveness will reject unsafe or evaluator-gamed repairs more accurately than accepting the highest-scoring repair, while testing fewer interventions than an exhaustive repair laboratory.

### Why this may still be distinct

- CUE-R measures per-evidence operational utility; it does not certify a selected repaired answer for release.
- MetaRAG uses metamorphic relations to detect hallucinations; it does not bind a repair action, evidence witness, independent verifier, and resource budget into a replayable release artifact.
- D2R-RAG selects repairs under compute constraints; its published abstract does not describe bidirectional post-repair contracts or independent replay certificates.
- RePAIR maps responses to actions without an explicit diagnostic taxonomy; it does not establish that the resulting repair reacts correctly to decisive evidence.
- Claim-selective certification controls which medical claims/actions are emitted; it is not described as post-repair resist/revise testing.
- Counterfactual report-coordinate work studies internal activation-level incentive compatibility, whereas BIRC is black-box and evidence-grounded.

This distinction is provisional. The next literature/patent pass must inspect full methods and appendices, not just abstracts.

### What would falsify the idea

- A close method already performs the same post-repair resist/revise contract under a budget.
- BIRC accepts evaluator-gamed repairs as often as score-only selection.
- Nuisance/decisive intervention generators are invalid or require expensive human labeling at deployment.
- Independent verification removes any accuracy/cost advantage.
- The certificate adds latency without improving unsafe-repair rejection or engineer trust.

### Required baselines

1. Accept the repair with the highest original evaluator score.
2. Always rewrite from cited spans.
3. Always retrieve more.
4. Random repair with the same action distribution.
5. D2R-style diagnosis-to-action policy.
6. Exhaustive intervention oracle, used only as an upper bound.
7. BIRC without `resist` probes.
8. BIRC without `revise` probes.
9. BIRC using the same verifier versus an independent verifier.

### Primary endpoints

- unsafe-repair acceptance rate;
- successful safe-repair coverage;
- risk-coverage curve and selective risk;
- repair regression rate on previously correct claims;
- certificate false-pass and false-abstain rates;
- evidence-witness attribution F1;
- total tokens, external calls, latency, RAM/VRAM, and dollars;
- replay success rate from hashes and cached model outputs.

## Six frozen project gates

Each step gets a frozen protocol, an untouched test set, negative-result reporting, and a stopping rule.

1. **Answer-level detection — complete.** MREG, whole-response HHEM, and real Ragas on frozen RAGTruth.
2. **Claim/span localization — completed negative result.** On the frozen 100+300 source-exclusive run, MREG whole-clause projection was significantly worse than raw clause HHEM (micro character F1 0.2602 versus 0.2710; paired 95% CI for the delta [-0.0175, -0.0046]). Stop this branch. Exact-boundary projection remains an open engineering problem, not a novelty claim. Evidence attribution still requires a separate source-side gold standard.
3. **Selective escalation.** Compare a calibrated budget policy with always-local, always-judge, fixed-threshold, and random routing at matched call rates.
4. **Causal diagnosis.** Use a purpose-built dataset with one known pipeline intervention per example; do not infer root cause from final-answer labels.
5. **Controlled repair selection.** Execute a small fixed repair set and compare with generic, random, and oracle actions.
6. **BIRC certification and product gate.** Test whether bidirectional certificates prevent unsafe repair release at an acceptable quality/cost frontier.

## Primary sources checked

- [RT4CHART: hierarchical verification](https://arxiv.org/abs/2603.27752)
- [CUE-R: evidence interventions](https://arxiv.org/abs/2604.05467)
- [MetaRAG: metamorphic hallucination testing](https://arxiv.org/abs/2509.09360)
- [EAEV: counterfactual entity/evidence stability](https://aclanthology.org/2026.findings-acl.1477/)
- [D2R-RAG: budgeted diagnosis and repair](https://arxiv.org/abs/2606.29377)
- [RePAIR: response-to-action learning](https://aclanthology.org/2026.acl-short.14/)
- [Minimal Evidence Groups](https://aclanthology.org/2025.trustnlp-main.8/)
- [SATER: token-efficient routing/cascading](https://aclanthology.org/2025.emnlp-main.531/)
- [Claim-selective certification](https://arxiv.org/abs/2605.21949)
- [Rethinking hallucination-detector evaluation / TRIVIA+](https://arxiv.org/abs/2605.11330)
- [RefusalBench](https://aclanthology.org/2026.eacl-long.321/)
- [Cleanlab TLM](https://help.cleanlab.ai/tlm/faq/)
- [Patronus Lynx 2.0](https://docs.patronus.ai/docs/evaluators/advanced_concepts/lynx)
- [Galileo Luna](https://galileo.ai/research)
- [Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model)
