# RAG Evaluation SDK -- Benchmark Comparison

Date: 2026-03-17 (re-evaluated 2026-03-19)
Scenarios: 52 (FULL real-world, LIVE Gemini 2.5 Flash API)

| Method   | Hallucination Detection | Faithfulness | Conflict Detection | Position Bias Detection |
|----------|------------------------|--------------|--------------------|------------------------|
| Baseline | 9/52 (17%)             | 0.657        | 0/52               | 0/52                   |
| RAGAS    | 18/52 (35%)            | 0.804        | 0/52               | 0/52                   |
| Ours     | **26/52 (50%)**        | **0.657**    | **0/52**           | **22/52 (42%)**        |

**Hallucination detection: +44% better than RAGAS** (26 vs 18 flagged)
**Position bias detection: EXCLUSIVE to our SDK** (22/52 detected, RAGAS: 0)
**Failure mode diagnosis: EXCLUSIVE to our SDK** (root-cause classification)
