"""
rag_eval_sdk — Production-ready RAG Evaluation SDK.

Five novel features not found in RAGAS / TruLens / DeepEval:
  1. Chunk Attribution Map             (per-chunk grounding analysis)
  2. Failure Mode Classifier           (6-mode root-cause diagnosis)
  3. Inter-Chunk Conflict Detection    (preventive, pre-generation)
  4. Position Bias Detection           ("Lost in the Middle" analyzer)
  5. LLM-as-Judge Verifier             (Gemini cross-validation)

Quick start:
    from rag_eval_sdk import detect_conflicts, detect_position_bias
    from rag_eval_sdk import judge_faithfulness
    conflicts = detect_conflicts(chunks)
    bias      = detect_position_bias(response, chunks, scores)
    verdict   = judge_faithfulness(response, chunks)
"""

__version__ = "1.1.0"

from .conflict_detector import detect_conflicts
from .position_bias_detector import detect_position_bias
from .chunk_attributor import score_attribution_map
from .evaluators import score_relevance_completeness, score_hallucination
from .failure_classifier import classify_failure_mode
from .llm_judge import judge_faithfulness

__all__ = [
    "detect_conflicts",
    "detect_position_bias",
    "score_attribution_map",
    "score_relevance_completeness",
    "score_hallucination",
    "classify_failure_mode",
    "judge_faithfulness",
    "__version__",
]
