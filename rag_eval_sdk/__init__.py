"""Auditable, budget-aware verification and diagnostics for RAG systems.

The v2 API is experimental.  It reports claim-level decisions, evidence,
controlled intervention results, and resource usage.  Legacy v1 functions are
still exported for compatibility but should not be used as benchmark baselines.
"""

__version__ = "2.0.0a2"

from .calibration import (
    BinaryMetrics,
    CalibrationResult,
    RankingMetrics,
    binary_metrics,
    ranking_metrics,
    select_threshold,
)
from .config import HeuristicWeights, VerificationConfig
from .diagnostics import DiagnosticReport, FailureHypothesis, StageObservations, diagnose_failure
from .embeddings import HashingSimilarityBackend, SentenceTransformerBackend
from .pipeline import RAGEvaluator
from .position_experiment import analyse_position_effect, create_position_variants
from .risk_model import (
    MREG_COMPACT_FEATURE_NAMES,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_SCHEMA,
    RiskCalibrationModel,
    extract_risk_features,
    fit_risk_calibrator,
)
from .providers import (
    DiskCachedGenerator,
    GenerationResult,
    GoogleGenAIProvider,
    OpenAICompatibleProvider,
    TextGenerator,
)
from .schema import Decision, EvidenceChunk, PairScore, Severity, VerificationReport
from .text_features import TextSpan, split_claim_spans
from .verifiers import (
    CachedPairVerifier,
    HHEMPairVerifier,
    HeuristicPairVerifier,
    LLMPairVerifier,
    PairVerifier,
)

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
    "RAGEvaluator",
    "VerificationConfig",
    "HeuristicWeights",
    "Decision",
    "Severity",
    "EvidenceChunk",
    "PairScore",
    "VerificationReport",
    "TextSpan",
    "split_claim_spans",
    "HashingSimilarityBackend",
    "SentenceTransformerBackend",
    "HeuristicPairVerifier",
    "HHEMPairVerifier",
    "CachedPairVerifier",
    "PairVerifier",
    "LLMPairVerifier",
    "OpenAICompatibleProvider",
    "GoogleGenAIProvider",
    "DiskCachedGenerator",
    "GenerationResult",
    "TextGenerator",
    "BinaryMetrics",
    "CalibrationResult",
    "RankingMetrics",
    "binary_metrics",
    "ranking_metrics",
    "select_threshold",
    "DiagnosticReport",
    "FailureHypothesis",
    "StageObservations",
    "diagnose_failure",
    "create_position_variants",
    "analyse_position_effect",
    "RISK_FEATURE_NAMES",
    "MREG_COMPACT_FEATURE_NAMES",
    "RISK_FEATURE_SCHEMA",
    "RiskCalibrationModel",
    "extract_risk_features",
    "fit_risk_calibrator",
    "__version__",
]
