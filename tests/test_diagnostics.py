from rag_eval_sdk import (
    FailureHypothesis,
    HashingSimilarityBackend,
    RAGEvaluator,
    StageObservations,
    diagnose_failure,
)


def test_diagnosis_requires_stage_evidence_for_retrieval_failure():
    report = RAGEvaluator(similarity_backend=HashingSimilarityBackend(256)).evaluate(
        "The answer is forty two.", []
    )
    diagnosis = diagnose_failure(
        report,
        StageObservations(
            answer_supported_in_corpus=True,
            answer_supported_in_retrieved_set=False,
        ),
    )
    assert diagnosis.primary is FailureHypothesis.RETRIEVAL_COVERAGE
    assert diagnosis.confidence < 1.0


def test_unknown_telemetry_does_not_invent_a_root_cause():
    report = RAGEvaluator(similarity_backend=HashingSimilarityBackend(256)).evaluate(
        "The answer is forty two.", []
    )
    diagnosis = diagnose_failure(report)
    assert diagnosis.primary is FailureHypothesis.UNDETERMINED
    assert "answer_supported_in_corpus" in diagnosis.missing_observations

