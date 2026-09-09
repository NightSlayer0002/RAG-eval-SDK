"""Unit tests for evaluators.py and chunk_attributor.py"""
import pytest
from rag_eval_sdk.evaluators import score_relevance_completeness, score_hallucination
from rag_eval_sdk.chunk_attributor import score_attribution_map


class TestRelevanceScore:
    def test_high_relevance(self):
        response = "The clinic is open Monday through Friday from 9 AM to 5 PM."
        chunks = [
            {"id": 1, "text": "Our clinic operates Monday to Friday, 9 AM to 5 PM."},
            {"id": 2, "text": "We are closed on weekends and public holidays."},
        ]
        score = score_relevance_completeness(response, chunks)
        unrelated = score_relevance_completeness(
            "A spacecraft entered orbit around a distant planet.", chunks
        )
        assert 0.0 <= score <= 1.0
        assert score > unrelated

    def test_low_relevance(self):
        response = "The stock market reached all-time highs yesterday."
        chunks = [
            {"id": 1, "text": "IVF treatment involves fertilizing eggs in a laboratory."},
            {"id": 2, "text": "The clinic offers genetic screening services."},
        ]
        score = score_relevance_completeness(response, chunks)
        assert -0.2 <= score <= 1.0   # cosine sim can be slightly negative
        assert score < 0.6

    def test_empty_chunks(self):
        score = score_relevance_completeness("Some response.", [])
        assert score == 0.0


class TestHallucinationScore:
    def test_grounded_response(self):
        response = "The clinic is open Monday to Friday from 9 AM to 5 PM."
        chunks = [{"id": 1, "text": "We are open Monday to Friday, 9 AM to 5 PM."}]
        score = score_hallucination(response, chunks)
        assert 0.0 <= score < 0.5

    def test_hallucinated_response(self):
        response = "Hotel Raj Palace and Hotel Blue Moon are located nearby."
        chunks = [{"id": 1, "text": "IVF treatment is available at the fertility clinic."}]
        score = score_hallucination(response, chunks)
        assert score >= 0.3

    def test_empty_chunks_is_maximum_risk(self):
        score = score_hallucination("Any response.", [])
        assert score == 1.0


class TestChunkAttribution:
    def test_attributed_chunks(self):
        response = "The clinic is open Monday to Friday, 9 AM to 5 PM."
        chunks = [{"id": 1, "text": "We are open Monday to Friday, from 9 in the morning to 5 in the evening."}]
        result = score_attribution_map(response, chunks)
        assert result["attributed_count"] >= 0
        assert isinstance(result["attributed_chunks"], list)

    def test_phantom_chunks(self):
        response = "Hotel Raj Palace and Hotel Blue Moon are located nearby."
        chunks = [{"id": 1, "text": "IVF treatment requires careful embryo monitoring in the lab."}]
        result = score_attribution_map(response, chunks)
        assert result["phantom_count"] >= 0
        assert isinstance(result["phantom_chunk_ids"], list)

    def test_output_structure(self):
        response = "Some response text."
        chunks = [{"id": 1, "text": "Some context text about IVF clinic treatment."}]
        result = score_attribution_map(response, chunks)
        assert "attributed_chunks" in result
        assert "phantom_chunk_ids" in result
        assert "total_chunks_retrieved" in result
        assert "attributed_count" in result
        assert "phantom_count" in result
        assert "coverage_gap" in result
        assert "attribution_summary" in result
