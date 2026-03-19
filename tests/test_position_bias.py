"""Unit tests for position_bias_detector.py"""
import pytest
from rag_eval_sdk.position_bias_detector import detect_position_bias


class TestDetectPositionBias:
    def test_no_bias_uniform_usage(self):
        chunks = [
            {"id": i, "text": "The clinic provides IVF treatment for infertile couples seeking fertility solutions."}
            for i in range(1, 7)
        ]
        response = "The clinic provides IVF treatment for infertile couples seeking fertility solutions."
        scores = [0.8] * 6
        result = detect_position_bias(response, chunks, scores)
        assert "bias_detected" in result
        assert "bias_type" in result

    def test_too_few_chunks(self):
        chunks = [
            {"id": 1, "text": "The clinic is open Monday to Friday."},
            {"id": 2, "text": "Appointments available all week."},
        ]
        result = detect_position_bias("Clinic is open weekdays.", chunks, [0.8, 0.7])
        assert result["bias_detected"] is False
        assert "Too few chunks" in result["bias_summary"]

    def test_empty_chunks(self):
        result = detect_position_bias("No context available.", [], [])
        assert result["bias_detected"] is False

    def test_output_structure(self):
        chunks = [{"id": i, "text": f"Chunk {i} text about IVF treatment and fertility clinic."} for i in range(1, 5)]
        result = detect_position_bias("IVF treatment text.", chunks, [0.8, 0.7, 0.6, 0.5])
        assert "position_analysis" in result
        assert "bias_detected" in result
        assert "bias_type" in result
        assert "bias_confidence" in result
        assert "middle_phantom_ratio" in result
        assert "bias_summary" in result
        assert "recommended_action" in result

    def test_position_bias_with_middle_ignored(self):
        chunks = [
            {"id": 1, "text": "IVF treatment overview and fertility clinic services."},
            {"id": 2, "text": "Stock market trading completely unrelated to medical topics."},
            {"id": 3, "text": "Weather forecast for tomorrow rain and sunshine expected."},
            {"id": 4, "text": "IVF clinic appointment booking and treatment schedules."},
        ]
        response = "IVF treatment overview and fertility clinic appointment booking services."
        scores = [0.9, 0.9, 0.9, 0.9]
        result = detect_position_bias(response, chunks, scores)
        assert isinstance(result["bias_detected"], bool)
        assert isinstance(result["middle_phantom_ratio"], float)

    def test_no_retriever_scores(self):
        chunks = [{"id": i, "text": f"IVF treatment clinic services chunk {i}."} for i in range(1, 5)]
        result = detect_position_bias("IVF clinic treatment.", chunks, None)
        assert "bias_detected" in result
        assert result["wasted_good_chunks"] == 0
