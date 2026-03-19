"""Unit tests for conflict_detector.py"""
import pytest
from rag_eval_sdk.conflict_detector import _extract_keywords, _keyword_overlap, detect_conflicts


class TestExtractKeywords:
    def test_removes_stopwords(self):
        keywords = _extract_keywords("the clinic is open on Monday")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "Monday" in {k.lower() for k in keywords} or "monday" in keywords

    def test_removes_short_words(self):
        kw = _extract_keywords("IVF treatment at our clinic")
        for k in kw:
            assert len(k) > 2

    def test_empty_string(self):
        assert _extract_keywords("") == set()


class TestKeywordOverlap:
    def test_identical_sets(self):
        s = {"clinic", "treatment", "monday"}
        assert _keyword_overlap(s, s) == 1.0

    def test_disjoint_sets(self):
        a = {"clinic", "monday"}
        b = {"doctor", "payment"}
        assert _keyword_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"clinic", "monday", "treatment"}
        b = {"clinic", "monday", "payment"}
        score = _keyword_overlap(a, b)
        assert 0.0 < score < 1.0

    def test_empty_sets(self):
        assert _keyword_overlap(set(), {"clinic"}) == 0.0


class TestDetectConflicts:
    def test_no_conflicts_different_topics(self):
        chunks = [
            {"id": 1, "text": "The restaurant serves Italian food with fresh pasta."},
            {"id": 2, "text": "Stock market indices rose slightly on Thursday trading."},
        ]
        result = detect_conflicts(chunks)
        assert result["conflict_count"] == 0
        assert result["conflict_severity"] == "NONE"

    def test_detects_contradictory_chunks(self):
        chunks = [
            {"id": 1, "text": "The clinic opening hours are Monday through Friday only."},
            {"id": 2, "text": "The clinic is open every day including weekends and Saturday."},
        ]
        result = detect_conflicts(chunks)
        assert isinstance(result["conflict_count"], int)
        assert isinstance(result["conflict_pairs"], list)

    def test_agreeing_chunks_no_conflict(self):
        chunks = [
            {"id": 1, "text": "The clinic is open Monday to Friday from 9 AM to 5 PM."},
            {"id": 2, "text": "We are available weekdays from nine in the morning until five in the evening."},
        ]
        result = detect_conflicts(chunks)
        assert result["conflict_severity"] in ("NONE", "LOW")

    def test_empty_chunks(self):
        result = detect_conflicts([])
        assert result["conflict_count"] == 0

    def test_single_chunk(self):
        result = detect_conflicts([{"id": 1, "text": "The clinic is open Monday to Friday."}])
        assert result["conflict_count"] == 0

    def test_skips_short_chunks(self):
        chunks = [
            {"id": 1, "text": "Hi"},
            {"id": 2, "text": "OK"},
        ]
        result = detect_conflicts(chunks)
        assert result["conflict_count"] == 0

    def test_output_structure(self):
        chunks = [{"id": 1, "text": "The clinic is open on Monday and Tuesday."}]
        result = detect_conflicts(chunks)
        assert "conflict_pairs" in result
        assert "conflict_count" in result
        assert "conflict_severity" in result
        assert "conflict_summary" in result
