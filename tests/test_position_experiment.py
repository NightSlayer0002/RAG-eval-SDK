import pytest

from rag_eval_sdk.position_experiment import analyse_position_effect, create_position_variants


def test_position_variants_only_move_target():
    chunks = [
        {"id": "a", "text": "Alpha evidence"},
        {"id": "b", "text": "Beta target evidence"},
        {"id": "c", "text": "Gamma evidence"},
        {"id": "d", "text": "Delta evidence"},
    ]
    variants = create_position_variants(chunks, "b")
    assert [variant.treatment for variant in variants] == ["beginning", "middle", "end"]
    assert [variant.target_index for variant in variants] == [0, 1, 3]
    for variant in variants:
        assert [chunk.id for chunk in variant.chunks if chunk.id != "b"] == ["a", "c", "d"]


def test_position_effect_uses_paired_score_gap():
    effect = analyse_position_effect(
        {"beginning": 0.9, "middle": 0.5, "end": 0.8},
        sensitivity_threshold=0.2,
    )
    assert effect.position_sensitive is True
    assert effect.preferred_position == "beginning"
    assert effect.middle_drop == pytest.approx(0.4)

