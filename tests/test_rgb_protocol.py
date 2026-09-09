from benchmarks.rgb_official import build_rgb_documents, score_rgb_prediction


def test_counterfactual_uses_positive_wrong_documents():
    row = {
        "positive": ["Paris is the capital."],
        "positive_wrong": ["London is the capital."],
        "negative": ["Unrelated text."],
    }
    documents = build_rgb_documents(
        row,
        dataset_name="en_fact",
        noise_rate=0.0,
        passage_count=1,
        correct_rate=0.0,
        seed=2333,
    )
    assert documents == ["London is the capital."]


def test_nested_integration_answers_require_every_component():
    answer = [["Paris"], ["France", "French Republic"]]
    assert score_rgb_prediction("Paris is in France.", answer)["answer_correct"] is True
    assert score_rgb_prediction("Paris is a city.", answer)["answer_correct"] is False

