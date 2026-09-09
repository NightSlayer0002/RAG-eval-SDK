from rag_eval_sdk.calibration import binary_metrics, ranking_metrics, select_threshold


def test_threshold_is_selected_from_validation_scores():
    result = select_threshold(
        [0.05, 0.10, 0.80, 0.90], [False, False, True, True]
    )
    assert 0.10 < result.threshold <= 0.80
    assert result.metrics.f1 == 1.0
    assert result.calibration_id.startswith("threshold-grid-v1:")


def test_ranking_metrics_perfect_order():
    result = ranking_metrics([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert result.auroc == 1.0
    assert result.average_precision == 1.0


def test_binary_metrics_support_lower_is_positive():
    result = binary_metrics(
        [0.1, 0.9], [True, False], 0.5, positive_when="lower"
    )
    assert result.accuracy == 1.0

