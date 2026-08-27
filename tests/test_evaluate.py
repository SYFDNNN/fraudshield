"""Tests for ranking, probability, and baseline error metrics."""

import pandas as pd
import pytest

from fraudshield.evaluate import (
    build_error_analysis_frame,
    calibration_table,
    evaluate_probabilities,
    ranking_metrics_at_review_rate,
)


def test_ranking_metrics_respect_review_capacity() -> None:
    """Top-k metrics should count only the highest-scored applications."""
    target = [1, 0, 1, 0]
    probabilities = [0.9, 0.8, 0.7, 0.1]

    metrics = ranking_metrics_at_review_rate(
        target,
        probabilities,
        review_rate=0.5,
    )

    assert metrics["review_count"] == 2
    assert metrics["captured_fraud_count"] == 1
    assert metrics["precision_at_capacity"] == pytest.approx(0.5)
    assert metrics["recall_at_capacity"] == pytest.approx(0.5)
    assert metrics["fraud_capture_rate"] == pytest.approx(0.5)
    assert metrics["score_cutoff"] == pytest.approx(0.8)
    assert metrics["captured_fraud_count_is_expected"] is False


def test_ranking_metrics_are_tie_aware_for_constant_scores() -> None:
    """A constant-score model must not gain performance from row ordering."""
    metrics = ranking_metrics_at_review_rate(
        [1, 0, 0, 0],
        [0.2, 0.2, 0.2, 0.2],
        review_rate=0.5,
    )

    assert metrics["review_count"] == 2
    assert metrics["captured_fraud_count"] == pytest.approx(0.5)
    assert metrics["precision_at_capacity"] == pytest.approx(0.25)
    assert metrics["recall_at_capacity"] == pytest.approx(0.5)
    assert metrics["captured_fraud_count_is_expected"] is True
    assert metrics["cutoff_tie_count"] == 4
    assert metrics["cutoff_tie_review_count"] == 2


def test_probability_evaluation_returns_required_baseline_metrics() -> None:
    """Phase 3 evaluation should include discrimination and calibration."""
    metrics = evaluate_probabilities(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.7, 0.1],
        review_rates=[0.5],
        diagnostic_threshold=0.5,
        calibration_bins=2,
    )

    assert metrics["pr_auc_average_precision"] == pytest.approx(
        0.8333333333
    )
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["expected_calibration_error"] <= 1
    assert metrics["diagnostic_confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 2,
    }
    assert "0.5000" in metrics["review_rate_metrics"]


def test_calibration_table_accounts_for_every_row() -> None:
    """Non-empty calibration bins must preserve the evaluation row count."""
    table = calibration_table(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.7, 0.1],
        number_of_bins=5,
    )

    assert table["row_count"].sum() == 4
    assert table["mean_predicted_probability"].between(0, 1).all()
    assert table["observed_fraud_rate"].between(0, 1).all()


def test_error_analysis_returns_both_error_types() -> None:
    """Error samples should include confident false positives and negatives."""
    feature_frame = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
        },
        index=[100, 101, 102, 103],
    )
    errors = build_error_analysis_frame(
        feature_frame,
        [1, 0, 1, 0],
        [0.4, 0.9, 0.8, 0.1],
        diagnostic_threshold=0.5,
        examples_per_error_type=5,
    )

    assert set(errors["error_type"]) == {
        "false_positive",
        "false_negative",
    }
    assert set(errors["source_index"]) == {100, 101}
