"""Probability, ranking, calibration, and error-analysis utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_binary_inputs(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional target and probability arrays."""
    target_array = np.asarray(target)
    probability_array = np.asarray(probabilities, dtype=float)

    if target_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("Target and probabilities must be one-dimensional.")

    if len(target_array) != len(probability_array):
        raise ValueError("Target and probabilities must have equal length.")

    if len(target_array) == 0:
        raise ValueError("Evaluation input must not be empty.")

    if not np.isin(target_array, [0, 1]).all():
        raise ValueError("Target must contain only binary values zero and one.")

    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must contain only finite values.")

    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilities must lie within the interval [0, 1].")

    if len(np.unique(target_array)) != 2:
        raise ValueError("Evaluation requires both target classes.")

    return target_array.astype(np.int8), probability_array


def ranking_metrics_at_review_rate(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    *,
    review_rate: float,
) -> dict[str, float | int | bool]:
    """Calculate tie-aware ranking metrics at a fixed capacity rate.

    When the capacity boundary falls inside a group of identical scores, the
    captured-fraud count is the expected value from random selection within
    that tied group. This prevents input row order from creating artificial
    ranking performance for constant-score models such as a prior dummy.
    """
    target_array, probability_array = _validate_binary_inputs(
        target,
        probabilities,
    )

    if not 0 < review_rate <= 1:
        raise ValueError("Review rate must lie within the interval (0, 1].")

    review_count = min(
        len(target_array),
        max(1, int(np.ceil(len(target_array) * review_rate))),
    )
    ranked_probabilities = np.sort(probability_array)[::-1]
    score_cutoff = float(ranked_probabilities[review_count - 1])
    above_cutoff_mask = probability_array > score_cutoff
    at_cutoff_mask = probability_array == score_cutoff
    above_cutoff_count = int(above_cutoff_mask.sum())
    cutoff_tie_count = int(at_cutoff_mask.sum())
    cutoff_tie_review_count = review_count - above_cutoff_count

    if not 0 < cutoff_tie_review_count <= cutoff_tie_count:
        raise RuntimeError("Invalid score-tie accounting at review cutoff.")

    captured_above_cutoff = float(target_array[above_cutoff_mask].sum())
    fraud_at_cutoff = float(target_array[at_cutoff_mask].sum())
    expected_captured_at_cutoff = (
        cutoff_tie_review_count / cutoff_tie_count
    ) * fraud_at_cutoff
    expected_captured_fraud = (
        captured_above_cutoff + expected_captured_at_cutoff
    )
    total_fraud = int(target_array.sum())

    return {
        "requested_review_rate": float(review_rate),
        "actual_review_rate": float(review_count / len(target_array)),
        "review_count": review_count,
        "captured_fraud_count": float(expected_captured_fraud),
        "captured_fraud_count_is_expected": (
            cutoff_tie_review_count < cutoff_tie_count
        ),
        "total_fraud_count": total_fraud,
        "precision_at_capacity": float(
            expected_captured_fraud / review_count
        ),
        "recall_at_capacity": float(
            expected_captured_fraud / total_fraud
        ),
        "fraud_capture_rate": float(
            expected_captured_fraud / total_fraud
        ),
        "score_cutoff": score_cutoff,
        "cutoff_tie_count": cutoff_tie_count,
        "cutoff_tie_review_count": cutoff_tie_review_count,
    }


def expected_calibration_error(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    *,
    number_of_bins: int = 10,
) -> float:
    """Calculate equal-width expected calibration error."""
    target_array, probability_array = _validate_binary_inputs(
        target,
        probabilities,
    )

    if number_of_bins < 2:
        raise ValueError("Calibration evaluation requires at least two bins.")

    bin_indexes = np.minimum(
        (probability_array * number_of_bins).astype(int),
        number_of_bins - 1,
    )
    calibration_error = 0.0

    for bin_index in range(number_of_bins):
        bin_mask = bin_indexes == bin_index

        if not bin_mask.any():
            continue

        observed_rate = float(target_array[bin_mask].mean())
        mean_probability = float(probability_array[bin_mask].mean())
        bin_weight = float(bin_mask.mean())
        calibration_error += bin_weight * abs(
            observed_rate - mean_probability
        )

    return float(calibration_error)


def calibration_table(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    *,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    """Return equal-width calibration-bin statistics for plotting."""
    target_array, probability_array = _validate_binary_inputs(
        target,
        probabilities,
    )

    if number_of_bins < 2:
        raise ValueError("Calibration evaluation requires at least two bins.")

    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    bin_indexes = np.minimum(
        (probability_array * number_of_bins).astype(int),
        number_of_bins - 1,
    )
    records: list[dict[str, float | int]] = []

    for bin_index in range(number_of_bins):
        bin_mask = bin_indexes == bin_index

        if not bin_mask.any():
            continue

        records.append(
            {
                "bin": bin_index,
                "lower_bound": float(bin_edges[bin_index]),
                "upper_bound": float(bin_edges[bin_index + 1]),
                "row_count": int(bin_mask.sum()),
                "mean_predicted_probability": float(
                    probability_array[bin_mask].mean()
                ),
                "observed_fraud_rate": float(
                    target_array[bin_mask].mean()
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def evaluate_probabilities(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    *,
    review_rates: Sequence[float],
    diagnostic_threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, object]:
    """Evaluate discrimination, ranking, and uncalibrated probabilities.

    The diagnostic threshold is not a selected business threshold. Threshold
    selection remains outside Phase 3 and must not use the untouched test set.
    """
    target_array, probability_array = _validate_binary_inputs(
        target,
        probabilities,
    )

    if not 0 <= diagnostic_threshold <= 1:
        raise ValueError(
            "Diagnostic threshold must lie within the interval [0, 1]."
        )

    predictions = (probability_array >= diagnostic_threshold).astype(np.int8)
    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(target_array, predictions, labels=[0, 1]).ravel()
    )
    review_metrics = {
        f"{float(review_rate):.4f}": ranking_metrics_at_review_rate(
            target_array,
            probability_array,
            review_rate=float(review_rate),
        )
        for review_rate in review_rates
    }

    return {
        "row_count": len(target_array),
        "positive_count": int(target_array.sum()),
        "prevalence": float(target_array.mean()),
        "pr_auc_average_precision": float(
            average_precision_score(target_array, probability_array)
        ),
        "roc_auc": float(roc_auc_score(target_array, probability_array)),
        "brier_score": float(
            brier_score_loss(target_array, probability_array)
        ),
        "log_loss": float(
            log_loss(target_array, probability_array, labels=[0, 1])
        ),
        "expected_calibration_error": expected_calibration_error(
            target_array,
            probability_array,
            number_of_bins=calibration_bins,
        ),
        "diagnostic_threshold": float(diagnostic_threshold),
        "diagnostic_precision": float(
            precision_score(target_array, predictions, zero_division=0)
        ),
        "diagnostic_recall": float(
            recall_score(target_array, predictions, zero_division=0)
        ),
        "diagnostic_f1": float(
            f1_score(target_array, predictions, zero_division=0)
        ),
        "diagnostic_confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
        "review_rate_metrics": review_metrics,
    }


def build_error_analysis_frame(
    feature_frame: pd.DataFrame,
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    *,
    diagnostic_threshold: float = 0.5,
    examples_per_error_type: int = 50,
) -> pd.DataFrame:
    """Return the highest-confidence false positives and false negatives."""
    if not isinstance(feature_frame, pd.DataFrame):
        raise TypeError("feature_frame must be a pandas DataFrame.")

    target_array, probability_array = _validate_binary_inputs(
        target,
        probabilities,
    )

    if len(feature_frame) != len(target_array):
        raise ValueError("Feature and evaluation row counts must match.")

    if not 0 <= diagnostic_threshold <= 1:
        raise ValueError(
            "Diagnostic threshold must lie within the interval [0, 1]."
        )

    if examples_per_error_type < 1:
        raise ValueError("examples_per_error_type must be positive.")

    predictions = (probability_array >= diagnostic_threshold).astype(np.int8)
    analysis = feature_frame.reset_index(names="source_index").copy()
    analysis.insert(1, "target", target_array)
    analysis.insert(2, "fraud_probability", probability_array)
    analysis.insert(3, "diagnostic_prediction", predictions)
    analysis.insert(4, "error_type", "correct")

    false_positive_mask = (target_array == 0) & (predictions == 1)
    false_negative_mask = (target_array == 1) & (predictions == 0)
    analysis.loc[false_positive_mask, "error_type"] = "false_positive"
    analysis.loc[false_negative_mask, "error_type"] = "false_negative"

    false_positives = (
        analysis.loc[analysis["error_type"] == "false_positive"]
        .nlargest(examples_per_error_type, "fraud_probability")
    )
    false_negatives = (
        analysis.loc[analysis["error_type"] == "false_negative"]
        .nsmallest(examples_per_error_type, "fraud_probability")
    )

    return pd.concat(
        [false_positives, false_negatives],
        ignore_index=True,
    )
