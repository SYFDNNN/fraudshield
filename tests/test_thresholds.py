"""Tests for Phase 5 capacity thresholds and risk bands."""

import numpy as np
import pandas as pd
import pytest

from fraudshield.thresholds import (
    assign_risk_bands,
    build_risk_band_policy,
    capacity_review_flags,
    risk_band_summary,
    select_capacity_threshold,
    threshold_review_flags,
)


def test_capacity_threshold_matches_requested_top_k() -> None:
    """A unique boundary should produce the exact requested review count."""
    policy = select_capacity_threshold(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.7, 0.1],
        review_rate=0.5,
    )

    assert policy.target_review_count == 2
    assert policy.score_threshold == pytest.approx(0.8)
    assert policy.observed_at_or_above_threshold_count == 2
    assert policy.expected_precision_at_capacity == pytest.approx(0.5)
    assert policy.expected_recall_at_capacity == pytest.approx(0.5)
    assert policy.tie_break_required is False


def test_capacity_threshold_records_boundary_tie() -> None:
    """A cutoff tie must be explicit because a threshold can exceed capacity."""
    policy = select_capacity_threshold(
        [1, 0, 0, 0],
        [0.2, 0.2, 0.2, 0.2],
        review_rate=0.5,
    )

    assert policy.target_review_count == 2
    assert policy.boundary_tie_count == 4
    assert policy.boundary_review_slots == 2
    assert policy.observed_at_or_above_threshold_count == 4
    assert policy.tie_break_required is True
    assert policy.expected_captured_fraud_count == pytest.approx(0.5)


def test_risk_band_policy_assigns_ordered_capacity_labels() -> None:
    """Scores should be assigned from the highest boundary downward."""
    probabilities = np.asarray([0.99, 0.80, 0.60, 0.40, 0.20, 0.01])
    policy = build_risk_band_policy(
        probabilities,
        [
            {"label": "sangat_tinggi", "cumulative_review_rate": 1 / 6},
            {"label": "tinggi", "cumulative_review_rate": 3 / 6},
            {"label": "rendah", "cumulative_review_rate": 1.0},
        ],
    )

    labels = assign_risk_bands(probabilities, policy)

    assert labels.tolist() == [
        "sangat_tinggi",
        "tinggi",
        "tinggi",
        "rendah",
        "rendah",
        "rendah",
    ]
    assert policy.labels == ("sangat_tinggi", "tinggi", "rendah")


def test_risk_band_summary_preserves_rows_and_fraud() -> None:
    """Band summaries must account for the full validation population."""
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.2])
    target = pd.Series([1, 0, 1, 0])
    policy = build_risk_band_policy(
        probabilities,
        [
            {"label": "tinggi", "cumulative_review_rate": 0.5},
            {"label": "rendah", "cumulative_review_rate": 1.0},
        ],
    )

    summary = risk_band_summary(target, probabilities, policy)

    assert summary["row_count"].sum() == 4
    assert summary["fraud_count"].sum() == 2
    assert summary["row_share"].sum() == pytest.approx(1.0)
    assert summary["fraud_share"].sum() == pytest.approx(1.0)


def test_risk_band_policy_rejects_incomplete_coverage() -> None:
    """The final risk band must cover all remaining applications."""
    with pytest.raises(ValueError, match="final risk band"):
        build_risk_band_policy(
            [0.9, 0.2],
            [
                {"label": "tinggi", "cumulative_review_rate": 0.5},
                {"label": "rendah", "cumulative_review_rate": 0.9},
            ],
        )


def test_threshold_flags_include_boundary_ties() -> None:
    """Stored threshold flags should disclose their tie-inclusive behavior."""
    probabilities = np.asarray([0.9, 0.5, 0.5, 0.1])
    policy = select_capacity_threshold(
        [1, 0, 1, 0],
        probabilities,
        review_rate=0.5,
    )

    flags = threshold_review_flags(probabilities, policy)

    assert flags.tolist() == [1, 1, 1, 0]
    assert policy.tie_break_required is True


def test_capacity_review_flags_apply_exact_deterministic_tie_break() -> None:
    """Batch recommendations must not exceed capacity when scores tie."""
    flags = capacity_review_flags(
        [0.5, 0.5, 0.5, 0.1],
        review_rate=0.5,
        secondary_scores=[0.7, 0.9, 0.8, 0.1],
        stable_keys=[10, 11, 12, 13],
    )

    assert flags.tolist() == [0, 1, 1, 0]
    assert flags.sum() == 2
