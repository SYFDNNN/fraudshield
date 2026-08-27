"""Capacity-based review thresholds and risk bands for FraudShield."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.evaluate import ranking_metrics_at_review_rate


@dataclass(frozen=True, slots=True)
class CapacityThresholdPolicy:
    """One review-capacity cutoff selected on a development split."""

    source_split: str
    requested_review_rate: float
    row_count: int
    target_review_count: int
    score_threshold: float
    strictly_above_threshold_count: int
    boundary_tie_count: int
    boundary_review_slots: int
    observed_at_or_above_threshold_count: int
    observed_at_or_above_threshold_rate: float
    expected_captured_fraud_count: float
    expected_precision_at_capacity: float
    expected_recall_at_capacity: float
    tie_break_required: bool
    tie_break_policy: str


@dataclass(frozen=True, slots=True)
class RiskBandBoundary:
    """Upper cumulative-capacity boundary for one risk band."""

    label: str
    cumulative_review_rate: float
    score_threshold: float
    observed_at_or_above_count: int
    observed_at_or_above_rate: float
    boundary_tie_count: int


@dataclass(frozen=True, slots=True)
class RiskBandPolicy:
    """Ordered development cutoffs used to label calibrated scores."""

    source_split: str
    boundaries: tuple[RiskBandBoundary, ...]
    default_label: str
    tie_break_policy: str

    @property
    def labels(self) -> tuple[str, ...]:
        """Return labels from highest to lowest risk."""
        return tuple(
            boundary.label for boundary in self.boundaries
        ) + (self.default_label,)


def _probability_array(
    probabilities: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Return a validated one-dimensional probability array."""
    probability_array = np.asarray(probabilities, dtype=float)

    if probability_array.ndim != 1 or len(probability_array) == 0:
        raise ValueError("Probabilities must be a non-empty 1D array.")

    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must contain only finite values.")

    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilities must lie within [0, 1].")

    return probability_array


def _score_threshold(
    probabilities: np.ndarray,
    cumulative_rate: float,
) -> tuple[float, int]:
    """Return the top-capacity score cutoff and requested row count."""
    if not 0 < cumulative_rate <= 1:
        raise ValueError("Cumulative review rate must lie within (0, 1].")

    review_count = min(
        len(probabilities),
        max(1, int(np.ceil(len(probabilities) * cumulative_rate))),
    )
    threshold = float(np.sort(probabilities)[::-1][review_count - 1])

    return threshold, review_count


def select_capacity_threshold(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    *,
    review_rate: float,
    source_split: str = "validation",
    tie_break_policy: str = (
        "calibrated_score_desc_then_raw_score_desc_then_"
        "stable_application_key"
    ),
) -> CapacityThresholdPolicy:
    """Select one auditable top-capacity review cutoff."""
    probability_array = _probability_array(probabilities)
    ranking_metrics = ranking_metrics_at_review_rate(
        target,
        probability_array,
        review_rate=review_rate,
    )
    threshold = float(ranking_metrics["score_cutoff"])
    strictly_above_count = int((probability_array > threshold).sum())
    at_or_above_count = int((probability_array >= threshold).sum())
    boundary_tie_count = int(ranking_metrics["cutoff_tie_count"])
    boundary_review_slots = int(
        ranking_metrics["cutoff_tie_review_count"]
    )

    return CapacityThresholdPolicy(
        source_split=source_split,
        requested_review_rate=float(review_rate),
        row_count=len(probability_array),
        target_review_count=int(ranking_metrics["review_count"]),
        score_threshold=threshold,
        strictly_above_threshold_count=strictly_above_count,
        boundary_tie_count=boundary_tie_count,
        boundary_review_slots=boundary_review_slots,
        observed_at_or_above_threshold_count=at_or_above_count,
        observed_at_or_above_threshold_rate=float(
            at_or_above_count / len(probability_array)
        ),
        expected_captured_fraud_count=float(
            ranking_metrics["captured_fraud_count"]
        ),
        expected_precision_at_capacity=float(
            ranking_metrics["precision_at_capacity"]
        ),
        expected_recall_at_capacity=float(
            ranking_metrics["recall_at_capacity"]
        ),
        tie_break_required=(boundary_review_slots < boundary_tie_count),
        tie_break_policy=tie_break_policy,
    )


def build_risk_band_policy(
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    band_config: Sequence[Mapping[str, Any]],
    *,
    source_split: str = "validation",
    tie_break_policy: str = (
        "calibrated_score_desc_then_raw_score_desc_then_"
        "stable_application_key"
    ),
) -> RiskBandPolicy:
    """Build ordered score bands from cumulative review capacities."""
    probability_array = _probability_array(probabilities)

    if len(band_config) < 2:
        raise ValueError("At least two risk bands must be configured.")

    labels: list[str] = []
    cumulative_rates: list[float] = []

    for band in band_config:
        if not isinstance(band, Mapping):
            raise TypeError("Every risk-band definition must be a mapping.")

        label = band.get("label")
        cumulative_rate = float(band.get("cumulative_review_rate", -1))

        if not isinstance(label, str) or not label.strip():
            raise ValueError("Every risk band requires a non-empty label.")

        if not 0 < cumulative_rate <= 1:
            raise ValueError(
                "Risk-band cumulative rates must lie within (0, 1]."
            )

        labels.append(label)
        cumulative_rates.append(cumulative_rate)

    if len(labels) != len(set(labels)):
        raise ValueError("Risk-band labels must be unique.")

    if cumulative_rates != sorted(cumulative_rates):
        raise ValueError("Risk-band cumulative rates must be increasing.")

    if len(cumulative_rates) != len(set(cumulative_rates)):
        raise ValueError("Risk-band cumulative rates must be unique.")

    if cumulative_rates[-1] != 1.0:
        raise ValueError("The final risk band must end at rate 1.0.")

    boundaries: list[RiskBandBoundary] = []

    for label, cumulative_rate in zip(
        labels[:-1],
        cumulative_rates[:-1],
        strict=True,
    ):
        threshold, _ = _score_threshold(
            probability_array,
            cumulative_rate,
        )
        at_or_above_count = int(
            (probability_array >= threshold).sum()
        )
        boundaries.append(
            RiskBandBoundary(
                label=label,
                cumulative_review_rate=cumulative_rate,
                score_threshold=threshold,
                observed_at_or_above_count=at_or_above_count,
                observed_at_or_above_rate=float(
                    at_or_above_count / len(probability_array)
                ),
                boundary_tie_count=int(
                    (probability_array == threshold).sum()
                ),
            )
        )

    return RiskBandPolicy(
        source_split=source_split,
        boundaries=tuple(boundaries),
        default_label=labels[-1],
        tie_break_policy=tie_break_policy,
    )


def assign_risk_bands(
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    policy: RiskBandPolicy,
) -> np.ndarray:
    """Assign the configured ordered label to each calibrated score."""
    probability_array = _probability_array(probabilities)
    labels = np.full(
        len(probability_array),
        policy.default_label,
        dtype=object,
    )
    unassigned = np.ones(len(probability_array), dtype=bool)

    for boundary in policy.boundaries:
        boundary_mask = (
            unassigned
            & (probability_array >= boundary.score_threshold)
        )
        labels[boundary_mask] = boundary.label
        unassigned[boundary_mask] = False

    return labels


def risk_band_summary(
    target: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    policy: RiskBandPolicy,
) -> pd.DataFrame:
    """Summarize rows and observed fraud for every configured band."""
    probability_array = _probability_array(probabilities)
    target_array = np.asarray(target)

    if target_array.ndim != 1 or len(target_array) != len(probability_array):
        raise ValueError("Target and probabilities must be equal-length 1D arrays.")

    if not np.isin(target_array, [0, 1]).all():
        raise ValueError("Target must contain only zero and one.")

    assigned_labels = assign_risk_bands(probability_array, policy)
    total_fraud = int(target_array.sum())
    records: list[dict[str, Any]] = []

    for label in policy.labels:
        band_mask = assigned_labels == label
        row_count = int(band_mask.sum())
        fraud_count = int(target_array[band_mask].sum())
        band_probabilities = probability_array[band_mask]
        records.append(
            {
                "risk_band": label,
                "row_count": row_count,
                "row_share": float(row_count / len(target_array)),
                "fraud_count": fraud_count,
                "fraud_share": float(
                    fraud_count / total_fraud if total_fraud else 0.0
                ),
                "fraud_prevalence": float(
                    fraud_count / row_count if row_count else 0.0
                ),
                "minimum_probability": float(
                    band_probabilities.min() if row_count else np.nan
                ),
                "maximum_probability": float(
                    band_probabilities.max() if row_count else np.nan
                ),
                "mean_probability": float(
                    band_probabilities.mean() if row_count else np.nan
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def threshold_review_flags(
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    policy: CapacityThresholdPolicy,
) -> np.ndarray:
    """Return review flags using the stored cutoff, including boundary ties."""
    probability_array = _probability_array(probabilities)
    return (probability_array >= policy.score_threshold).astype(np.int8)


def capacity_review_flags(
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    *,
    review_rate: float,
    secondary_scores: Sequence[float] | np.ndarray | pd.Series | None = None,
    stable_keys: Sequence[Any] | np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Return an exact top-capacity batch using deterministic tie-breakers.

    Calibrated probability is the primary ranking key. A raw model score may
    be supplied as the secondary key, which preserves ordering inside the
    piecewise-constant groups produced by isotonic calibration. Stable keys
    are the final deterministic tie-breaker and must be unique.
    """
    probability_array = _probability_array(probabilities)

    if not 0 < review_rate <= 1:
        raise ValueError("Review rate must lie within (0, 1].")

    if secondary_scores is None:
        secondary_array = probability_array
    else:
        secondary_array = np.asarray(secondary_scores, dtype=float)

        if (
            secondary_array.ndim != 1
            or len(secondary_array) != len(probability_array)
            or not np.isfinite(secondary_array).all()
        ):
            raise ValueError(
                "Secondary scores must be finite and match probabilities."
            )

    if stable_keys is None:
        stable_key_array = np.arange(len(probability_array))
    else:
        stable_key_array = np.asarray(stable_keys)

        if (
            stable_key_array.ndim != 1
            or len(stable_key_array) != len(probability_array)
        ):
            raise ValueError("Stable keys must match probabilities.")

        if len(np.unique(stable_key_array)) != len(stable_key_array):
            raise ValueError("Stable keys must be unique.")

    review_count = min(
        len(probability_array),
        max(1, int(np.ceil(len(probability_array) * review_rate))),
    )
    ranking = np.lexsort(
        (stable_key_array, -secondary_array, -probability_array)
    )
    flags = np.zeros(len(probability_array), dtype=np.int8)
    flags[ranking[:review_count]] = 1

    return flags
