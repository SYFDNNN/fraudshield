"""One-time untouched-test evaluation for FraudShield Phase 6."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from fraudshield.calibrate import CalibratedFraudModel
from fraudshield.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
)
from fraudshield.data import load_base_dataset, select_model_features
from fraudshield.evaluate import (
    calibration_table,
    evaluate_probabilities,
)
from fraudshield.thresholds import capacity_review_flags, risk_band_summary

_PHASE = 6
_COMPLETION_FILENAME = "final_evaluation_completion.json"
_FREEZE_FILENAME = "freeze_manifest.json"
_STATE_FILENAME = "evaluation_state.json"
_RESULT_ARTIFACT_NAMES = (
    "test_metrics",
    "fixed_threshold_metrics",
    "capacity_metrics",
    "bootstrap_intervals",
    "temporal_comparison",
    "stability_assessment",
    "risk_band_summary",
    "slice_metrics",
    "error_analysis",
    "calibration_table",
    "predictions",
    "metadata",
)


@dataclass(slots=True)
class Phase6FinalEvaluationResult:
    """Stored outputs from the one allowed untouched-test evaluation."""

    test_metrics: dict[str, dict[str, object]]
    fixed_threshold_metrics: dict[str, object]
    capacity_metrics: dict[str, object]
    bootstrap_intervals: pd.DataFrame
    temporal_comparison: pd.DataFrame
    stability_assessment: pd.DataFrame
    risk_band_summary: pd.DataFrame
    slice_metrics: pd.DataFrame
    error_analysis: pd.DataFrame
    metadata: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    reused_existing_result: bool = False


def _utc_now() -> str:
    """Return an ISO-formatted UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    """Write one deterministic UTF-8 JSON artifact."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON mapping and fail closed on malformed content."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain a mapping: {path}")

    return payload


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    file_path = Path(path)
    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    """Return a stable digest for one JSON-serializable payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase5_paths(
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Path]:
    """Resolve every frozen Phase 5 artifact required by Phase 6."""
    final_config = config["final_evaluation"]
    configured_paths = {
        "model": final_config["phase5_model_artifact"],
        "metadata": final_config["phase5_metadata_artifact"],
        "metrics": final_config["phase5_metrics_artifact"],
        "decision": final_config["phase5_decision_artifact"],
        "capacity_policy": final_config["phase5_capacity_artifact"],
        "risk_band_policy": final_config["phase5_risk_band_artifact"],
    }

    return {
        name: resolve_project_path(value, config_path=config_path)
        for name, value in configured_paths.items()
    }


def _phase6_paths(
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Path]:
    """Return every Phase 6 output path."""
    artifact_directory = resolve_project_path(
        config["artifacts"]["phase6_directory"],
        config_path=config_path,
    )
    return {
        "directory": artifact_directory,
        "freeze_manifest": artifact_directory / _FREEZE_FILENAME,
        "evaluation_state": artifact_directory / _STATE_FILENAME,
        "completion": artifact_directory / _COMPLETION_FILENAME,
        "test_metrics": artifact_directory / "final_test_metrics.json",
        "fixed_threshold_metrics": (
            artifact_directory / "fixed_threshold_metrics.json"
        ),
        "capacity_metrics": artifact_directory / "capacity_metrics.json",
        "bootstrap_intervals": (
            artifact_directory / "bootstrap_intervals.csv"
        ),
        "temporal_comparison": (
            artifact_directory / "validation_test_comparison.csv"
        ),
        "stability_assessment": (
            artifact_directory / "stability_assessment.csv"
        ),
        "risk_band_summary": (
            artifact_directory / "test_risk_bands.csv"
        ),
        "slice_metrics": artifact_directory / "test_slice_metrics.csv",
        "error_analysis": artifact_directory / "test_error_analysis.csv",
        "calibration_table": (
            artifact_directory / "test_calibration_table.csv"
        ),
        "predictions": artifact_directory / "test_predictions.parquet",
        "metadata": artifact_directory / "phase6_metadata.json",
    }


def _validate_final_configuration(config: dict[str, Any]) -> None:
    """Validate the pre-registered one-time evaluation policy."""
    final_config = config["final_evaluation"]
    test_periods = tuple(config["split"]["test_periods"])
    evaluation_periods = tuple(final_config["evaluation_periods"])

    if evaluation_periods != test_periods:
        raise ValueError(
            "Final evaluation periods must exactly match reserved test "
            "periods."
        )

    development_periods = {
        period
        for split_name in (
            "train_periods",
            "calibration_periods",
            "validation_periods",
        )
        for period in config["split"][split_name]
    }

    if development_periods & set(evaluation_periods):
        raise ValueError("Final test periods overlap development periods.")

    if final_config["completed_result_policy"] != (
        "reuse_without_test_re_evaluation"
    ):
        raise ValueError(
            "Completed Phase 6 results must be reused without re-evaluating "
            "test."
        )

    if config["review_policy"]["automated_rejection_allowed"] is not False:
        raise ValueError("Final evaluation does not allow automated rejection.")

    if float(config["review_policy"]["capacity_rate"]) != float(
        config["threshold_selection"]["capacity_rate"]
    ):
        raise ValueError("Review and threshold capacity rates must match.")

    bootstrap_config = final_config["bootstrap"]
    confidence_level = float(bootstrap_config["confidence_level"])
    number_of_resamples = int(bootstrap_config["number_of_resamples"])

    if bootstrap_config["method"] != "stratified_percentile":
        raise ValueError("Phase 6 requires stratified percentile bootstrap.")

    if not 0 < confidence_level < 1:
        raise ValueError("Bootstrap confidence level must lie within (0, 1).")

    if number_of_resamples < 20:
        raise ValueError("Bootstrap requires at least 20 resamples.")


def _locked_policy_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return the configuration subset frozen before test access."""
    return {
        "split": config["split"],
        "feature_policy": config["feature_policy"],
        "xgboost": config["xgboost"],
        "calibration": config["calibration"],
        "threshold_selection": config["threshold_selection"],
        "evaluation": config["evaluation"],
        "review_policy": config["review_policy"],
        "final_evaluation": config["final_evaluation"],
    }


def validate_frozen_phase5_artifacts(
    config: dict[str, Any],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[
    CalibratedFraudModel,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    """Load and verify the exact Phase 5 candidate before test access."""
    paths = _phase5_paths(config, config_path=config_path)
    missing = sorted(name for name, path in paths.items() if not path.is_file())

    if missing:
        raise FileNotFoundError(
            "Required Phase 5 artifacts are missing: " + ", ".join(missing)
        )

    metadata = _read_json(paths["metadata"])
    metrics = _read_json(paths["metrics"])
    decision = _read_json(paths["decision"])
    capacity_policy = _read_json(paths["capacity_policy"])
    risk_band_policy = _read_json(paths["risk_band_policy"])

    required_metadata = {
        "test_evaluated": False,
        "test_features_exposed": False,
        "base_model_refit_with_later_periods": False,
        "preprocessing_fitted_on": "train_periods_only",
        "calibrator_fitted_on": "calibration_periods_only",
        "calibrator_selected_on": "validation_periods_only",
        "threshold_selected_on": "validation_periods_only",
        "business_threshold_selected": True,
        "automated_rejection_allowed": False,
    }

    for key, expected_value in required_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Phase 5 metadata guardrail failed for {key!r}: "
                f"expected {expected_value!r}, received {metadata.get(key)!r}."
            )

    model = joblib.load(paths["model"])

    if not isinstance(model, CalibratedFraudModel):
        raise TypeError(
            "Phase 5 model artifact is not a CalibratedFraudModel."
        )

    expected_model_name = config["calibration"]["base_model_name"]

    if metadata.get("base_model_name") != expected_model_name:
        raise ValueError("Frozen Phase 5 base model differs from configuration.")

    selected_calibrator = metadata.get("selected_calibrator_name")

    if model.calibrator_name != selected_calibrator:
        raise ValueError("Calibrator in model and Phase 5 metadata differ.")

    if decision.get("selected_calibrator_name") != selected_calibrator:
        raise ValueError("Calibrator in decision and metadata differ.")

    configured_capacity = float(
        config["threshold_selection"]["capacity_rate"]
    )

    if not np.isclose(
        model.capacity_policy.requested_review_rate,
        configured_capacity,
    ):
        raise ValueError("Frozen model capacity differs from configuration.")

    if not np.isclose(
        float(capacity_policy["requested_review_rate"]),
        configured_capacity,
    ):
        raise ValueError("Capacity artifact differs from configuration.")

    configured_labels = tuple(
        band["label"] for band in config["threshold_selection"]["risk_bands"]
    )

    if model.risk_band_policy.labels != configured_labels:
        raise ValueError("Frozen model risk bands differ from configuration.")

    artifact_labels = tuple(
        boundary["label"] for boundary in risk_band_policy["boundaries"]
    ) + (risk_band_policy["default_label"],)

    if artifact_labels != configured_labels:
        raise ValueError("Risk-band artifact differs from configuration.")

    if selected_calibrator not in metrics:
        raise KeyError("Selected calibrator is absent from Phase 5 metrics.")

    artifact_hashes = {
        name: sha256_file(path) for name, path in paths.items()
    }

    return model, metadata, metrics, artifact_hashes


def build_freeze_manifest(
    config: dict[str, Any],
    *,
    phase5_metadata: dict[str, Any],
    phase5_artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build the immutable policy record written before test access."""
    locked_policy = _locked_policy_payload(config)
    return {
        "phase": _PHASE,
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "test_access_started": False,
        "test_evaluated": False,
        "evaluation_periods": config["final_evaluation"][
            "evaluation_periods"
        ],
        "base_model_name": phase5_metadata["base_model_name"],
        "selected_calibrator_name": phase5_metadata[
            "selected_calibrator_name"
        ],
        "capacity_rate": float(config["review_policy"]["capacity_rate"]),
        "fixed_threshold_source": "validation_periods_only",
        "risk_band_source": "validation_periods_only",
        "locked_policy": locked_policy,
        "locked_policy_sha256": _payload_sha256(locked_policy),
        "phase5_artifact_sha256": phase5_artifact_hashes,
        "no_refit_after_test": True,
        "no_recalibration_after_test": True,
        "no_threshold_reselection_after_test": True,
        "automated_rejection_allowed": False,
    }


def _prepare_or_validate_freeze(
    paths: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write the first freeze or validate an interrupted locked attempt."""
    freeze_path = paths["freeze_manifest"]

    if not freeze_path.is_file():
        _write_json(freeze_path, manifest)
        return manifest

    existing = _read_json(freeze_path)

    for key in ("locked_policy_sha256", "phase5_artifact_sha256"):
        if existing.get(key) != manifest.get(key):
            raise RuntimeError(
                "Frozen Phase 6 policy differs from current artifacts or "
                "configuration. Test evaluation is blocked."
            )

    return existing


def _extract_locked_test_frame(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Expose only the pre-registered final test rows after the freeze."""
    data_config = config["data"]
    split_config = config["split"]
    time_column = data_config["time_column"]
    target_column = data_config["target_column"]
    test_periods = tuple(config["final_evaluation"]["evaluation_periods"])

    if time_column not in dataframe or target_column not in dataframe:
        raise KeyError("Final dataset lacks time or target column.")

    configured_periods = {
        period
        for split_name in (
            "train_periods",
            "calibration_periods",
            "validation_periods",
            "test_periods",
        )
        for period in split_config[split_name]
    }
    observed_periods = set(dataframe[time_column].unique().tolist())

    if observed_periods != configured_periods:
        raise ValueError(
            "Observed periods differ from the locked temporal split."
        )

    test_frame = dataframe.loc[
        dataframe[time_column].isin(test_periods)
    ].copy()

    if test_frame.empty:
        raise ValueError("Locked final test split is empty.")

    stable_keys = test_frame.index.to_numpy(copy=True)

    if len(np.unique(stable_keys)) != len(stable_keys):
        raise ValueError("Test source indexes must be unique stable keys.")

    test_features = select_model_features(
        test_frame,
        excluded_columns=config["feature_policy"]["excluded_from_model"],
    )
    test_target = test_frame[target_column].astype(np.int8).copy()

    return test_features, test_target, stable_keys


def binary_review_metrics(
    target: Any,
    review_flags: Any,
) -> dict[str, object]:
    """Evaluate one already-frozen binary review policy."""
    target_array = np.asarray(target)
    flag_array = np.asarray(review_flags)

    if target_array.ndim != 1 or flag_array.ndim != 1:
        raise ValueError("Target and review flags must be one-dimensional.")

    if len(target_array) == 0 or len(target_array) != len(flag_array):
        raise ValueError("Target and review flags must have equal non-zero length.")

    if not np.isin(target_array, [0, 1]).all():
        raise ValueError("Target must contain only zero and one.")

    if not np.isin(flag_array, [0, 1]).all():
        raise ValueError("Review flags must contain only zero and one.")

    target_array = target_array.astype(np.int8)
    flag_array = flag_array.astype(np.int8)
    reviewed = flag_array == 1
    fraud = target_array == 1
    true_positive = int((reviewed & fraud).sum())
    false_positive = int((reviewed & ~fraud).sum())
    false_negative = int((~reviewed & fraud).sum())
    true_negative = int((~reviewed & ~fraud).sum())
    review_count = int(reviewed.sum())
    total_fraud = int(fraud.sum())

    return {
        "row_count": len(target_array),
        "review_count": review_count,
        "review_rate": float(review_count / len(target_array)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": float(
            true_positive / review_count if review_count else 0.0
        ),
        "recall": float(
            true_positive / total_fraud if total_fraud else 0.0
        ),
        "fraud_capture_rate": float(
            true_positive / total_fraud if total_fraud else 0.0
        ),
    }


def bootstrap_metric_intervals(
    target: Any,
    probabilities: Any,
    *,
    review_rate: float,
    number_of_resamples: int,
    confidence_level: float,
    random_seed: int,
    secondary_scores: Any | None = None,
    stable_keys: Any | None = None,
) -> pd.DataFrame:
    """Estimate intervals using the exact locked capacity-ranking policy."""
    target_array = np.asarray(target, dtype=np.int8)
    probability_array = np.asarray(probabilities, dtype=float)

    if (
        target_array.ndim != 1
        or probability_array.ndim != 1
        or len(target_array) != len(probability_array)
        or len(target_array) == 0
    ):
        raise ValueError("Bootstrap inputs must be equal-length non-empty arrays.")

    if len(np.unique(target_array)) != 2:
        raise ValueError("Bootstrap requires both target classes.")

    if number_of_resamples < 20:
        raise ValueError("Bootstrap requires at least 20 resamples.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie within (0, 1).")

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
            "Bootstrap secondary scores must be finite and match rows."
        )

    if stable_keys is None:
        stable_key_array = np.arange(len(target_array))
    else:
        stable_key_array = np.asarray(stable_keys)

    if (
        stable_key_array.ndim != 1
        or len(stable_key_array) != len(target_array)
        or len(np.unique(stable_key_array)) != len(stable_key_array)
    ):
        raise ValueError("Bootstrap stable keys must be unique and match rows.")

    negative_indexes = np.flatnonzero(target_array == 0)
    positive_indexes = np.flatnonzero(target_array == 1)
    random_generator = np.random.default_rng(random_seed)
    samples: dict[str, list[float]] = {
        "average_precision": [],
        "roc_auc": [],
        "brier_score": [],
        "precision_at_capacity": [],
        "recall_at_capacity": [],
    }

    def calculate_metrics(
        sampled_target: np.ndarray,
        sampled_probability: np.ndarray,
        sampled_secondary: np.ndarray,
        sampled_stable_keys: np.ndarray,
    ) -> dict[str, float]:
        capacity_flags = capacity_review_flags(
            sampled_probability,
            review_rate=review_rate,
            secondary_scores=sampled_secondary,
            stable_keys=sampled_stable_keys,
        )
        capacity = binary_review_metrics(sampled_target, capacity_flags)
        return {
            "average_precision": float(
                average_precision_score(
                    sampled_target,
                    sampled_probability,
                )
            ),
            "roc_auc": float(
                roc_auc_score(sampled_target, sampled_probability)
            ),
            "brier_score": float(
                brier_score_loss(sampled_target, sampled_probability)
            ),
            "precision_at_capacity": float(
                capacity["precision"]
            ),
            "recall_at_capacity": float(capacity["recall"]),
        }

    point_estimates = calculate_metrics(
        target_array,
        probability_array,
        secondary_array,
        stable_key_array,
    )

    for _ in range(number_of_resamples):
        sampled_indexes = np.concatenate(
            [
                random_generator.choice(
                    negative_indexes,
                    size=len(negative_indexes),
                    replace=True,
                ),
                random_generator.choice(
                    positive_indexes,
                    size=len(positive_indexes),
                    replace=True,
                ),
            ]
        )
        random_generator.shuffle(sampled_indexes)
        sampled_metrics = calculate_metrics(
            target_array[sampled_indexes],
            probability_array[sampled_indexes],
            secondary_array[sampled_indexes],
            np.arange(len(sampled_indexes)),
        )

        for metric_name, metric_value in sampled_metrics.items():
            samples[metric_name].append(metric_value)

    tail_probability = (1.0 - confidence_level) / 2.0
    records = []

    for metric_name, metric_samples in samples.items():
        records.append(
            {
                "metric": metric_name,
                "point_estimate": point_estimates[metric_name],
                "ci_lower": float(
                    np.quantile(metric_samples, tail_probability)
                ),
                "ci_upper": float(
                    np.quantile(metric_samples, 1.0 - tail_probability)
                ),
                "confidence_level": float(confidence_level),
                "number_of_resamples": int(number_of_resamples),
                "method": "stratified_percentile",
            }
        )

    return pd.DataFrame.from_records(records)


def _slice_record(
    *,
    feature_name: str,
    value: str,
    mask: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    review_flags: np.ndarray,
    minimum_rows: int,
    minimum_positive_rows: int,
) -> dict[str, object]:
    """Calculate one descriptive test slice without changing the model."""
    slice_target = target[mask]
    slice_probability = probabilities[mask]
    slice_review = review_flags[mask]
    row_count = int(mask.sum())
    fraud_count = int(slice_target.sum())
    non_fraud_count = row_count - fraud_count
    reviewed_count = int(slice_review.sum())
    reviewed_fraud_count = int(
        ((slice_review == 1) & (slice_target == 1)).sum()
    )
    eligible = (
        row_count >= minimum_rows
        and fraud_count >= minimum_positive_rows
        and non_fraud_count >= minimum_positive_rows
    )

    return {
        "slice_feature": feature_name,
        "slice_value": value,
        "row_count": row_count,
        "row_share": float(row_count / len(target)),
        "fraud_count": fraud_count,
        "prevalence": float(fraud_count / row_count),
        "mean_probability": float(slice_probability.mean()),
        "calibration_gap": float(
            slice_probability.mean() - slice_target.mean()
        ),
        "brier_score": float(
            brier_score_loss(slice_target, slice_probability)
        ),
        "average_precision": float(
            average_precision_score(slice_target, slice_probability)
        )
        if eligible
        else np.nan,
        "roc_auc": float(roc_auc_score(slice_target, slice_probability))
        if eligible
        else np.nan,
        "reviewed_count": reviewed_count,
        "review_rate": float(reviewed_count / row_count),
        "review_precision": float(
            reviewed_fraud_count / reviewed_count if reviewed_count else 0.0
        ),
        "within_slice_fraud_capture": float(
            reviewed_fraud_count / fraud_count if fraud_count else 0.0
        ),
        "eligible_for_discrimination_metrics": eligible,
    }


def build_slice_metrics(
    features: pd.DataFrame,
    target: Any,
    probabilities: Any,
    review_flags: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build pre-registered descriptive category and numeric-bin slices."""
    target_array = np.asarray(target, dtype=np.int8)
    probability_array = np.asarray(probabilities, dtype=float)
    review_array = np.asarray(review_flags, dtype=np.int8)
    slice_config = config["final_evaluation"]["slices"]

    if not (
        len(features)
        == len(target_array)
        == len(probability_array)
        == len(review_array)
    ):
        raise ValueError("Slice inputs must have equal row counts.")

    minimum_rows = int(slice_config["minimum_rows"])
    minimum_positive_rows = int(slice_config["minimum_positive_rows"])
    records: list[dict[str, object]] = []

    for feature_name in slice_config["categorical_features"]:
        if feature_name not in features:
            raise KeyError(f"Configured slice feature is missing: {feature_name}")

        values = features[feature_name].astype("string").fillna("<missing>")

        for value in sorted(values.unique().tolist()):
            mask = values.eq(value).to_numpy()
            records.append(
                _slice_record(
                    feature_name=feature_name,
                    value=str(value),
                    mask=mask,
                    target=target_array,
                    probabilities=probability_array,
                    review_flags=review_array,
                    minimum_rows=minimum_rows,
                    minimum_positive_rows=minimum_positive_rows,
                )
            )

    for feature_name, edges in slice_config["numeric_bins"].items():
        if feature_name not in features:
            raise KeyError(f"Configured slice feature is missing: {feature_name}")

        numeric_edges = [float(edge) for edge in edges]

        if numeric_edges != sorted(numeric_edges):
            raise ValueError(f"Numeric slice edges are unordered: {feature_name}")

        if len(numeric_edges) != len(set(numeric_edges)):
            raise ValueError(f"Numeric slice edges repeat: {feature_name}")

        values = pd.cut(
            features[feature_name],
            bins=numeric_edges,
            right=False,
            include_lowest=True,
        )

        if values.isna().any():
            raise ValueError(
                f"Numeric slice bins do not cover {feature_name}."
            )

        for value in values.cat.categories:
            mask = values.eq(value).to_numpy()

            if not mask.any():
                continue

            records.append(
                _slice_record(
                    feature_name=feature_name,
                    value=str(value),
                    mask=mask,
                    target=target_array,
                    probabilities=probability_array,
                    review_flags=review_array,
                    minimum_rows=minimum_rows,
                    minimum_positive_rows=minimum_positive_rows,
                )
            )

    return pd.DataFrame.from_records(records).sort_values(
        ["slice_feature", "slice_value"],
        kind="mergesort",
        ignore_index=True,
    )


def build_capacity_error_analysis(
    features: pd.DataFrame,
    target: Any,
    probabilities: Any,
    review_flags: Any,
    *,
    examples_per_type: int = 25,
) -> pd.DataFrame:
    """Return representative reviewed non-fraud and missed-fraud rows."""
    target_array = np.asarray(target, dtype=np.int8)
    probability_array = np.asarray(probabilities, dtype=float)
    review_array = np.asarray(review_flags, dtype=np.int8)

    if examples_per_type < 1:
        raise ValueError("examples_per_type must be positive.")

    if not (
        len(features)
        == len(target_array)
        == len(probability_array)
        == len(review_array)
    ):
        raise ValueError("Error-analysis inputs must have equal row counts.")

    analysis = features.reset_index(names="source_index").copy()
    analysis.insert(1, "target", target_array)
    analysis.insert(2, "fraud_probability", probability_array)
    analysis.insert(3, "review_recommended", review_array)
    reviewed_non_fraud = analysis.loc[
        (target_array == 0) & (review_array == 1)
    ].nlargest(examples_per_type, "fraud_probability")
    reviewed_non_fraud = reviewed_non_fraud.assign(
        error_type="reviewed_non_fraud"
    )
    missed_fraud = analysis.loc[
        (target_array == 1) & (review_array == 0)
    ]
    missed_near_boundary = missed_fraud.nlargest(
        examples_per_type,
        "fraud_probability",
    ).assign(error_type="missed_fraud_near_boundary")
    missed_low_score = missed_fraud.nsmallest(
        examples_per_type,
        "fraud_probability",
    ).assign(error_type="missed_fraud_low_score")

    return pd.concat(
        [reviewed_non_fraud, missed_near_boundary, missed_low_score],
        ignore_index=True,
    )


def build_temporal_comparison(
    phase5_metrics: dict[str, Any],
    selected_calibrator_name: str,
    test_metrics: dict[str, object],
    *,
    review_rate: float,
) -> pd.DataFrame:
    """Compare frozen validation evidence with the one final test result."""
    validation_metrics = phase5_metrics[selected_calibrator_name]["validation"]
    review_key = f"{review_rate:.4f}"
    rows = []

    for split_name, metrics in (
        ("validation", validation_metrics),
        ("test", test_metrics),
    ):
        capacity = metrics["review_rate_metrics"][review_key]
        rows.append(
            {
                "split": split_name,
                "row_count": metrics["row_count"],
                "positive_count": metrics["positive_count"],
                "prevalence": metrics["prevalence"],
                "average_precision": metrics["pr_auc_average_precision"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "expected_calibration_error": metrics[
                    "expected_calibration_error"
                ],
                "precision_at_capacity": capacity["precision_at_capacity"],
                "recall_at_capacity": capacity["recall_at_capacity"],
            }
        )

    return pd.DataFrame.from_records(rows)


def build_stability_assessment(
    temporal_comparison: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply pre-registered reporting alerts without changing the model."""
    indexed = temporal_comparison.set_index("split")

    if set(indexed.index) != {"validation", "test"}:
        raise ValueError("Stability assessment requires validation and test.")

    alerts = config["final_evaluation"]["stability_alerts"]
    definitions = (
        (
            "average_precision",
            "drop",
            float(alerts["maximum_average_precision_drop"]),
        ),
        (
            "recall_at_capacity",
            "drop",
            float(alerts["maximum_recall_at_capacity_drop"]),
        ),
        (
            "brier_score",
            "increase",
            float(alerts["maximum_brier_score_increase"]),
        ),
        (
            "expected_calibration_error",
            "increase",
            float(
                alerts[
                    "maximum_expected_calibration_error_increase"
                ]
            ),
        ),
    )
    records = []

    for metric_name, direction, maximum_allowed_change in definitions:
        validation_value = float(indexed.loc["validation", metric_name])
        test_value = float(indexed.loc["test", metric_name])
        signed_change = test_value - validation_value
        adverse_change = (
            -signed_change if direction == "drop" else signed_change
        )
        records.append(
            {
                "metric": metric_name,
                "validation_value": validation_value,
                "test_value": test_value,
                "test_minus_validation": signed_change,
                "adverse_direction": direction,
                "maximum_allowed_adverse_change": maximum_allowed_change,
                "alert_triggered": adverse_change > maximum_allowed_change,
                "action": "report_and_investigate_only",
            }
        )

    return pd.DataFrame.from_records(records)


def _evaluate_locked_test(
    model: CalibratedFraudModel,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    stable_keys: np.ndarray,
    phase5_metrics: dict[str, Any],
    phase5_metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate the complete pre-registered test report without fitting."""
    expected_features = list(
        model.base_model.named_steps[
            "feature_selection"
        ].selected_columns_
    )

    if test_features.columns.tolist() != expected_features:
        raise ValueError("Test feature schema differs from frozen model schema.")

    raw_probability = model.raw_probability(test_features)
    calibrated_probability = model.fraud_probability(test_features)
    fixed_threshold_flags = model.at_or_above_threshold(test_features)
    capacity_flags = model.recommend_review(
        test_features,
        stable_keys=stable_keys,
    )
    risk_bands = model.risk_band(test_features)
    evaluation_config = config["evaluation"]
    test_metrics = {
        "uncalibrated": evaluate_probabilities(
            test_target,
            raw_probability,
            review_rates=evaluation_config["review_rates"],
            diagnostic_threshold=evaluation_config[
                "diagnostic_threshold"
            ],
            calibration_bins=evaluation_config["calibration_bins"],
        ),
        "calibrated": evaluate_probabilities(
            test_target,
            calibrated_probability,
            review_rates=evaluation_config["review_rates"],
            diagnostic_threshold=model.capacity_policy.score_threshold,
            calibration_bins=evaluation_config["calibration_bins"],
        ),
    }
    fixed_metrics = binary_review_metrics(
        test_target,
        fixed_threshold_flags,
    )
    fixed_metrics.update(
        {
            "policy": "fixed_validation_score_threshold",
            "score_threshold": float(model.capacity_policy.score_threshold),
            "threshold_source_split": model.capacity_policy.source_split,
            "tie_inclusive": True,
        }
    )
    capacity_metrics = binary_review_metrics(test_target, capacity_flags)
    capacity_metrics.update(
        {
            "policy": "exact_batch_capacity",
            "requested_review_rate": float(
                model.capacity_policy.requested_review_rate
            ),
            "ranking_primary": "calibrated_probability_desc",
            "ranking_secondary": "raw_probability_desc",
            "ranking_final": "stable_source_index",
        }
    )
    final_config = config["final_evaluation"]
    bootstrap_config = final_config["bootstrap"]
    bootstrap_intervals = bootstrap_metric_intervals(
        test_target,
        calibrated_probability,
        review_rate=float(config["review_policy"]["capacity_rate"]),
        number_of_resamples=int(bootstrap_config["number_of_resamples"]),
        confidence_level=float(bootstrap_config["confidence_level"]),
        random_seed=int(bootstrap_config["random_seed"]),
        secondary_scores=raw_probability,
        stable_keys=stable_keys,
    )
    temporal_comparison = build_temporal_comparison(
        phase5_metrics,
        phase5_metadata["selected_calibrator_name"],
        test_metrics["calibrated"],
        review_rate=float(config["review_policy"]["capacity_rate"]),
    )
    stability_assessment = build_stability_assessment(
        temporal_comparison,
        config,
    )
    risk_summary = risk_band_summary(
        test_target,
        calibrated_probability,
        model.risk_band_policy,
    )
    slice_metrics = build_slice_metrics(
        test_features,
        test_target,
        calibrated_probability,
        capacity_flags,
        config,
    )
    error_analysis = build_capacity_error_analysis(
        test_features,
        test_target,
        calibrated_probability,
        capacity_flags,
        examples_per_type=int(
            final_config.get("error_examples_per_type", 25)
        ),
    )
    calibration_tables = []

    for method_name, probability in (
        ("uncalibrated", raw_probability),
        ("calibrated", calibrated_probability),
    ):
        calibration_tables.append(
            calibration_table(
                test_target,
                probability,
                number_of_bins=evaluation_config["calibration_bins"],
            ).assign(probability_method=method_name)
        )

    prediction_frame = pd.DataFrame(
        {
            "source_index": stable_keys,
            "target": test_target.to_numpy(),
            "raw_probability": raw_probability,
            "fraud_probability": calibrated_probability,
            "fixed_threshold_review": fixed_threshold_flags,
            "capacity_review": capacity_flags,
            "risk_band": risk_bands,
        }
    )

    return {
        "test_metrics": test_metrics,
        "fixed_threshold_metrics": fixed_metrics,
        "capacity_metrics": capacity_metrics,
        "bootstrap_intervals": bootstrap_intervals,
        "temporal_comparison": temporal_comparison,
        "stability_assessment": stability_assessment,
        "risk_band_summary": risk_summary,
        "slice_metrics": slice_metrics,
        "error_analysis": error_analysis,
        "calibration_table": pd.concat(
            calibration_tables,
            ignore_index=True,
        ),
        "predictions": prediction_frame,
    }


def _save_evaluation_artifacts(
    outputs: dict[str, Any],
    paths: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """Write every result artifact; completion is written separately last."""
    _write_json(paths["test_metrics"], outputs["test_metrics"])
    _write_json(
        paths["fixed_threshold_metrics"],
        outputs["fixed_threshold_metrics"],
    )
    _write_json(paths["capacity_metrics"], outputs["capacity_metrics"])
    outputs["bootstrap_intervals"].to_csv(
        paths["bootstrap_intervals"],
        index=False,
    )
    outputs["temporal_comparison"].to_csv(
        paths["temporal_comparison"],
        index=False,
    )
    outputs["stability_assessment"].to_csv(
        paths["stability_assessment"],
        index=False,
    )
    outputs["risk_band_summary"].to_csv(
        paths["risk_band_summary"],
        index=False,
    )
    outputs["slice_metrics"].to_csv(paths["slice_metrics"], index=False)
    outputs["error_analysis"].to_csv(paths["error_analysis"], index=False)
    outputs["calibration_table"].to_csv(
        paths["calibration_table"],
        index=False,
    )
    outputs["predictions"].to_parquet(paths["predictions"], index=False)
    _write_json(paths["metadata"], metadata)


def _result_artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Hash every completed result artifact except state and completion."""
    return {
        name: sha256_file(paths[name]) for name in _RESULT_ARTIFACT_NAMES
    }


def _validate_result_artifact_hashes(
    paths: dict[str, Path],
    expected_hashes: dict[str, str],
) -> None:
    """Fail closed unless the complete stored result set is unchanged."""
    if set(expected_hashes) != set(_RESULT_ARTIFACT_NAMES):
        raise RuntimeError("Completed Phase 6 artifact manifest is incomplete.")

    for name, expected_hash in expected_hashes.items():
        if not paths[name].is_file() or sha256_file(paths[name]) != expected_hash:
            raise RuntimeError(f"Completed Phase 6 artifact changed: {name}")


def _load_completed_result(
    paths: dict[str, Path],
    *,
    expected_locked_policy_sha256: str,
    expected_phase5_hashes: dict[str, str],
) -> Phase6FinalEvaluationResult:
    """Reuse stored final evidence without touching the test dataset again."""
    completion = _read_json(paths["completion"])

    if completion.get("status") != "completed":
        raise RuntimeError("Phase 6 completion artifact is not completed.")

    if completion.get("locked_policy_sha256") != (
        expected_locked_policy_sha256
    ):
        raise RuntimeError("Completed result uses a different locked policy.")

    if completion.get("phase5_artifact_sha256") != expected_phase5_hashes:
        raise RuntimeError("Completed result uses different Phase 5 artifacts.")

    expected_result_hashes = completion.get("result_artifact_sha256", {})

    if not isinstance(expected_result_hashes, dict):
        raise TypeError("Completed Phase 6 artifact hashes are malformed.")

    _validate_result_artifact_hashes(paths, expected_result_hashes)

    return Phase6FinalEvaluationResult(
        test_metrics=_read_json(paths["test_metrics"]),
        fixed_threshold_metrics=_read_json(
            paths["fixed_threshold_metrics"]
        ),
        capacity_metrics=_read_json(paths["capacity_metrics"]),
        bootstrap_intervals=pd.read_csv(paths["bootstrap_intervals"]),
        temporal_comparison=pd.read_csv(paths["temporal_comparison"]),
        stability_assessment=pd.read_csv(paths["stability_assessment"]),
        risk_band_summary=pd.read_csv(paths["risk_band_summary"]),
        slice_metrics=pd.read_csv(paths["slice_metrics"]),
        error_analysis=pd.read_csv(paths["error_analysis"]),
        metadata=_read_json(paths["metadata"]),
        artifact_paths={
            name: path for name, path in paths.items() if name != "directory"
        },
        reused_existing_result=True,
    )


def _recover_completed_result(
    paths: dict[str, Path],
    state: dict[str, Any],
    *,
    expected_locked_policy_sha256: str,
    expected_phase5_hashes: dict[str, str],
) -> Phase6FinalEvaluationResult:
    """Finalize an interrupted completion write without reopening test."""
    if state.get("status") != "results_written_awaiting_completion":
        raise RuntimeError(
            "An earlier Phase 6 attempt already started test access. "
            "Automatic test re-evaluation is blocked. Audit the local "
            "artifacts before recovery."
        )

    if state.get("locked_policy_sha256") != expected_locked_policy_sha256:
        raise RuntimeError("Interrupted result uses a different locked policy.")

    if state.get("phase5_artifact_sha256") != expected_phase5_hashes:
        raise RuntimeError("Interrupted result uses different Phase 5 artifacts.")

    result_hashes = state.get("result_artifact_sha256")

    if not isinstance(result_hashes, dict):
        raise TypeError("Interrupted Phase 6 result lacks artifact hashes.")

    _validate_result_artifact_hashes(paths, result_hashes)
    metadata = _read_json(paths["metadata"])

    if (
        metadata.get("test_evaluation_count") != 1
        or metadata.get("locked_policy_sha256")
        != expected_locked_policy_sha256
        or metadata.get("phase5_artifact_sha256")
        != expected_phase5_hashes
    ):
        raise RuntimeError("Interrupted Phase 6 metadata is inconsistent.")

    completion = {
        "phase": _PHASE,
        "status": "completed",
        "completed_at_utc": state["completed_at_utc"],
        "test_evaluated": True,
        "test_evaluation_count": 1,
        "reuse_policy": "reuse_without_test_re_evaluation",
        "locked_policy_sha256": expected_locked_policy_sha256,
        "phase5_artifact_sha256": expected_phase5_hashes,
        "result_artifact_sha256": result_hashes,
        "completion_recovered_without_test_access": True,
    }
    _write_json(paths["completion"], completion)
    return _load_completed_result(
        paths,
        expected_locked_policy_sha256=expected_locked_policy_sha256,
        expected_phase5_hashes=expected_phase5_hashes,
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a small Markdown table without optional dependencies."""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def render_model_card(result: Phase6FinalEvaluationResult) -> str:
    """Render a factual Indonesian model card from stored final evidence."""
    temporal_rows = []

    for row in result.temporal_comparison.to_dict(orient="records"):
        temporal_rows.append(
            [
                str(row["split"]),
                f"{int(row['row_count']):,}",
                f"{float(row['prevalence']):.4%}",
                f"{float(row['average_precision']):.6f}",
                f"{float(row['roc_auc']):.6f}",
                f"{float(row['brier_score']):.6f}",
                f"{float(row['expected_calibration_error']):.6f}",
                f"{float(row['precision_at_capacity']):.6f}",
                f"{float(row['recall_at_capacity']):.6f}",
            ]
        )

    interval_rows = [
        [
            str(row["metric"]),
            f"{float(row['point_estimate']):.6f}",
            f"{float(row['ci_lower']):.6f}",
            f"{float(row['ci_upper']):.6f}",
        ]
        for row in result.bootstrap_intervals.to_dict(orient="records")
    ]
    risk_rows = [
        [
            str(row["risk_band"]),
            f"{int(row['row_count']):,}",
            f"{float(row['row_share']):.4%}",
            f"{int(row['fraud_count']):,}",
            f"{float(row['fraud_share']):.4%}",
            f"{float(row['fraud_prevalence']):.4%}",
        ]
        for row in result.risk_band_summary.to_dict(orient="records")
    ]
    fixed = result.fixed_threshold_metrics
    capacity = result.capacity_metrics
    alerts_triggered = int(
        result.stability_assessment["alert_triggered"].astype(bool).sum()
    )
    metadata = result.metadata

    return f"""# Model Card FraudShield

## Status

Kandidat final telah dievaluasi satu kali pada untouched test month 7. Model
belum merupakan sistem produksi dan tetap memerlukan human review.

## Model dan kebijakan

- Model dasar: `{metadata['base_model_name']}`.
- Calibrator: `{metadata['selected_calibrator_name']}`.
- Train model dasar: month 0–4.
- Fit calibrator: month 5.
- Seleksi model, calibrator, threshold, dan risk band: month 6.
- Evaluasi final satu kali: month 7.
- Kapasitas review: `{float(capacity['requested_review_rate']):.2%}`.
- Automated rejection: **tidak diizinkan**.

## Intended use

Model memberi probabilitas risiko dan prioritas antrean untuk membantu fraud
operations analyst melakukan review manual pada aplikasi pembukaan rekening.

## Out-of-scope use

- Menolak atau menerima aplikasi secara otomatis.
- Menggantikan investigasi manusia.
- Digunakan pada populasi atau proses bank nyata tanpa validasi baru.
- Menafsirkan score sebagai bukti bahwa seseorang melakukan fraud.

## Performa temporal

{_markdown_table(
    [
        'Split', 'Rows', 'Prevalence', 'AP', 'ROC-AUC', 'Brier', 'ECE',
        'Precision@5%', 'Recall@5%'
    ],
    temporal_rows,
)}

## Ketidakpastian test

Interval berikut menggunakan stratified percentile bootstrap
`{int(result.bootstrap_intervals.iloc[0]['number_of_resamples'])}` resample.

{_markdown_table(['Metric', 'Estimate', 'CI lower', 'CI upper'], interval_rows)}

## Kebijakan review pada test

| Kebijakan | Review count | Review rate | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Fixed validation threshold `{float(fixed['score_threshold']):.6f}` | {int(fixed['review_count']):,} | {float(fixed['review_rate']):.4%} | {float(fixed['precision']):.6f} | {float(fixed['recall']):.6f} |
| Exact batch capacity | {int(capacity['review_count']):,} | {float(capacity['review_rate']):.4%} | {float(capacity['precision']):.6f} | {float(capacity['recall']):.6f} |

Fixed threshold mengukur transfer cutoff month 6 ke month 7. Exact batch
capacity mempertahankan beban review 5% dengan ranking yang telah dikunci dan
bukan threshold baru hasil tuning test.

## Risk band pada test

{_markdown_table(
    ['Risk band', 'Rows', 'Row share', 'Fraud', 'Fraud share', 'Prevalence'],
    risk_rows,
)}

## Stability dan slice

Jumlah reporting alert yang terpicu: `{alerts_triggered}`. Alert hanya memicu
investigasi dan dokumentasi; hasil test tidak boleh dipakai untuk menyesuaikan
model ini lalu mengklaim evaluasi pada month 7 sebagai hasil test yang baru.

Metrik slice lengkap tersedia pada artifact lokal
`artifacts/phase6/test_slice_metrics.csv`. Slice bersifat diagnostic dan tidak
membuktikan fairness atau absence of bias.

## Keterbatasan

- Dataset bersifat sintetis dan bukan bukti performa pada bank nyata.
- Hanya satu periode test yang tersedia.
- Class imbalance membuat accuracy tidak informatif sebagai metrik utama.
- Calibration dan risk band dapat berubah ketika prevalensi atau distribusi
  populasi berubah.
- Fraud tetap terdapat pada band rendah; band tidak boleh menjadi aturan
  penerimaan otomatis.
- Confidence interval hanya menggambarkan sampling uncertainty pada test ini,
  bukan seluruh uncertainty deployment.

## Reproducibility dan governance

- Freeze policy SHA-256: `{metadata['locked_policy_sha256']}`.
- Model artifact SHA-256: `{metadata['phase5_artifact_sha256']['model']}`.
- Source dataset SHA-256: `{metadata['source_dataset_sha256']}`.
- Test evaluation count: `1`.
- Refit setelah test: `False`.
- Recalibration setelah test: `False`.
- Threshold reselection setelah test: `False`.
"""


def _write_model_card(
    result: Phase6FinalEvaluationResult,
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> Path:
    """Write the final factual model card to the configured report path."""
    model_card_path = resolve_project_path(
        config["reports"]["model_card_path"],
        config_path=config_path,
    )
    model_card_path.write_text(
        render_model_card(result),
        encoding="utf-8",
    )
    return model_card_path


def run_phase6_final_evaluation(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Phase6FinalEvaluationResult:
    """Evaluate untouched test once, then reuse immutable stored evidence."""
    config = load_config(config_path)
    _validate_final_configuration(config)
    paths = _phase6_paths(config, config_path=config_path)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    (
        model,
        phase5_metadata,
        phase5_metrics,
        phase5_hashes,
    ) = validate_frozen_phase5_artifacts(config, config_path=config_path)
    manifest = build_freeze_manifest(
        config,
        phase5_metadata=phase5_metadata,
        phase5_artifact_hashes=phase5_hashes,
    )
    frozen_manifest = _prepare_or_validate_freeze(paths, manifest)

    if paths["completion"].is_file():
        result = _load_completed_result(
            paths,
            expected_locked_policy_sha256=frozen_manifest[
                "locked_policy_sha256"
            ],
            expected_phase5_hashes=phase5_hashes,
        )
        result.artifact_paths["model_card"] = _write_model_card(
            result,
            config,
            config_path=config_path,
        )
        return result

    if paths["evaluation_state"].is_file():
        result = _recover_completed_result(
            paths,
            _read_json(paths["evaluation_state"]),
            expected_locked_policy_sha256=frozen_manifest[
                "locked_policy_sha256"
            ],
            expected_phase5_hashes=phase5_hashes,
        )
        result.artifact_paths["model_card"] = _write_model_card(
            result,
            config,
            config_path=config_path,
        )
        return result

    raw_dataset_path = resolve_project_path(
        config["data"]["raw_path"],
        config_path=config_path,
    )

    if not raw_dataset_path.is_file():
        raise FileNotFoundError(
            f"Final evaluation dataset was not found: {raw_dataset_path}"
        )

    evaluation_started_at = _utc_now()
    state = {
        "phase": _PHASE,
        "status": "test_access_started",
        "test_access_started": True,
        "test_evaluated": False,
        "attempt_count": 1,
        "started_at_utc": evaluation_started_at,
        "locked_policy_sha256": frozen_manifest["locked_policy_sha256"],
        "phase5_artifact_sha256": phase5_hashes,
    }
    _write_json(paths["evaluation_state"], state)
    source_dataset_sha256 = sha256_file(raw_dataset_path)
    dataframe = load_base_dataset(raw_dataset_path, validate=True)
    test_features, test_target, stable_keys = _extract_locked_test_frame(
        dataframe,
        config,
    )
    del dataframe
    outputs = _evaluate_locked_test(
        model,
        test_features,
        test_target,
        stable_keys,
        phase5_metrics,
        phase5_metadata,
        config,
    )
    completed_at = _utc_now()
    metadata = {
        "phase": _PHASE,
        "created_at_utc": completed_at,
        "evaluation_started_at_utc": evaluation_started_at,
        "test_evaluated": True,
        "test_evaluation_count": 1,
        "test_periods": config["final_evaluation"]["evaluation_periods"],
        "test_rows": len(test_target),
        "test_positive_rows": int(test_target.sum()),
        "test_features_returned_by_result": False,
        "test_target_returned_by_result": False,
        "base_model_name": phase5_metadata["base_model_name"],
        "selected_calibrator_name": phase5_metadata[
            "selected_calibrator_name"
        ],
        "base_model_refit_after_test": False,
        "calibrator_refit_after_test": False,
        "threshold_reselected_after_test": False,
        "risk_bands_reselected_after_test": False,
        "model_selected_on": "validation_periods_only",
        "calibrator_selected_on": "validation_periods_only",
        "threshold_selected_on": "validation_periods_only",
        "test_used_for": "one_time_final_reporting_only",
        "fixed_threshold_source": "validation_periods_only",
        "exact_capacity_policy": "locked_5_percent_batch_ranking",
        "automated_rejection_allowed": False,
        "locked_policy_sha256": frozen_manifest["locked_policy_sha256"],
        "phase5_artifact_sha256": phase5_hashes,
        "source_dataset_sha256": source_dataset_sha256,
        "python_packages": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    _save_evaluation_artifacts(outputs, paths, metadata)
    result = Phase6FinalEvaluationResult(
        test_metrics=outputs["test_metrics"],
        fixed_threshold_metrics=outputs["fixed_threshold_metrics"],
        capacity_metrics=outputs["capacity_metrics"],
        bootstrap_intervals=outputs["bootstrap_intervals"],
        temporal_comparison=outputs["temporal_comparison"],
        stability_assessment=outputs["stability_assessment"],
        risk_band_summary=outputs["risk_band_summary"],
        slice_metrics=outputs["slice_metrics"],
        error_analysis=outputs["error_analysis"],
        metadata=metadata,
        artifact_paths={
            name: path for name, path in paths.items() if name != "directory"
        },
        reused_existing_result=False,
    )
    result.artifact_paths["model_card"] = _write_model_card(
        result,
        config,
        config_path=config_path,
    )
    result_hashes = _result_artifact_hashes(paths)
    completion = {
        "phase": _PHASE,
        "status": "completed",
        "completed_at_utc": completed_at,
        "test_evaluated": True,
        "test_evaluation_count": 1,
        "reuse_policy": "reuse_without_test_re_evaluation",
        "locked_policy_sha256": frozen_manifest["locked_policy_sha256"],
        "phase5_artifact_sha256": phase5_hashes,
        "result_artifact_sha256": result_hashes,
    }
    state.update(
        {
            "status": "results_written_awaiting_completion",
            "test_evaluated": True,
            "completed_at_utc": completed_at,
            "result_artifact_sha256": result_hashes,
        }
    )
    _write_json(paths["evaluation_state"], state)
    _write_json(paths["completion"], completion)
    return result


def _print_summary(result: Phase6FinalEvaluationResult) -> None:
    """Print a concise Indonesian final-test summary."""
    calibrated = result.test_metrics["calibrated"]
    print("Evaluasi final Fase 6 tersedia.")
    print("Menggunakan hasil tersimpan:", result.reused_existing_result)
    print("Test rows:", f"{calibrated['row_count']:,}")
    print(
        "Test average precision:",
        f"{calibrated['pr_auc_average_precision']:.6f}",
    )
    print("Test ROC-AUC:", f"{calibrated['roc_auc']:.6f}")
    print("Test Brier score:", f"{calibrated['brier_score']:.6f}")
    print(
        "Recall pada exact capacity 5%:",
        f"{result.capacity_metrics['recall']:.6f}",
    )
    print("Test evaluation count: 1")


def main() -> None:
    """Run or safely reuse the one-time Phase 6 final evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Jalankan sekali atau muat ulang evaluasi final FraudShield."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path konfigurasi YAML proyek.",
    )
    arguments = parser.parse_args()
    result = run_phase6_final_evaluation(arguments.config)
    _print_summary(result)


if __name__ == "__main__":
    main()
