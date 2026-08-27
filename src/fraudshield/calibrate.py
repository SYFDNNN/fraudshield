"""Leakage-safe probability calibration and threshold selection for Phase 5."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from fraudshield.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
)
from fraudshield.data import load_base_dataset
from fraudshield.evaluate import (
    build_error_analysis_frame,
    calibration_table,
    evaluate_probabilities,
)
from fraudshield.model_selection import (
    build_xgboost_models,
    fit_phase4_models,
)
from fraudshield.thresholds import (
    CapacityThresholdPolicy,
    RiskBandPolicy,
    assign_risk_bands,
    build_risk_band_policy,
    capacity_review_flags,
    risk_band_summary,
    select_capacity_threshold,
    threshold_review_flags,
)
from fraudshield.train import (
    BaselineDevelopmentData,
    positive_class_probability,
    prepare_development_data,
)

_SAFE_CANDIDATE_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_UNCALIBRATED_NAME = "uncalibrated"


def _probability_array(probabilities: Any) -> np.ndarray:
    """Return validated one-dimensional probabilities."""
    probability_array = np.asarray(probabilities, dtype=float)

    if probability_array.ndim != 1 or len(probability_array) == 0:
        raise ValueError("Probabilities must be a non-empty 1D array.")

    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must contain only finite values.")

    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilities must lie within [0, 1].")

    return probability_array


def _calibration_inputs(
    probabilities: Any,
    target: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate calibration scores and binary labels."""
    probability_array = _probability_array(probabilities)
    target_array = np.asarray(target)

    if target_array.ndim != 1 or len(target_array) != len(probability_array):
        raise ValueError(
            "Calibration target and probabilities must be equal-length "
            "1D arrays."
        )

    if not np.isin(target_array, [0, 1]).all():
        raise ValueError("Calibration target must contain only zero and one.")

    if len(np.unique(target_array)) != 2:
        raise ValueError("Calibration fitting requires both target classes.")

    return probability_array, target_array.astype(np.int8)


class IdentityProbabilityCalibrator(BaseEstimator):
    """Return raw probabilities while exposing a calibrator interface."""

    def fit(
        self,
        probabilities: Any,
        target: Any,
    ) -> IdentityProbabilityCalibrator:
        """Validate inputs and mark the identity mapping as fitted."""
        _calibration_inputs(probabilities, target)
        self.is_fitted_ = True
        return self

    def transform(self, probabilities: Any) -> np.ndarray:
        """Return an isolated copy of raw probabilities."""
        check_is_fitted(self, "is_fitted_")
        return _probability_array(probabilities).copy()


class SigmoidProbabilityCalibrator(BaseEstimator):
    """Fit a Platt-style sigmoid mapping on raw probability logits."""

    def __init__(
        self,
        *,
        C: float = 1_000_000.0,
        max_iter: int = 1_000,
        clip_epsilon: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.clip_epsilon = clip_epsilon
        self.random_state = random_state

    def _logits(self, probabilities: np.ndarray) -> np.ndarray:
        epsilon = float(self.clip_epsilon)

        if not 0 < epsilon < 0.5:
            raise ValueError("clip_epsilon must lie within (0, 0.5).")

        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(
        self,
        probabilities: Any,
        target: Any,
    ) -> SigmoidProbabilityCalibrator:
        """Fit an unweighted sigmoid using calibration-period labels."""
        probability_array, target_array = _calibration_inputs(
            probabilities,
            target,
        )

        if float(self.C) <= 0:
            raise ValueError("Sigmoid C must be positive.")

        if int(self.max_iter) < 1:
            raise ValueError("Sigmoid max_iter must be positive.")

        self.estimator_ = LogisticRegression(
            C=float(self.C),
            solver="lbfgs",
            max_iter=int(self.max_iter),
            class_weight=None,
            random_state=int(self.random_state),
        )
        self.estimator_.fit(self._logits(probability_array), target_array)
        return self

    def transform(self, probabilities: Any) -> np.ndarray:
        """Apply the fitted sigmoid mapping."""
        check_is_fitted(self, "estimator_")
        probability_array = _probability_array(probabilities)
        return self.estimator_.predict_proba(
            self._logits(probability_array)
        )[:, 1]


class IsotonicProbabilityCalibrator(BaseEstimator):
    """Fit a non-parametric monotonic probability mapping."""

    def __init__(self, *, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds

    def fit(
        self,
        probabilities: Any,
        target: Any,
    ) -> IsotonicProbabilityCalibrator:
        """Fit isotonic regression using calibration-period labels."""
        probability_array, target_array = _calibration_inputs(
            probabilities,
            target,
        )
        self.estimator_ = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds=self.out_of_bounds,
        )
        self.estimator_.fit(probability_array, target_array)
        return self

    def transform(self, probabilities: Any) -> np.ndarray:
        """Apply the fitted isotonic mapping."""
        check_is_fitted(self, "estimator_")
        calibrated = self.estimator_.predict(
            _probability_array(probabilities)
        )
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class CalibrationSelectionDecision:
    """Auditable choice between raw and calibrated probabilities."""

    baseline_name: str
    best_calibrator_name: str
    selected_calibrator_name: str
    calibrator_promoted: bool
    validation_brier_score_baseline: float
    validation_brier_score_challenger: float
    absolute_brier_improvement: float
    minimum_absolute_brier_improvement: float
    validation_average_precision_baseline: float
    validation_average_precision_challenger: float
    average_precision_change: float
    maximum_average_precision_drop: float
    brier_guardrail_passed: bool
    ranking_guardrail_passed: bool
    ranked_calibration_candidates: tuple[str, ...]
    reason: str


@dataclass(slots=True)
class CalibratedFraudModel:
    """Frozen base model plus calibration and manual-review policies."""

    base_model: Pipeline
    calibrator: BaseEstimator
    calibrator_name: str
    capacity_policy: CapacityThresholdPolicy
    risk_band_policy: RiskBandPolicy

    @property
    def classes_(self) -> np.ndarray:
        """Expose binary classes for probability-consumer compatibility."""
        return np.asarray([0, 1], dtype=np.int8)

    def raw_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Return the frozen base model's class-one probability."""
        return positive_class_probability(self.base_model, features)

    def fraud_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Return calibrated fraud probabilities."""
        return self.calibrator.transform(self.raw_probability(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return two-column calibrated class probabilities."""
        positive_probability = self.fraud_probability(features)
        return np.column_stack(
            [1.0 - positive_probability, positive_probability]
        )

    def recommend_review(
        self,
        features: pd.DataFrame,
        *,
        stable_keys: Any = None,
    ) -> np.ndarray:
        """Return an exact capacity-limited review recommendation batch."""
        raw_probability = self.raw_probability(features)
        calibrated_probability = self.calibrator.transform(raw_probability)
        return capacity_review_flags(
            calibrated_probability,
            review_rate=self.capacity_policy.requested_review_rate,
            secondary_scores=raw_probability,
            stable_keys=stable_keys,
        )

    def at_or_above_threshold(self, features: pd.DataFrame) -> np.ndarray:
        """Return tie-inclusive cutoff flags for threshold diagnostics."""
        return threshold_review_flags(
            self.fraud_probability(features),
            self.capacity_policy,
        )

    def risk_band(self, features: pd.DataFrame) -> np.ndarray:
        """Assign stored development risk-band labels."""
        return assign_risk_bands(
            self.fraud_probability(features),
            self.risk_band_policy,
        )


@dataclass(slots=True)
class Phase5ExperimentResult:
    """All development-only outputs from one Phase 5 experiment."""

    config: dict[str, Any]
    development_data: BaselineDevelopmentData
    base_model_name: str
    base_model: Pipeline
    scale_pos_weight: float
    base_training_seconds: float
    calibrators: dict[str, BaseEstimator]
    calibration_seconds: dict[str, float]
    probabilities: dict[str, dict[str, np.ndarray]]
    metrics: dict[str, dict[str, dict[str, object]]]
    comparison: pd.DataFrame
    decision: CalibrationSelectionDecision
    capacity_policy: CapacityThresholdPolicy
    risk_band_policy: RiskBandPolicy
    risk_band_summary: pd.DataFrame
    threshold_metrics: dict[str, object]
    calibrated_model: CalibratedFraudModel
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def build_calibration_candidates(
    config: dict[str, Any],
) -> dict[str, BaseEstimator]:
    """Build the raw identity baseline and configured calibrators."""
    candidates: dict[str, BaseEstimator] = {
        _UNCALIBRATED_NAME: IdentityProbabilityCalibrator()
    }
    configured_candidates = config["calibration"].get("candidates")

    if not isinstance(configured_candidates, list) or not configured_candidates:
        raise ValueError("calibration.candidates must be a non-empty list.")

    for candidate in configured_candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Every calibration candidate must be a mapping.")

        name = candidate.get("name")
        method = candidate.get("method")

        if not isinstance(name, str) or not _SAFE_CANDIDATE_NAME.fullmatch(name):
            raise ValueError(
                "Calibration candidate names must use lowercase letters, "
                "numbers, and underscores."
            )

        if name in candidates:
            raise ValueError(f"Duplicate calibration candidate name: {name}")

        if method == "sigmoid":
            candidates[name] = SigmoidProbabilityCalibrator(
                C=float(candidate["C"]),
                max_iter=int(candidate["max_iter"]),
                clip_epsilon=float(candidate["clip_epsilon"]),
                random_state=int(config["project"]["random_seed"]),
            )
        elif method == "isotonic":
            candidates[name] = IsotonicProbabilityCalibrator(
                out_of_bounds=candidate["out_of_bounds"]
            )
        else:
            raise ValueError(
                f"Unsupported calibration method for {name!r}: {method!r}."
            )

    return candidates


def fit_calibration_candidates(
    calibrators: dict[str, BaseEstimator],
    raw_calibration_probabilities: np.ndarray,
    calibration_target: pd.Series,
) -> dict[str, float]:
    """Fit every mapping on calibration month only."""
    fitting_seconds: dict[str, float] = {}

    for candidate_name, calibrator in calibrators.items():
        started_at = time.perf_counter()
        calibrator.fit(raw_calibration_probabilities, calibration_target)
        fitting_seconds[candidate_name] = float(
            time.perf_counter() - started_at
        )

    return fitting_seconds


def evaluate_calibration_candidates(
    calibrators: dict[str, BaseEstimator],
    *,
    raw_calibration_probabilities: np.ndarray,
    calibration_target: pd.Series,
    raw_validation_probabilities: np.ndarray,
    validation_target: pd.Series,
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[str, object]]],
]:
    """Evaluate mappings on calibration and validation, never test."""
    evaluation_config = config["evaluation"]
    raw_split_probabilities = {
        "calibration": raw_calibration_probabilities,
        "validation": raw_validation_probabilities,
    }
    split_targets = {
        "calibration": calibration_target,
        "validation": validation_target,
    }
    calibrated_probabilities: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict[str, object]]] = {}

    for candidate_name, calibrator in calibrators.items():
        candidate_probabilities: dict[str, np.ndarray] = {}
        candidate_metrics: dict[str, dict[str, object]] = {}

        for split_name in ("calibration", "validation"):
            probabilities = calibrator.transform(
                raw_split_probabilities[split_name]
            )
            candidate_probabilities[split_name] = probabilities
            candidate_metrics[split_name] = evaluate_probabilities(
                split_targets[split_name],
                probabilities,
                review_rates=evaluation_config["review_rates"],
                diagnostic_threshold=evaluation_config[
                    "diagnostic_threshold"
                ],
                calibration_bins=evaluation_config["calibration_bins"],
            )

        calibrated_probabilities[candidate_name] = candidate_probabilities
        metrics[candidate_name] = candidate_metrics

    return calibrated_probabilities, metrics


def build_calibration_comparison_frame(
    metrics: dict[str, dict[str, dict[str, object]]],
    calibration_seconds: dict[str, float],
) -> pd.DataFrame:
    """Build the validation table used by calibration selection."""
    if set(metrics) != set(calibration_seconds):
        raise ValueError(
            "Calibration durations must match evaluated candidates."
        )

    records: list[dict[str, Any]] = []

    for candidate_name, split_metrics in metrics.items():
        if set(split_metrics) != {"calibration", "validation"}:
            raise ValueError(
                "Calibration selection accepts calibration and validation "
                "metrics only."
            )

        calibration_metrics = split_metrics["calibration"]
        validation_metrics = split_metrics["validation"]
        records.append(
            {
                "calibration_method": candidate_name,
                "calibration_brier_score_in_sample": calibration_metrics[
                    "brier_score"
                ],
                "calibration_ece_in_sample": calibration_metrics[
                    "expected_calibration_error"
                ],
                "validation_average_precision": validation_metrics[
                    "pr_auc_average_precision"
                ],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_brier_score": validation_metrics[
                    "brier_score"
                ],
                "validation_log_loss": validation_metrics["log_loss"],
                "validation_ece": validation_metrics[
                    "expected_calibration_error"
                ],
                "calibration_seconds": calibration_seconds[candidate_name],
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        [
            "validation_brier_score",
            "validation_ece",
            "validation_log_loss",
            "validation_average_precision",
            "calibration_method",
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
        ignore_index=True,
    )


def select_calibration_method(
    comparison: pd.DataFrame,
    config: dict[str, Any],
) -> CalibrationSelectionDecision:
    """Select a calibrator using validation-only locked guardrails."""
    calibration_config = config["calibration"]
    selection_config = calibration_config["selection"]

    if tuple(calibration_config["fit_periods"]) != tuple(
        config["split"]["calibration_periods"]
    ):
        raise ValueError(
            "Calibration fit periods must match the calibration split."
        )

    if tuple(calibration_config["selection_periods"]) != tuple(
        config["split"]["validation_periods"]
    ):
        raise ValueError(
            "Calibration selection periods must match validation periods."
        )

    if set(calibration_config["selection_periods"]) & set(
        config["split"]["test_periods"]
    ):
        raise ValueError("Calibration selection must not use test periods.")

    if selection_config["primary_metric"] != "brier_score":
        raise ValueError("Phase 5 requires brier_score as primary metric.")

    expected_tie_breakers = (
        "expected_calibration_error",
        "log_loss",
        "average_precision",
        "method_name",
    )

    if tuple(selection_config["tie_breakers"]) != expected_tie_breakers:
        raise ValueError(
            "Configured calibration tie-breakers do not match the locked "
            "selection order."
        )

    baseline_name = selection_config["baseline_name"]
    baseline_rows = comparison.loc[
        comparison["calibration_method"] == baseline_name
    ]
    challenger_rows = comparison.loc[
        comparison["calibration_method"] != baseline_name
    ].sort_values(
        [
            "validation_brier_score",
            "validation_ece",
            "validation_log_loss",
            "validation_average_precision",
            "calibration_method",
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )

    if len(baseline_rows) != 1 or challenger_rows.empty:
        raise ValueError(
            "Calibration selection requires one raw baseline and a challenger."
        )

    baseline = baseline_rows.iloc[0]
    challenger = challenger_rows.iloc[0]
    brier_improvement = float(
        baseline["validation_brier_score"]
        - challenger["validation_brier_score"]
    )
    average_precision_change = float(
        challenger["validation_average_precision"]
        - baseline["validation_average_precision"]
    )
    minimum_brier_improvement = float(
        selection_config["minimum_absolute_brier_improvement"]
    )
    maximum_ap_drop = float(
        selection_config["maximum_average_precision_drop"]
    )

    if minimum_brier_improvement < 0 or maximum_ap_drop < 0:
        raise ValueError("Calibration guardrails must be non-negative.")

    brier_guardrail_passed = (
        brier_improvement >= minimum_brier_improvement
    )
    ranking_guardrail_passed = average_precision_change >= -maximum_ap_drop
    calibrator_promoted = (
        brier_guardrail_passed and ranking_guardrail_passed
    )
    best_calibrator_name = str(challenger["calibration_method"])
    selected_calibrator_name = (
        best_calibrator_name if calibrator_promoted else baseline_name
    )

    if calibrator_promoted:
        reason = (
            "Metode kalibrasi terbaik memenuhi guardrail Brier score dan "
            "average precision pada validation month 6."
        )
    else:
        failed_guardrails = []

        if not brier_guardrail_passed:
            failed_guardrails.append("peningkatan Brier score")

        if not ranking_guardrail_passed:
            failed_guardrails.append("stabilitas average precision")

        reason = (
            "Probabilitas mentah dipertahankan karena kandidat terbaik tidak "
            "memenuhi: " + ", ".join(failed_guardrails) + "."
        )

    return CalibrationSelectionDecision(
        baseline_name=baseline_name,
        best_calibrator_name=best_calibrator_name,
        selected_calibrator_name=selected_calibrator_name,
        calibrator_promoted=calibrator_promoted,
        validation_brier_score_baseline=float(
            baseline["validation_brier_score"]
        ),
        validation_brier_score_challenger=float(
            challenger["validation_brier_score"]
        ),
        absolute_brier_improvement=brier_improvement,
        minimum_absolute_brier_improvement=minimum_brier_improvement,
        validation_average_precision_baseline=float(
            baseline["validation_average_precision"]
        ),
        validation_average_precision_challenger=float(
            challenger["validation_average_precision"]
        ),
        average_precision_change=average_precision_change,
        maximum_average_precision_drop=maximum_ap_drop,
        brier_guardrail_passed=brier_guardrail_passed,
        ranking_guardrail_passed=ranking_guardrail_passed,
        ranked_calibration_candidates=tuple(
            challenger_rows["calibration_method"]
        ),
        reason=reason,
    )


def _build_phase5_base_model(
    development_data: BaselineDevelopmentData,
    config: dict[str, Any],
) -> tuple[str, Pipeline, float, float]:
    """Rebuild and fit only the Phase 4 selected XGBoost specification."""
    models, scale_pos_weight = build_xgboost_models(
        development_data.train_features,
        development_data.train_target,
        config,
    )
    base_model_name = config["calibration"]["base_model_name"]

    if base_model_name not in models:
        raise KeyError(
            "Configured Phase 5 base model is not an XGBoost candidate: "
            f"{base_model_name}"
        )

    base_model = models[base_model_name]
    training_seconds = fit_phase4_models(
        {base_model_name: base_model},
        development_data.train_features,
        development_data.train_target,
    )[base_model_name]

    return (
        base_model_name,
        base_model,
        scale_pos_weight,
        training_seconds,
    )


def _validate_review_policy(config: dict[str, Any]) -> None:
    """Fail closed when the operating policy contradicts Phase 5."""
    review_policy = config["review_policy"]
    threshold_config = config["threshold_selection"]

    if review_policy.get("automated_rejection_allowed") is not False:
        raise ValueError("Phase 5 does not allow automated rejection.")

    if review_policy.get("threshold_purpose") != (
        "manual_review_prioritization_only"
    ):
        raise ValueError(
            "Phase 5 threshold must be limited to manual review "
            "prioritization."
        )

    if float(review_policy["capacity_rate"]) != float(
        threshold_config["capacity_rate"]
    ):
        raise ValueError(
            "Review-policy capacity must match threshold-selection capacity."
        )


def save_phase5_artifacts(
    result: Phase5ExperimentResult,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Path]:
    """Persist the calibrated development candidate and audit record."""
    artifact_directory = resolve_project_path(
        result.config["artifacts"]["phase5_directory"],
        config_path=config_path,
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}

    model_path = artifact_directory / "calibrated_review_model.joblib"
    joblib.dump(result.calibrated_model, model_path, compress=3)
    artifact_paths["calibrated_model"] = model_path

    metrics_path = artifact_directory / "calibration_metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["metrics"] = metrics_path

    comparison_path = artifact_directory / "calibration_comparison.csv"
    result.comparison.to_csv(comparison_path, index=False)
    artifact_paths["comparison"] = comparison_path

    decision_path = artifact_directory / "calibration_decision.json"
    decision_path.write_text(
        json.dumps(asdict(result.decision), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["decision"] = decision_path

    capacity_path = artifact_directory / "capacity_threshold_policy.json"
    capacity_path.write_text(
        json.dumps(asdict(result.capacity_policy), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["capacity_policy"] = capacity_path

    risk_policy_path = artifact_directory / "risk_band_policy.json"
    risk_policy_path.write_text(
        json.dumps(asdict(result.risk_band_policy), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["risk_band_policy"] = risk_policy_path

    risk_summary_path = artifact_directory / "validation_risk_bands.csv"
    result.risk_band_summary.to_csv(risk_summary_path, index=False)
    artifact_paths["risk_band_summary"] = risk_summary_path

    threshold_metrics_path = artifact_directory / "threshold_metrics.json"
    threshold_metrics_path.write_text(
        json.dumps(result.threshold_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["threshold_metrics"] = threshold_metrics_path

    validation_tables = []

    for candidate_name, split_probabilities in result.probabilities.items():
        candidate_table = calibration_table(
            result.development_data.validation_target,
            split_probabilities["validation"],
            number_of_bins=result.config["evaluation"]["calibration_bins"],
        ).assign(calibration_method=candidate_name)
        validation_tables.append(candidate_table)

    calibration_table_path = (
        artifact_directory / "validation_calibration_table.csv"
    )
    pd.concat(validation_tables, ignore_index=True).to_csv(
        calibration_table_path,
        index=False,
    )
    artifact_paths["calibration_table"] = calibration_table_path

    selected_probabilities = result.probabilities[
        result.decision.selected_calibrator_name
    ]["validation"]
    error_examples = build_error_analysis_frame(
        result.development_data.validation_features,
        result.development_data.validation_target,
        selected_probabilities,
        diagnostic_threshold=result.capacity_policy.score_threshold,
    )
    error_path = artifact_directory / "threshold_validation_errors.csv"
    error_examples.to_csv(error_path, index=False)
    artifact_paths["validation_errors"] = error_path

    metadata = {
        "phase": 5,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "random_seed": int(result.config["project"]["random_seed"]),
        "python_packages": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "base_model_name": result.base_model_name,
        "base_model_train_periods": result.config["split"]["train_periods"],
        "calibrator_fit_periods": result.config["calibration"]["fit_periods"],
        "calibrator_selection_periods": result.config["calibration"][
            "selection_periods"
        ],
        "threshold_selection_periods": result.config[
            "threshold_selection"
        ]["selection_periods"],
        "reserved_test_periods": result.config["split"]["test_periods"],
        "reserved_test_rows": result.development_data.reserved_test_rows,
        "test_features_exposed": False,
        "test_evaluated": False,
        "base_model_retrained_from_locked_specification": True,
        "base_model_refit_with_later_periods": False,
        "preprocessing_fitted_on": "train_periods_only",
        "calibrator_fitted_on": "calibration_periods_only",
        "calibrator_selected_on": "validation_periods_only",
        "threshold_selected_on": "validation_periods_only",
        "selected_calibrator_name": result.decision.selected_calibrator_name,
        "probability_calibrated": (
            result.decision.selected_calibrator_name != _UNCALIBRATED_NAME
        ),
        "business_threshold_selected": True,
        "threshold_purpose": "manual_review_prioritization_only",
        "automated_rejection_allowed": False,
        "scale_pos_weight": result.scale_pos_weight,
        "base_training_seconds": result.base_training_seconds,
        "calibration_seconds": result.calibration_seconds,
    }
    metadata_path = artifact_directory / "phase5_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["metadata"] = metadata_path
    result.artifact_paths = artifact_paths

    return artifact_paths


def run_phase5_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    save_artifacts: bool = True,
) -> Phase5ExperimentResult:
    """Run calibration, threshold selection, and risk-band construction."""
    config = load_config(config_path)
    _validate_review_policy(config)
    raw_dataset_path = resolve_project_path(
        config["data"]["raw_path"],
        config_path=config_path,
    )
    dataframe = load_base_dataset(raw_dataset_path, validate=True)
    development_data = prepare_development_data(dataframe, config)
    del dataframe

    (
        base_model_name,
        base_model,
        scale_pos_weight,
        base_training_seconds,
    ) = _build_phase5_base_model(development_data, config)
    raw_calibration_probabilities = positive_class_probability(
        base_model,
        development_data.calibration_features,
    )
    raw_validation_probabilities = positive_class_probability(
        base_model,
        development_data.validation_features,
    )
    calibrators = build_calibration_candidates(config)
    calibration_seconds = fit_calibration_candidates(
        calibrators,
        raw_calibration_probabilities,
        development_data.calibration_target,
    )
    probabilities, metrics = evaluate_calibration_candidates(
        calibrators,
        raw_calibration_probabilities=raw_calibration_probabilities,
        calibration_target=development_data.calibration_target,
        raw_validation_probabilities=raw_validation_probabilities,
        validation_target=development_data.validation_target,
        config=config,
    )
    comparison = build_calibration_comparison_frame(
        metrics,
        calibration_seconds,
    )
    decision = select_calibration_method(comparison, config)
    comparison = comparison.assign(
        best_calibrator=lambda frame: frame["calibration_method"].eq(
            decision.best_calibrator_name
        ),
        selected=lambda frame: frame["calibration_method"].eq(
            decision.selected_calibrator_name
        ),
    )
    selected_validation_probabilities = probabilities[
        decision.selected_calibrator_name
    ]["validation"]
    threshold_config = config["threshold_selection"]

    if tuple(threshold_config["selection_periods"]) != tuple(
        config["split"]["validation_periods"]
    ):
        raise ValueError(
            "Threshold selection periods must match validation periods."
        )

    if set(threshold_config["selection_periods"]) & set(
        config["split"]["test_periods"]
    ):
        raise ValueError("Threshold selection must not use test periods.")

    capacity_policy = select_capacity_threshold(
        development_data.validation_target,
        selected_validation_probabilities,
        review_rate=float(threshold_config["capacity_rate"]),
        source_split="validation",
        tie_break_policy=threshold_config["tie_break_policy"],
    )
    risk_policy = build_risk_band_policy(
        selected_validation_probabilities,
        threshold_config["risk_bands"],
        source_split="validation",
        tie_break_policy=threshold_config["tie_break_policy"],
    )
    validation_risk_summary = risk_band_summary(
        development_data.validation_target,
        selected_validation_probabilities,
        risk_policy,
    )
    threshold_metrics = evaluate_probabilities(
        development_data.validation_target,
        selected_validation_probabilities,
        review_rates=config["evaluation"]["review_rates"],
        diagnostic_threshold=capacity_policy.score_threshold,
        calibration_bins=config["evaluation"]["calibration_bins"],
    )
    calibrated_model = CalibratedFraudModel(
        base_model=base_model,
        calibrator=calibrators[decision.selected_calibrator_name],
        calibrator_name=decision.selected_calibrator_name,
        capacity_policy=capacity_policy,
        risk_band_policy=risk_policy,
    )
    result = Phase5ExperimentResult(
        config=config,
        development_data=development_data,
        base_model_name=base_model_name,
        base_model=base_model,
        scale_pos_weight=scale_pos_weight,
        base_training_seconds=base_training_seconds,
        calibrators=calibrators,
        calibration_seconds=calibration_seconds,
        probabilities=probabilities,
        metrics=metrics,
        comparison=comparison,
        decision=decision,
        capacity_policy=capacity_policy,
        risk_band_policy=risk_policy,
        risk_band_summary=validation_risk_summary,
        threshold_metrics=threshold_metrics,
        calibrated_model=calibrated_model,
    )

    if save_artifacts:
        save_phase5_artifacts(result, config_path=config_path)

    return result


def _print_phase5_summary(result: Phase5ExperimentResult) -> None:
    """Print a concise Indonesian development summary."""
    print("Eksperimen kalibrasi dan threshold Fase 5 selesai.")
    print(
        "Reserved test rows (tidak dievaluasi): "
        f"{result.development_data.reserved_test_rows:,}"
    )
    print(f"Model dasar: {result.base_model_name}")
    print(
        "Metode kalibrasi terpilih: "
        f"{result.decision.selected_calibrator_name}"
    )
    print(
        "Perbaikan validation Brier score: "
        f"{result.decision.absolute_brier_improvement:+.6f}"
    )
    print(
        "Kapasitas review: "
        f"{result.capacity_policy.requested_review_rate:.2%}"
    )
    print(
        "Threshold probabilitas: "
        f"{result.capacity_policy.score_threshold:.6f}"
    )
    print(result.decision.reason)


def main() -> None:
    """Run the Phase 5 command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Jalankan kalibrasi probabilitas dan seleksi threshold FraudShield."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path konfigurasi YAML proyek.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Jalankan tanpa menulis artifact lokal.",
    )
    arguments = parser.parse_args()
    result = run_phase5_experiment(
        arguments.config,
        save_artifacts=not arguments.no_save,
    )
    _print_phase5_summary(result)


if __name__ == "__main__":
    main()
