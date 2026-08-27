"""Tests for Phase 5 probability calibration and orchestration."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import fraudshield.calibrate as calibrate_module
from fraudshield.calibrate import (
    CalibratedFraudModel,
    IdentityProbabilityCalibrator,
    IsotonicProbabilityCalibrator,
    SigmoidProbabilityCalibrator,
    build_calibration_candidates,
    build_calibration_comparison_frame,
    run_phase5_experiment,
    select_calibration_method,
)
from fraudshield.evaluate import evaluate_probabilities
from fraudshield.thresholds import (
    build_risk_band_policy,
    select_capacity_threshold,
)
from fraudshield.train import positive_class_probability


def build_phase5_config() -> dict[str, object]:
    """Return a compact configuration for Phase 5 tests."""
    return {
        "project": {"random_seed": 42},
        "data": {
            "raw_path": "data/raw/Base.csv",
            "target_column": "fraud_bool",
            "time_column": "month",
        },
        "split": {
            "train_periods": [0, 1, 2, 3, 4],
            "calibration_periods": [5],
            "validation_periods": [6],
            "test_periods": [7],
        },
        "feature_policy": {
            "high_drift_features": ["velocity_4w"],
            "excluded_from_model": ["fraud_bool", "month"],
        },
        "preprocessing": {
            "categorical_features": ["payment_type"],
            "semantic_missing_rules": {
                "prev_address_months_count": {
                    "operator": "equals",
                    "value": -1,
                }
            },
            "add_missing_indicators": True,
            "numeric_imputer_strategy": "median",
            "categorical_imputer_strategy": "most_frequent",
            "scale_numeric": True,
        },
        "baseline": {
            "primary_model_name": "logistic_regression_balanced",
            "dummy": {"strategy": "prior"},
            "logistic_regression": {
                "C": 1.0,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 100,
                "tolerance": 0.0001,
            },
            "high_drift_ablation": {
                "enabled": False,
                "model_name": "logistic_without_drift",
            },
        },
        "xgboost": {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "importance_type": "gain",
            "n_jobs": 1,
            "verbosity": 0,
            "scale_numeric": False,
            "maximum_candidates": 5,
            "candidates": [
                {
                    "name": "xgboost_selected",
                    "n_estimators": 8,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 1.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 0.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                }
            ],
        },
        "model_selection": {
            "baseline_model_name": "logistic_regression_balanced",
            "operational_review_rate": 0.5,
        },
        "calibration": {
            "base_model_name": "xgboost_selected",
            "fit_periods": [5],
            "selection_periods": [6],
            "candidates": [
                {
                    "name": "sigmoid",
                    "method": "sigmoid",
                    "C": 1_000_000.0,
                    "max_iter": 200,
                    "clip_epsilon": 1e-6,
                },
                {
                    "name": "isotonic",
                    "method": "isotonic",
                    "out_of_bounds": "clip",
                },
            ],
            "selection": {
                "baseline_name": "uncalibrated",
                "primary_metric": "brier_score",
                "minimum_absolute_brier_improvement": 0.001,
                "maximum_average_precision_drop": 0.05,
                "tie_breakers": [
                    "expected_calibration_error",
                    "log_loss",
                    "average_precision",
                    "method_name",
                ],
            },
        },
        "threshold_selection": {
            "selection_periods": [6],
            "capacity_rate": 0.5,
            "tie_break_policy": (
                "calibrated_score_desc_then_raw_score_desc_then_stable_key"
            ),
            "risk_bands": [
                {
                    "label": "sangat_tinggi",
                    "cumulative_review_rate": 0.25,
                },
                {
                    "label": "tinggi",
                    "cumulative_review_rate": 0.5,
                },
                {
                    "label": "rendah",
                    "cumulative_review_rate": 1.0,
                },
            ],
        },
        "evaluation": {
            "review_rates": [0.25, 0.5],
            "diagnostic_threshold": 0.5,
            "calibration_bins": 2,
        },
        "review_policy": {
            "capacity_rate": 0.5,
            "decision_threshold": "derived_from_validation",
            "threshold_purpose": "manual_review_prioritization_only",
            "automated_rejection_allowed": False,
        },
        "artifacts": {
            "phase5_directory": "artifacts/phase5",
        },
    }


def build_synthetic_dataset() -> pd.DataFrame:
    """Create chronological rows with both classes in every period."""
    records = []

    for month in range(8):
        for row, target in enumerate((0, 0, 0, 1)):
            records.append(
                {
                    "month": month,
                    "fraud_bool": target,
                    "income": 0.2 + (0.6 * target) + (row * 0.01),
                    "prev_address_months_count": (
                        -1 if row == 0 else 5 + (15 * target)
                    ),
                    "velocity_4w": 100 + (600 * target) + month + row,
                    "payment_type": "AA" if target == 0 else "AB",
                }
            )

    return pd.DataFrame.from_records(records)


def build_selection_comparison(
    *,
    raw_brier: float = 0.12,
    sigmoid_brier: float = 0.02,
    isotonic_brier: float = 0.03,
    raw_ap: float = 0.18,
    sigmoid_ap: float = 0.18,
    isotonic_ap: float = 0.17,
) -> pd.DataFrame:
    """Return a deterministic comparison frame for selection tests."""
    return pd.DataFrame(
        [
            {
                "calibration_method": "uncalibrated",
                "validation_brier_score": raw_brier,
                "validation_ece": 0.20,
                "validation_log_loss": 0.40,
                "validation_average_precision": raw_ap,
            },
            {
                "calibration_method": "sigmoid",
                "validation_brier_score": sigmoid_brier,
                "validation_ece": 0.01,
                "validation_log_loss": 0.08,
                "validation_average_precision": sigmoid_ap,
            },
            {
                "calibration_method": "isotonic",
                "validation_brier_score": isotonic_brier,
                "validation_ece": 0.02,
                "validation_log_loss": 0.09,
                "validation_average_precision": isotonic_ap,
            },
        ]
    )


def test_identity_calibrator_preserves_probabilities() -> None:
    """The uncalibrated baseline should be an exact identity mapping."""
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    calibrator = IdentityProbabilityCalibrator().fit(
        probabilities,
        [0, 0, 1, 1],
    )

    transformed = calibrator.transform(probabilities)

    assert np.array_equal(transformed, probabilities)
    assert transformed is not probabilities


def test_sigmoid_calibrator_returns_ordered_finite_probabilities() -> None:
    """Platt-style calibration should remain monotonic and bounded."""
    calibrator = SigmoidProbabilityCalibrator().fit(
        [0.05, 0.10, 0.20, 0.70, 0.80, 0.95],
        [0, 0, 0, 1, 1, 1],
    )

    transformed = calibrator.transform([0.01, 0.30, 0.60, 0.99])

    assert np.isfinite(transformed).all()
    assert ((transformed >= 0) & (transformed <= 1)).all()
    assert np.all(np.diff(transformed) >= 0)


def test_isotonic_calibrator_clips_out_of_sample_scores() -> None:
    """Isotonic inference should support scores outside the fitted range."""
    calibrator = IsotonicProbabilityCalibrator().fit(
        [0.2, 0.3, 0.7, 0.8],
        [0, 0, 1, 1],
    )

    transformed = calibrator.transform([0.0, 0.5, 1.0])

    assert transformed.shape == (3,)
    assert ((transformed >= 0) & (transformed <= 1)).all()
    assert np.all(np.diff(transformed) >= 0)


def test_configured_calibration_candidates_are_explicit() -> None:
    """Phase 5 should include raw, sigmoid, and isotonic mappings only."""
    candidates = build_calibration_candidates(build_phase5_config())

    assert set(candidates) == {"uncalibrated", "sigmoid", "isotonic"}
    assert isinstance(candidates["sigmoid"], SigmoidProbabilityCalibrator)
    assert isinstance(candidates["isotonic"], IsotonicProbabilityCalibrator)


def test_calibration_selection_promotes_guardrail_compliant_method() -> None:
    """The lowest-Brier challenger should win when ranking is preserved."""
    decision = select_calibration_method(
        build_selection_comparison(),
        build_phase5_config(),
    )

    assert decision.best_calibrator_name == "sigmoid"
    assert decision.selected_calibrator_name == "sigmoid"
    assert decision.calibrator_promoted is True
    assert decision.absolute_brier_improvement == pytest.approx(0.10)
    assert decision.brier_guardrail_passed is True
    assert decision.ranking_guardrail_passed is True


def test_calibration_selection_retains_raw_when_ranking_drops() -> None:
    """Better Brier score cannot compensate for excessive AP degradation."""
    comparison = build_selection_comparison(
        sigmoid_ap=0.10,
        isotonic_ap=0.11,
    )

    decision = select_calibration_method(
        comparison,
        build_phase5_config(),
    )

    assert decision.selected_calibrator_name == "uncalibrated"
    assert decision.calibrator_promoted is False
    assert decision.ranking_guardrail_passed is False


def test_calibration_selection_rejects_test_period() -> None:
    """A test-period calibration decision must fail closed."""
    config = build_phase5_config()
    config["calibration"]["selection_periods"] = [7]

    with pytest.raises(ValueError, match="validation periods"):
        select_calibration_method(
            build_selection_comparison(),
            config,
        )


def test_calibration_comparison_rejects_test_metrics() -> None:
    """The comparison builder must reject any introduced test result."""
    evaluated = evaluate_probabilities(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        review_rates=[0.5],
        calibration_bins=2,
    )
    metrics = {
        "uncalibrated": {
            "calibration": evaluated,
            "validation": evaluated,
            "test": evaluated,
        }
    }

    with pytest.raises(ValueError, match="calibration and validation"):
        build_calibration_comparison_frame(
            metrics,
            {"uncalibrated": 0.0},
        )


def test_phase5_policy_rejects_automated_rejection() -> None:
    """A review score must never silently become an automated decision."""
    config = build_phase5_config()
    config["review_policy"]["automated_rejection_allowed"] = True

    with pytest.raises(ValueError, match="automated rejection"):
        calibrate_module._validate_review_policy(config)


def test_phase5_policy_requires_one_capacity_rate() -> None:
    """The operating and threshold policies must describe one capacity."""
    config = build_phase5_config()
    config["review_policy"]["capacity_rate"] = 0.25

    with pytest.raises(ValueError, match="capacity must match"):
        calibrate_module._validate_review_policy(config)


def test_calibrated_model_exposes_probability_review_and_band_outputs() -> None:
    """The saved wrapper should provide the three required score products."""
    features = pd.DataFrame(
        {"feature": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]}
    )
    target = pd.Series([0, 0, 1, 1, 0, 1])
    base_model = Pipeline(
        [("classifier", LogisticRegression(random_state=42))]
    ).fit(features, target)
    raw_probability = positive_class_probability(base_model, features)
    calibrator = IdentityProbabilityCalibrator().fit(
        raw_probability,
        target,
    )
    capacity_policy = select_capacity_threshold(
        target,
        raw_probability,
        review_rate=0.5,
    )
    risk_policy = build_risk_band_policy(
        raw_probability,
        [
            {"label": "tinggi", "cumulative_review_rate": 0.5},
            {"label": "rendah", "cumulative_review_rate": 1.0},
        ],
    )
    model = CalibratedFraudModel(
        base_model=base_model,
        calibrator=calibrator,
        calibrator_name="uncalibrated",
        capacity_policy=capacity_policy,
        risk_band_policy=risk_policy,
    )

    probabilities = model.predict_proba(features)
    review_flags = model.recommend_review(features)
    risk_bands = model.risk_band(features)

    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert review_flags.sum() == 3
    assert set(review_flags).issubset({0, 1})
    assert set(risk_bands) == {"tinggi", "rendah"}


def test_phase5_run_and_artifacts_keep_test_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end metadata must preserve every untouched-test guardrail."""
    config = build_phase5_config()
    config["artifacts"]["phase5_directory"] = str(tmp_path / "phase5")
    dataframe = build_synthetic_dataset()
    monkeypatch.setattr(
        calibrate_module,
        "load_config",
        lambda path: deepcopy(config),
    )
    monkeypatch.setattr(
        calibrate_module,
        "resolve_project_path",
        lambda value, config_path: Path(value),
    )
    monkeypatch.setattr(
        calibrate_module,
        "load_base_dataset",
        lambda path, validate: dataframe,
    )

    result = run_phase5_experiment(
        "configs/base.yaml",
        save_artifacts=True,
    )
    metadata = json.loads(
        result.artifact_paths["metadata"].read_text(encoding="utf-8")
    )

    assert result.development_data.reserved_test_rows == 4
    assert not hasattr(result.development_data, "test_features")
    assert not hasattr(result.development_data, "test_target")
    assert result.artifact_paths["calibrated_model"].is_file()
    assert metadata["test_features_exposed"] is False
    assert metadata["test_evaluated"] is False
    assert metadata["calibrator_fitted_on"] == "calibration_periods_only"
    assert metadata["threshold_selected_on"] == "validation_periods_only"
    assert metadata["base_model_retrained_from_locked_specification"] is True
    assert metadata["base_model_refit_with_later_periods"] is False
    assert metadata["business_threshold_selected"] is True
    assert metadata["automated_rejection_allowed"] is False
