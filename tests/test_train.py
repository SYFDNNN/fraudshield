"""Tests for Phase 3 baseline-model construction and fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.train import (
    BaselineDevelopmentData,
    BaselineExperimentResult,
    build_baseline_models,
    evaluate_baseline_models,
    fit_baseline_models,
    logistic_coefficient_frame,
    positive_class_probability,
    save_baseline_artifacts,
)


def build_phase3_config() -> dict[str, object]:
    """Return the minimal configuration required by baseline utilities."""
    return {
        "project": {"random_seed": 42},
        "data": {"target_column": "fraud_bool"},
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
                "max_iter": 500,
                "tolerance": 0.0001,
            },
            "high_drift_ablation": {
                "enabled": True,
                "model_name": "logistic_regression_without_high_drift",
            },
        },
        "evaluation": {
            "review_rates": [0.5],
            "diagnostic_threshold": 0.5,
            "calibration_bins": 2,
        },
        "artifacts": {
            "phase3_directory": "artifacts/phase3",
        },
    }


def build_development_data() -> BaselineDevelopmentData:
    """Create small chronological-like partitions for model tests."""
    train_features = pd.DataFrame(
        {
            "income": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.3, 0.7],
            "prev_address_months_count": [-1, 2, 20, 30, 5, -1, 8, 24],
            "velocity_4w": [100, 120, 800, 900, 150, 850, 250, 700],
            "payment_type": ["AA", "AA", "AB", "AB"] * 2,
        }
    )
    train_target = pd.Series([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    calibration_features = train_features.iloc[[0, 2, 4, 5]].copy()
    calibration_features.loc[5, "payment_type"] = "AC"
    calibration_target = pd.Series(
        [0, 1, 0, 1],
        index=calibration_features.index,
        dtype=np.int8,
    )
    validation_features = train_features.iloc[[1, 3, 6, 7]].copy()
    validation_target = pd.Series(
        [0, 1, 0, 1],
        index=validation_features.index,
        dtype=np.int8,
    )

    return BaselineDevelopmentData(
        train_features=train_features,
        train_target=train_target,
        calibration_features=calibration_features,
        calibration_target=calibration_target,
        validation_features=validation_features,
        validation_target=validation_target,
        reserved_test_rows=4,
    )


def test_baseline_models_fit_and_score_without_test_data() -> None:
    """All configured baselines should fit and evaluate on development data."""
    config = build_phase3_config()
    development_data = build_development_data()
    models = build_baseline_models(
        development_data.train_features,
        config,
    )

    training_seconds = fit_baseline_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )
    metrics = evaluate_baseline_models(models, development_data, config)

    assert set(models) == {
        "dummy_prior",
        "logistic_regression_balanced",
        "logistic_regression_without_high_drift",
    }
    assert all(duration >= 0 for duration in training_seconds.values())
    assert set(metrics["logistic_regression_balanced"]) == {
        "calibration",
        "validation",
    }
    assert "pr_auc_average_precision" in metrics[
        "logistic_regression_balanced"
    ]["validation"]


def test_unknown_category_is_supported_by_fitted_logistic_pipeline() -> None:
    """The persisted primary model should score an unseen later category."""
    config = build_phase3_config()
    development_data = build_development_data()
    models = build_baseline_models(
        development_data.train_features,
        config,
    )
    fit_baseline_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )

    probabilities = positive_class_probability(
        models["logistic_regression_balanced"],
        development_data.calibration_features,
    )

    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_ablation_pipeline_excludes_configured_drift_feature() -> None:
    """High-drift ablation should be explicit and inspectable."""
    config = build_phase3_config()
    development_data = build_development_data()
    models = build_baseline_models(
        development_data.train_features,
        config,
    )
    fit_baseline_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )
    ablation_model = models["logistic_regression_without_high_drift"]
    selected_features = ablation_model.named_steps[
        "feature_selection"
    ].selected_columns_
    coefficient_frame = logistic_coefficient_frame(ablation_model)

    assert "velocity_4w" not in selected_features
    assert not coefficient_frame.empty
    assert {
        "feature",
        "coefficient",
        "absolute_coefficient",
    }.issubset(coefficient_frame.columns)


def test_phase3_artifacts_are_serialized_without_test_predictions(
    tmp_path: Path,
) -> None:
    """Saved metadata should preserve the untouched-test guardrail."""
    config = build_phase3_config()
    development_data = build_development_data()
    models = build_baseline_models(
        development_data.train_features,
        config,
    )
    training_seconds = fit_baseline_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )
    metrics = evaluate_baseline_models(models, development_data, config)
    result = BaselineExperimentResult(
        config=config,
        development_data=development_data,
        models=models,
        training_seconds=training_seconds,
        metrics=metrics,
    )
    config_path = tmp_path / "configs" / "base.yaml"

    artifact_paths = save_baseline_artifacts(
        result,
        config_path=config_path,
    )
    metadata = json.loads(
        artifact_paths["metadata"].read_text(encoding="utf-8")
    )

    assert artifact_paths["metrics"].is_file()
    assert artifact_paths["model:dummy_prior"].is_file()
    assert metadata["test_evaluated"] is False
    assert metadata["reserved_test_rows"] == 4
