"""Tests for leakage-safe XGBoost training and Phase 4 selection."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fraudshield.model_selection as model_selection_module
from fraudshield.model_selection import (
    Phase4ExperimentResult,
    build_model_comparison_frame,
    build_xgboost_models,
    calculate_scale_pos_weight,
    evaluate_phase4_models,
    fit_phase4_models,
    run_phase4_experiment,
    save_phase4_artifacts,
    select_development_model,
    xgboost_importance_frame,
)
from fraudshield.train import (
    BaselineDevelopmentData,
    build_baseline_models,
    positive_class_probability,
)


def build_phase4_config() -> dict[str, object]:
    """Return a compact Phase 4 configuration for unit tests."""
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
                    "name": "xgboost_candidate_a",
                    "n_estimators": 5,
                    "learning_rate": 0.1,
                    "max_depth": 2,
                    "min_child_weight": 1.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 0.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                },
                {
                    "name": "xgboost_candidate_b",
                    "n_estimators": 4,
                    "learning_rate": 0.2,
                    "max_depth": 1,
                    "min_child_weight": 1.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 0.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 2.0,
                },
            ],
        },
        "model_selection": {
            "baseline_model_name": "logistic_regression_balanced",
            "primary_metric": "pr_auc_average_precision",
            "operational_review_rate": 0.5,
            "minimum_absolute_ap_improvement": 0.002,
            "maximum_recall_at_capacity_drop": 0.02,
            "selection_periods": [6],
            "tie_breakers": [
                "recall_at_capacity",
                "brier_score",
                "training_seconds",
                "model_name",
            ],
        },
        "evaluation": {
            "review_rates": [0.5],
            "diagnostic_threshold": 0.5,
            "calibration_bins": 2,
        },
        "artifacts": {
            "phase4_directory": "artifacts/phase4",
        },
    }


def build_development_data() -> BaselineDevelopmentData:
    """Create small development partitions with no exposed test features."""
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
    calibration_features.loc[5, "payment_type"] = "UNSEEN"
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


def build_metrics(
    *,
    baseline_ap: float = 0.20,
    baseline_recall: float = 0.50,
    candidate_a_ap: float = 0.205,
    candidate_a_recall: float = 0.49,
    candidate_b_ap: float = 0.204,
    candidate_b_recall: float = 0.55,
) -> dict[str, dict[str, dict[str, object]]]:
    """Build metrics with enough detail for deterministic selection."""

    def one_model(
        validation_ap: float,
        validation_recall: float,
        *,
        calibration_ap: float,
        brier: float,
    ) -> dict[str, dict[str, object]]:
        return {
            "calibration": {
                "pr_auc_average_precision": calibration_ap,
            },
            "validation": {
                "pr_auc_average_precision": validation_ap,
                "roc_auc": 0.80,
                "brier_score": brier,
                "review_rate_metrics": {
                    "0.5000": {
                        "precision_at_capacity": 0.10,
                        "recall_at_capacity": validation_recall,
                    }
                },
            },
        }

    return {
        "logistic_regression_balanced": one_model(
            baseline_ap,
            baseline_recall,
            calibration_ap=0.99,
            brier=0.10,
        ),
        "xgboost_candidate_a": one_model(
            candidate_a_ap,
            candidate_a_recall,
            calibration_ap=0.01,
            brier=0.08,
        ),
        "xgboost_candidate_b": one_model(
            candidate_b_ap,
            candidate_b_recall,
            calibration_ap=0.98,
            brier=0.07,
        ),
    }


def build_comparison(
    config: dict[str, object],
    metrics: dict[str, dict[str, dict[str, object]]],
) -> pd.DataFrame:
    """Create the model comparison table used in selection tests."""
    return build_model_comparison_frame(
        metrics,
        {
            "logistic_regression_balanced": 1.0,
            "xgboost_candidate_a": 2.0,
            "xgboost_candidate_b": 1.5,
        },
        xgboost_candidate_names=(
            "xgboost_candidate_a",
            "xgboost_candidate_b",
        ),
        config=config,
    )


def test_scale_pos_weight_uses_binary_training_counts() -> None:
    """Class weighting should be the train-only negative/positive ratio."""
    target = pd.Series([0, 0, 0, 0, 0, 0, 1, 1])

    assert calculate_scale_pos_weight(target) == pytest.approx(3.0)

    with pytest.raises(ValueError, match="both target classes"):
        calculate_scale_pos_weight(pd.Series([0, 0, 0]))


def test_xgboost_candidates_fit_inside_preprocessing_pipelines() -> None:
    """Every candidate should fit, score, and tolerate a future category."""
    config = build_phase4_config()
    development_data = build_development_data()
    models, scale_pos_weight = build_xgboost_models(
        development_data.train_features,
        development_data.train_target,
        config,
    )

    durations = fit_phase4_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )
    probabilities = positive_class_probability(
        models["xgboost_candidate_a"],
        development_data.calibration_features,
    )

    assert scale_pos_weight == pytest.approx(1.0)
    assert set(models) == {
        "xgboost_candidate_a",
        "xgboost_candidate_b",
    }
    assert models["xgboost_candidate_a"].named_steps[
        "classifier"
    ].scale_pos_weight == pytest.approx(1.0)
    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert all(duration >= 0 for duration in durations.values())


def test_xgboost_candidate_search_is_explicitly_bounded() -> None:
    """Configuration must prevent accidental expansion into a large search."""
    config = build_phase4_config()
    config["xgboost"]["maximum_candidates"] = 1
    development_data = build_development_data()

    with pytest.raises(ValueError, match="exceeds maximum_candidates"):
        build_xgboost_models(
            development_data.train_features,
            development_data.train_target,
            config,
        )


def test_selection_uses_validation_ap_and_locked_guardrails() -> None:
    """Calibration performance must not override validation selection."""
    config = build_phase4_config()
    comparison = build_comparison(config, build_metrics())

    decision = select_development_model(
        comparison,
        xgboost_candidate_names=(
            "xgboost_candidate_a",
            "xgboost_candidate_b",
        ),
        config=config,
    )

    assert decision.best_xgboost_model_name == "xgboost_candidate_a"
    assert decision.selected_model_name == "xgboost_candidate_a"
    assert decision.xgboost_promoted is True
    assert decision.average_precision_guardrail_passed is True
    assert decision.recall_guardrail_passed is True


def test_selection_retains_baseline_for_materiality_shortfall() -> None:
    """A negligible AP gain should not replace the simpler baseline."""
    config = build_phase4_config()
    metrics = build_metrics(
        candidate_a_ap=0.201,
        candidate_b_ap=0.2005,
    )

    decision = select_development_model(
        build_comparison(config, metrics),
        xgboost_candidate_names=(
            "xgboost_candidate_a",
            "xgboost_candidate_b",
        ),
        config=config,
    )

    assert decision.selected_model_name == "logistic_regression_balanced"
    assert decision.xgboost_promoted is False
    assert decision.average_precision_guardrail_passed is False


def test_selection_rejects_non_validation_periods() -> None:
    """A test-period selection configuration must fail closed."""
    config = build_phase4_config()
    config["model_selection"]["selection_periods"] = [7]

    with pytest.raises(ValueError, match="validation periods"):
        select_development_model(
            build_comparison(config, build_metrics()),
            xgboost_candidate_names=(
                "xgboost_candidate_a",
                "xgboost_candidate_b",
            ),
            config=config,
        )


def test_comparison_rejects_test_metrics() -> None:
    """The selection table must fail if test metrics are introduced."""
    config = build_phase4_config()
    metrics = build_metrics()
    metrics["xgboost_candidate_a"]["test"] = metrics[
        "xgboost_candidate_a"
    ]["validation"]

    with pytest.raises(ValueError, match="calibration and validation"):
        build_comparison(config, metrics)


def test_full_phase4_orchestration_keeps_test_features_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The top-level run should carry only a reserved test row count."""
    config = build_phase4_config()
    records = []

    for month in range(8):
        for target in (0, 1):
            records.append(
                {
                    "month": month,
                    "fraud_bool": target,
                    "income": 0.2 + (0.6 * target),
                    "prev_address_months_count": -1 + (20 * target),
                    "velocity_4w": 100 + (700 * target) + month,
                    "payment_type": "AA" if target == 0 else "AB",
                }
            )

    dataframe = pd.DataFrame.from_records(records)
    monkeypatch.setattr(
        model_selection_module,
        "load_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        model_selection_module,
        "resolve_project_path",
        lambda value, config_path: Path(value),
    )
    monkeypatch.setattr(
        model_selection_module,
        "load_base_dataset",
        lambda path, validate: dataframe,
    )

    result = run_phase4_experiment("configs/base.yaml", save_artifacts=False)

    assert result.development_data.reserved_test_rows == 2
    assert not hasattr(result.development_data, "test_features")
    assert not hasattr(result.development_data, "test_target")
    assert all(
        set(split_metrics) == {"calibration", "validation"}
        for split_metrics in result.metrics.values()
    )
    assert result.decision.selected_model_name in result.models


def test_phase4_artifacts_record_untouched_test_guardrail(
    tmp_path: Path,
) -> None:
    """Saved artifacts must state that test features were never exposed."""
    config = build_phase4_config()
    development_data = build_development_data()
    baseline = build_baseline_models(
        development_data.train_features,
        config,
    )["logistic_regression_balanced"]
    xgboost_models, scale_pos_weight = build_xgboost_models(
        development_data.train_features,
        development_data.train_target,
        config,
    )
    models = {
        "logistic_regression_balanced": baseline,
        **xgboost_models,
    }
    training_seconds = fit_phase4_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )
    metrics = evaluate_phase4_models(models, development_data, config)
    comparison = build_model_comparison_frame(
        metrics,
        training_seconds,
        xgboost_candidate_names=tuple(xgboost_models),
        config=config,
    )
    decision = select_development_model(
        comparison,
        xgboost_candidate_names=tuple(xgboost_models),
        config=config,
    )
    decision = replace(
        decision,
        selected_model_name="xgboost_candidate_a",
        best_xgboost_model_name="xgboost_candidate_a",
    )
    result = Phase4ExperimentResult(
        config=config,
        development_data=development_data,
        models=models,
        xgboost_candidate_names=tuple(xgboost_models),
        scale_pos_weight=scale_pos_weight,
        training_seconds=training_seconds,
        metrics=metrics,
        comparison=comparison,
        decision=decision,
    )
    config_path = tmp_path / "configs" / "base.yaml"

    artifact_paths = save_phase4_artifacts(
        result,
        config_path=config_path,
    )
    metadata = json.loads(
        artifact_paths["metadata"].read_text(encoding="utf-8")
    )
    importance = xgboost_importance_frame(
        models["xgboost_candidate_a"]
    )

    assert artifact_paths["selected_model"].is_file()
    assert artifact_paths["decision"].is_file()
    assert metadata["test_features_exposed"] is False
    assert metadata["test_evaluated"] is False
    assert metadata["scale_pos_weight_source"] == "training_target_only"
    assert metadata["model_selected_on"] == "validation_periods_only"
    assert not importance.empty
