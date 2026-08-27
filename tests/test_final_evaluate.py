"""Tests for the one-time Phase 6 final evaluation workflow."""

import inspect
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import fraudshield.final_evaluate as final_module
from fraudshield.calibrate import (
    CalibratedFraudModel,
    IdentityProbabilityCalibrator,
)
from fraudshield.evaluate import evaluate_probabilities
from fraudshield.features import DataFrameColumnSelector
from fraudshield.final_evaluate import (
    binary_review_metrics,
    bootstrap_metric_intervals,
    build_capacity_error_analysis,
    build_freeze_manifest,
    build_slice_metrics,
    build_stability_assessment,
    render_model_card,
    run_phase6_final_evaluation,
    validate_frozen_phase5_artifacts,
)
from fraudshield.thresholds import (
    build_risk_band_policy,
    select_capacity_threshold,
)
from fraudshield.train import positive_class_probability


def build_phase6_config(tmp_path: Path) -> dict[str, object]:
    """Return a compact Phase 6 configuration using absolute temp paths."""
    phase5_directory = tmp_path / "phase5"
    phase6_directory = tmp_path / "phase6"
    return {
        "project": {"random_seed": 42},
        "data": {
            "raw_path": str(tmp_path / "Base.csv"),
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
            "excluded_from_model": ["fraud_bool", "month"],
            "high_drift_features": [],
        },
        "xgboost": {
            "candidates": [{"name": "xgboost_selected"}],
        },
        "calibration": {
            "base_model_name": "xgboost_selected",
            "fit_periods": [5],
            "selection_periods": [6],
        },
        "threshold_selection": {
            "selection_periods": [6],
            "capacity_rate": 0.25,
            "risk_bands": [
                {
                    "label": "tinggi",
                    "cumulative_review_rate": 0.25,
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
            "calibration_bins": 4,
        },
        "review_policy": {
            "capacity_rate": 0.25,
            "threshold_purpose": "manual_review_prioritization_only",
            "automated_rejection_allowed": False,
        },
        "final_evaluation": {
            "evaluation_periods": [7],
            "phase5_model_artifact": str(
                phase5_directory / "calibrated_review_model.joblib"
            ),
            "phase5_metadata_artifact": str(
                phase5_directory / "phase5_metadata.json"
            ),
            "phase5_metrics_artifact": str(
                phase5_directory / "calibration_metrics.json"
            ),
            "phase5_decision_artifact": str(
                phase5_directory / "calibration_decision.json"
            ),
            "phase5_capacity_artifact": str(
                phase5_directory / "capacity_threshold_policy.json"
            ),
            "phase5_risk_band_artifact": str(
                phase5_directory / "risk_band_policy.json"
            ),
            "completed_result_policy": (
                "reuse_without_test_re_evaluation"
            ),
            "bootstrap": {
                "method": "stratified_percentile",
                "confidence_level": 0.95,
                "number_of_resamples": 20,
                "random_seed": 42,
            },
            "stability_alerts": {
                "maximum_average_precision_drop": 0.03,
                "maximum_recall_at_capacity_drop": 0.05,
                "maximum_brier_score_increase": 0.01,
                "maximum_expected_calibration_error_increase": 0.02,
            },
            "slices": {
                "minimum_rows": 4,
                "minimum_positive_rows": 1,
                "categorical_features": [],
                "numeric_bins": {
                    "customer_age": [0, 40, 200],
                    "income": [0.0, 0.5, 1.1],
                },
            },
        },
        "artifacts": {
            "phase6_directory": str(phase6_directory),
        },
        "reports": {
            "model_card_path": str(tmp_path / "model_card.md"),
        },
    }


def build_synthetic_dataset() -> pd.DataFrame:
    """Create chronological numeric rows with both classes in each month."""
    records = []

    for month in range(8):
        for row in range(20):
            target = int(row in {2, 7, 15, 18})
            records.append(
                {
                    "month": month,
                    "fraud_bool": target,
                    "feature": float(target * 2.0 + row / 20 + month / 100),
                    "customer_age": 20 + (row * 3),
                    "income": 0.2 + (0.6 * target),
                }
            )

    return pd.DataFrame.from_records(records)


def write_phase5_artifacts(
    config: dict[str, object],
    dataframe: pd.DataFrame,
) -> None:
    """Create a compact frozen Phase 5 artifact set for orchestration tests."""
    dataframe.to_csv(config["data"]["raw_path"], index=False)
    final_config = config["final_evaluation"]
    model_path = Path(final_config["phase5_model_artifact"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_frame = dataframe.loc[dataframe["month"].le(4)]
    validation_frame = dataframe.loc[dataframe["month"].eq(6)]
    feature_columns = ["feature", "customer_age", "income"]
    train_features = train_frame.loc[:, feature_columns]
    train_target = train_frame["fraud_bool"]
    validation_features = validation_frame.loc[:, feature_columns]
    validation_target = validation_frame["fraud_bool"]
    base_model = Pipeline(
        [
            (
                "feature_selection",
                DataFrameColumnSelector(feature_columns),
            ),
            (
                "classifier",
                LogisticRegression(random_state=42),
            ),
        ]
    ).fit(train_features, train_target)
    raw_validation = positive_class_probability(
        base_model,
        validation_features,
    )
    calibrator = IdentityProbabilityCalibrator().fit(
        raw_validation,
        validation_target,
    )
    capacity_policy = select_capacity_threshold(
        validation_target,
        raw_validation,
        review_rate=0.25,
    )
    risk_policy = build_risk_band_policy(
        raw_validation,
        config["threshold_selection"]["risk_bands"],
    )
    model = CalibratedFraudModel(
        base_model=base_model,
        calibrator=calibrator,
        calibrator_name="uncalibrated",
        capacity_policy=capacity_policy,
        risk_band_policy=risk_policy,
    )
    joblib.dump(model, model_path)
    metadata = {
        "test_evaluated": False,
        "test_features_exposed": False,
        "base_model_refit_with_later_periods": False,
        "preprocessing_fitted_on": "train_periods_only",
        "calibrator_fitted_on": "calibration_periods_only",
        "calibrator_selected_on": "validation_periods_only",
        "threshold_selected_on": "validation_periods_only",
        "business_threshold_selected": True,
        "automated_rejection_allowed": False,
        "base_model_name": "xgboost_selected",
        "selected_calibrator_name": "uncalibrated",
    }
    validation_metrics = evaluate_probabilities(
        validation_target,
        raw_validation,
        review_rates=config["evaluation"]["review_rates"],
        diagnostic_threshold=capacity_policy.score_threshold,
        calibration_bins=config["evaluation"]["calibration_bins"],
    )
    metrics = {
        "uncalibrated": {
            "calibration": validation_metrics,
            "validation": validation_metrics,
        }
    }
    payloads = {
        "phase5_metadata_artifact": metadata,
        "phase5_metrics_artifact": metrics,
        "phase5_decision_artifact": {
            "selected_calibrator_name": "uncalibrated"
        },
        "phase5_capacity_artifact": asdict(capacity_policy),
        "phase5_risk_band_artifact": asdict(risk_policy),
    }

    for config_key, payload in payloads.items():
        Path(final_config[config_key]).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def test_binary_review_metrics_counts_policy_outcomes() -> None:
    """Frozen review flags should produce transparent confusion counts."""
    metrics = binary_review_metrics(
        [1, 0, 1, 0],
        [1, 1, 0, 0],
    )

    assert metrics["review_count"] == 2
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_phase6_module_never_fits_a_model() -> None:
    """Final evaluation code must contain no estimator fit call."""
    module_source = inspect.getsource(final_module)

    assert ".fit(" not in module_source


def test_final_configuration_rejects_non_test_period(
    tmp_path: Path,
) -> None:
    """Only the pre-reserved test period may enter final evaluation."""
    config = build_phase6_config(tmp_path)
    config["final_evaluation"]["evaluation_periods"] = [6]

    with pytest.raises(ValueError, match="reserved test periods"):
        final_module._validate_final_configuration(config)


def test_bootstrap_intervals_are_deterministic_and_complete() -> None:
    """The locked random seed should reproduce every reported interval."""
    target = np.asarray([0] * 30 + [1] * 10)
    probabilities = np.linspace(0.01, 0.99, 40)
    first = bootstrap_metric_intervals(
        target,
        probabilities,
        review_rate=0.25,
        number_of_resamples=20,
        confidence_level=0.95,
        random_seed=42,
    )
    second = bootstrap_metric_intervals(
        target,
        probabilities,
        review_rate=0.25,
        number_of_resamples=20,
        confidence_level=0.95,
        random_seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {
        "average_precision",
        "roc_auc",
        "brier_score",
        "precision_at_capacity",
        "recall_at_capacity",
    }
    assert (first["ci_lower"] <= first["ci_upper"]).all()


def test_slice_metrics_preserve_rows_with_locked_bins(tmp_path: Path) -> None:
    """Pre-registered bins should account for every test application."""
    config = build_phase6_config(tmp_path)
    features = pd.DataFrame(
        {
            "customer_age": [20, 30, 50, 70, 25, 60, 35, 80],
            "income": [0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8],
        }
    )
    target = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = np.asarray([0.1, 0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.6])
    review_flags = np.asarray([0, 1, 0, 0, 0, 1, 0, 0])

    slices = build_slice_metrics(
        features,
        target,
        probabilities,
        review_flags,
        config,
    )

    for feature_name in ("customer_age", "income"):
        feature_slices = slices.loc[
            slices["slice_feature"].eq(feature_name)
        ]
        assert feature_slices["row_count"].sum() == len(features)


def test_slice_metrics_reject_uncovered_numeric_values(tmp_path: Path) -> None:
    """Out-of-range rows must not silently disappear from slice reporting."""
    config = build_phase6_config(tmp_path)
    config["final_evaluation"]["slices"]["numeric_bins"]["income"] = [
        0.0,
        0.5,
    ]
    features = pd.DataFrame(
        {"customer_age": [20, 50], "income": [0.2, 0.8]}
    )

    with pytest.raises(ValueError, match="do not cover income"):
        build_slice_metrics(
            features,
            [0, 1],
            [0.1, 0.9],
            [0, 1],
            config,
        )


def test_stability_assessment_reports_alert_without_reselection(
    tmp_path: Path,
) -> None:
    """A test drop should trigger investigation, never model selection."""
    config = build_phase6_config(tmp_path)
    comparison = pd.DataFrame(
        [
            {
                "split": "validation",
                "average_precision": 0.20,
                "recall_at_capacity": 0.50,
                "brier_score": 0.01,
                "expected_calibration_error": 0.01,
            },
            {
                "split": "test",
                "average_precision": 0.10,
                "recall_at_capacity": 0.40,
                "brier_score": 0.03,
                "expected_calibration_error": 0.04,
            },
        ]
    )

    assessment = build_stability_assessment(comparison, config)

    assert assessment["alert_triggered"].all()
    assert set(assessment["action"]) == {"report_and_investigate_only"}


def test_capacity_error_analysis_labels_three_diagnostic_types() -> None:
    """Final error samples should represent queue cost and missed fraud."""
    features = pd.DataFrame({"feature": np.arange(8)})
    target = np.asarray([0, 0, 1, 1, 0, 1, 0, 1])
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.1, 0.6, 0.2, 0.5, 0.3])
    review_flags = np.asarray([1, 1, 1, 0, 0, 0, 0, 0])

    errors = build_capacity_error_analysis(
        features,
        target,
        probabilities,
        review_flags,
        examples_per_type=2,
    )

    assert set(errors["error_type"]) == {
        "reviewed_non_fraud",
        "missed_fraud_near_boundary",
        "missed_fraud_low_score",
    }


def test_freeze_manifest_records_no_post_test_changes(tmp_path: Path) -> None:
    """The pre-test manifest should explicitly freeze every decision layer."""
    config = build_phase6_config(tmp_path)
    manifest = build_freeze_manifest(
        config,
        phase5_metadata={
            "base_model_name": "xgboost_selected",
            "selected_calibrator_name": "uncalibrated",
        },
        phase5_artifact_hashes={"model": "abc"},
    )

    assert manifest["test_access_started"] is False
    assert manifest["test_evaluated"] is False
    assert manifest["no_refit_after_test"] is True
    assert manifest["no_recalibration_after_test"] is True
    assert manifest["no_threshold_reselection_after_test"] is True
    assert len(manifest["locked_policy_sha256"]) == 64


def test_phase5_artifact_validation_rejects_prior_test_access(
    tmp_path: Path,
) -> None:
    """Phase 6 must fail closed if Phase 5 already reports test evaluation."""
    config = build_phase6_config(tmp_path)
    dataframe = build_synthetic_dataset()
    write_phase5_artifacts(config, dataframe)
    metadata_path = Path(
        config["final_evaluation"]["phase5_metadata_artifact"]
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_evaluated"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="test_evaluated"):
        validate_frozen_phase5_artifacts(config)


def test_phase6_evaluates_once_then_reuses_stored_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second run must not load the dataset or evaluate test again."""
    config = build_phase6_config(tmp_path)
    dataframe = build_synthetic_dataset()
    write_phase5_artifacts(config, dataframe)
    load_calls = 0

    def fake_load_base_dataset(path: Path, *, validate: bool) -> pd.DataFrame:
        nonlocal load_calls
        del path, validate
        load_calls += 1
        return dataframe.copy()

    monkeypatch.setattr(
        final_module,
        "load_config",
        lambda path: deepcopy(config),
    )
    monkeypatch.setattr(
        final_module,
        "load_base_dataset",
        fake_load_base_dataset,
    )

    first = run_phase6_final_evaluation("configs/base.yaml")
    second = run_phase6_final_evaluation("configs/base.yaml")

    assert load_calls == 1
    assert first.reused_existing_result is False
    assert second.reused_existing_result is True
    assert first.metadata["test_evaluation_count"] == 1
    assert second.metadata["test_evaluation_count"] == 1
    assert first.capacity_metrics["review_count"] == 5
    assert not hasattr(first, "test_features")
    assert not hasattr(first, "test_target")
    assert Path(config["reports"]["model_card_path"]).is_file()
    assert first.artifact_paths["completion"].is_file()


def test_interrupted_test_access_blocks_automatic_re_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An incomplete attempt must never silently reopen the test period."""
    config = build_phase6_config(tmp_path)
    dataframe = build_synthetic_dataset()
    write_phase5_artifacts(config, dataframe)
    phase6_directory = Path(config["artifacts"]["phase6_directory"])
    phase6_directory.mkdir(parents=True)
    (phase6_directory / "evaluation_state.json").write_text(
        json.dumps(
            {
                "status": "test_access_started",
                "test_access_started": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        final_module,
        "load_config",
        lambda path: deepcopy(config),
    )
    load_calls = 0

    def forbidden_dataset_load(path: Path, *, validate: bool) -> pd.DataFrame:
        nonlocal load_calls
        del path, validate
        load_calls += 1
        return dataframe.copy()

    monkeypatch.setattr(
        final_module,
        "load_base_dataset",
        forbidden_dataset_load,
    )

    with pytest.raises(RuntimeError, match="re-evaluation is blocked"):
        run_phase6_final_evaluation("configs/base.yaml")

    assert load_calls == 0


def test_missing_completion_is_recovered_without_test_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A lost final marker may be rebuilt from the pre-hashed full result."""
    config = build_phase6_config(tmp_path)
    dataframe = build_synthetic_dataset()
    write_phase5_artifacts(config, dataframe)
    load_calls = 0

    def fake_load_base_dataset(path: Path, *, validate: bool) -> pd.DataFrame:
        nonlocal load_calls
        del path, validate
        load_calls += 1
        return dataframe.copy()

    monkeypatch.setattr(
        final_module,
        "load_config",
        lambda path: deepcopy(config),
    )
    monkeypatch.setattr(
        final_module,
        "load_base_dataset",
        fake_load_base_dataset,
    )
    first = run_phase6_final_evaluation("configs/base.yaml")
    first.artifact_paths["completion"].unlink()
    recovered = run_phase6_final_evaluation("configs/base.yaml")

    assert load_calls == 1
    assert recovered.reused_existing_result is True
    assert recovered.artifact_paths["completion"].is_file()
    completion = json.loads(
        recovered.artifact_paths["completion"].read_text(encoding="utf-8")
    )
    assert completion["completion_recovered_without_test_access"] is True


def test_completed_result_rejects_tampered_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stored final evidence must fail closed after artifact modification."""
    config = build_phase6_config(tmp_path)
    dataframe = build_synthetic_dataset()
    write_phase5_artifacts(config, dataframe)
    monkeypatch.setattr(
        final_module,
        "load_config",
        lambda path: deepcopy(config),
    )
    monkeypatch.setattr(
        final_module,
        "load_base_dataset",
        lambda path, validate: dataframe.copy(),
    )
    result = run_phase6_final_evaluation("configs/base.yaml")
    result.artifact_paths["test_metrics"].write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="artifact changed"):
        run_phase6_final_evaluation("configs/base.yaml")


def test_rendered_model_card_forbids_automated_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The generated final document must preserve human-review boundaries."""
    config = build_phase6_config(tmp_path)
    dataframe = build_synthetic_dataset()
    write_phase5_artifacts(config, dataframe)
    monkeypatch.setattr(
        final_module,
        "load_config",
        lambda path: deepcopy(config),
    )
    monkeypatch.setattr(
        final_module,
        "load_base_dataset",
        lambda path, validate: dataframe.copy(),
    )
    result = run_phase6_final_evaluation("configs/base.yaml")

    model_card = render_model_card(result)

    assert "Automated rejection: **tidak diizinkan**" in model_card
    assert "Test evaluation count: `1`" in model_card
    assert "month 7" in model_card
