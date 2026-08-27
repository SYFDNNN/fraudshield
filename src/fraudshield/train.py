"""Leakage-safe baseline training for FraudShield Phase 3."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraudshield.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
)
from fraudshield.data import (
    load_base_dataset,
    select_model_features,
    temporal_split,
)
from fraudshield.evaluate import (
    build_error_analysis_frame,
    evaluate_probabilities,
)
from fraudshield.features import (
    DataFrameColumnSelector,
    build_preprocessor,
    get_preprocessed_feature_names,
)


@dataclass(slots=True)
class BaselineDevelopmentData:
    """Development partitions exposed to Phase 3 modeling code.

    Test features and labels are intentionally absent from this container.
    """

    train_features: pd.DataFrame
    train_target: pd.Series
    calibration_features: pd.DataFrame
    calibration_target: pd.Series
    validation_features: pd.DataFrame
    validation_target: pd.Series
    reserved_test_rows: int


@dataclass(slots=True)
class BaselineExperimentResult:
    """Models, metrics, and metadata produced by one baseline run."""

    config: dict[str, Any]
    development_data: BaselineDevelopmentData
    models: dict[str, Pipeline]
    training_seconds: dict[str, float]
    metrics: dict[str, dict[str, dict[str, object]]]
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def prepare_development_data(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> BaselineDevelopmentData:
    """Create model inputs while keeping the test partition inaccessible."""
    data_config = config["data"]
    split_config = config["split"]
    feature_policy = config["feature_policy"]
    target_column = data_config["target_column"]

    splits = temporal_split(
        dataframe,
        time_column=data_config["time_column"],
        train_periods=split_config["train_periods"],
        calibration_periods=split_config["calibration_periods"],
        validation_periods=split_config["validation_periods"],
        test_periods=split_config["test_periods"],
    )
    excluded_columns = feature_policy["excluded_from_model"]
    train_features = select_model_features(
        splits.train,
        excluded_columns=excluded_columns,
    )
    calibration_features = select_model_features(
        splits.calibration,
        excluded_columns=excluded_columns,
    )
    validation_features = select_model_features(
        splits.validation,
        excluded_columns=excluded_columns,
    )

    for split_name, feature_frame in {
        "calibration": calibration_features,
        "validation": validation_features,
    }.items():
        if feature_frame.columns.tolist() != train_features.columns.tolist():
            raise ValueError(
                f"{split_name} feature schema differs from training schema."
            )

    return BaselineDevelopmentData(
        train_features=train_features,
        train_target=splits.train[target_column].astype(np.int8).copy(),
        calibration_features=calibration_features,
        calibration_target=(
            splits.calibration[target_column].astype(np.int8).copy()
        ),
        validation_features=validation_features,
        validation_target=(
            splits.validation[target_column].astype(np.int8).copy()
        ),
        reserved_test_rows=len(splits.test),
    )


def _build_logistic_pipeline(
    train_features: pd.DataFrame,
    *,
    selected_features: list[str],
    config: dict[str, Any],
    random_seed: int,
) -> Pipeline:
    """Build one class-weighted logistic-regression pipeline."""
    preprocessing_config = config["preprocessing"]
    model_config = config["baseline"]["logistic_regression"]
    selected_train_features = train_features.loc[:, selected_features]
    selected_categorical_features = [
        column
        for column in preprocessing_config["categorical_features"]
        if column in selected_features
    ]
    selected_missing_rules = {
        column: rule
        for column, rule in preprocessing_config[
            "semantic_missing_rules"
        ].items()
        if column in selected_features
    }
    preprocessor = build_preprocessor(
        selected_train_features,
        categorical_features=selected_categorical_features,
        semantic_missing_rules=selected_missing_rules,
        add_missing_indicators=preprocessing_config[
            "add_missing_indicators"
        ],
        numeric_imputer_strategy=preprocessing_config[
            "numeric_imputer_strategy"
        ],
        categorical_imputer_strategy=preprocessing_config[
            "categorical_imputer_strategy"
        ],
        scale_numeric=preprocessing_config["scale_numeric"],
    )
    classifier = LogisticRegression(
        C=float(model_config["C"]),
        class_weight=model_config["class_weight"],
        solver=model_config["solver"],
        max_iter=int(model_config["max_iter"]),
        tol=float(model_config["tolerance"]),
        random_state=random_seed,
    )

    return Pipeline(
        steps=[
            (
                "feature_selection",
                DataFrameColumnSelector(selected_features),
            ),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_baseline_models(
    train_features: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Pipeline]:
    """Build the dummy, primary logistic, and configured ablation models."""
    random_seed = int(config["project"]["random_seed"])
    all_features = train_features.columns.tolist()
    dummy_config = config["baseline"]["dummy"]
    models: dict[str, Pipeline] = {
        "dummy_prior": Pipeline(
            steps=[
                (
                    "feature_selection",
                    DataFrameColumnSelector(all_features),
                ),
                (
                    "classifier",
                    DummyClassifier(
                        strategy=dummy_config["strategy"],
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        config["baseline"]["primary_model_name"]: (
            _build_logistic_pipeline(
                train_features,
                selected_features=all_features,
                config=config,
                random_seed=random_seed,
            )
        ),
    }
    ablation_config = config["baseline"]["high_drift_ablation"]

    if ablation_config["enabled"]:
        high_drift_features = set(
            config["feature_policy"]["high_drift_features"]
        )
        unknown_drift_features = sorted(
            high_drift_features - set(all_features)
        )

        if unknown_drift_features:
            raise KeyError(
                "Configured high-drift features are absent from the model "
                f"feature set: {unknown_drift_features}"
            )

        ablation_features = [
            feature
            for feature in all_features
            if feature not in high_drift_features
        ]
        models[ablation_config["model_name"]] = _build_logistic_pipeline(
            train_features,
            selected_features=ablation_features,
            config=config,
            random_seed=random_seed,
        )

    return models


def fit_baseline_models(
    models: dict[str, Pipeline],
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> dict[str, float]:
    """Fit every baseline on training rows only and return fit durations."""
    training_seconds: dict[str, float] = {}

    for model_name, model in models.items():
        started_at = time.perf_counter()
        model.fit(train_features, train_target)
        training_seconds[model_name] = float(
            time.perf_counter() - started_at
        )

    return training_seconds


def positive_class_probability(
    model: Pipeline,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    """Return the probability assigned to class one by a fitted pipeline."""
    classifier = model.named_steps["classifier"]
    class_indexes = np.flatnonzero(classifier.classes_ == 1)

    if len(class_indexes) != 1:
        raise ValueError("Fitted classifier does not expose binary class one.")

    return model.predict_proba(feature_frame)[:, int(class_indexes[0])]


def evaluate_baseline_models(
    models: dict[str, Pipeline],
    development_data: BaselineDevelopmentData,
    config: dict[str, Any],
) -> dict[str, dict[str, dict[str, object]]]:
    """Evaluate baselines on calibration and validation, never on test."""
    evaluation_config = config["evaluation"]
    evaluation_splits = {
        "calibration": (
            development_data.calibration_features,
            development_data.calibration_target,
        ),
        "validation": (
            development_data.validation_features,
            development_data.validation_target,
        ),
    }
    metrics: dict[str, dict[str, dict[str, object]]] = {}

    for model_name, model in models.items():
        model_metrics: dict[str, dict[str, object]] = {}

        for split_name, (feature_frame, target) in evaluation_splits.items():
            probabilities = positive_class_probability(model, feature_frame)
            model_metrics[split_name] = evaluate_probabilities(
                target,
                probabilities,
                review_rates=evaluation_config["review_rates"],
                diagnostic_threshold=evaluation_config[
                    "diagnostic_threshold"
                ],
                calibration_bins=evaluation_config["calibration_bins"],
            )

        metrics[model_name] = model_metrics

    return metrics


def logistic_coefficient_frame(model: Pipeline) -> pd.DataFrame:
    """Return sorted coefficients from a fitted logistic pipeline."""
    if "preprocessor" not in model.named_steps:
        raise ValueError("Model does not contain a preprocessing pipeline.")

    preprocessor = model.named_steps["preprocessor"]
    feature_names = get_preprocessed_feature_names(preprocessor)
    coefficients = model.named_steps["classifier"].coef_.reshape(-1)

    if len(feature_names) != len(coefficients):
        raise RuntimeError(
            "Preprocessed feature names and coefficients have different lengths."
        )

    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )

    return coefficient_frame.sort_values(
        "absolute_coefficient",
        ascending=False,
        ignore_index=True,
    )


def save_baseline_artifacts(
    result: BaselineExperimentResult,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Path]:
    """Persist fitted baselines and development-only reports locally."""
    artifact_directory = resolve_project_path(
        result.config["artifacts"]["phase3_directory"],
        config_path=config_path,
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}

    for model_name, model in result.models.items():
        model_path = artifact_directory / f"{model_name}.joblib"
        joblib.dump(model, model_path, compress=3)
        artifact_paths[f"model:{model_name}"] = model_path

    metrics_path = artifact_directory / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["metrics"] = metrics_path

    primary_model_name = result.config["baseline"]["primary_model_name"]
    primary_model = result.models[primary_model_name]
    validation_probabilities = positive_class_probability(
        primary_model,
        result.development_data.validation_features,
    )
    error_examples = build_error_analysis_frame(
        result.development_data.validation_features,
        result.development_data.validation_target,
        validation_probabilities,
        diagnostic_threshold=result.config["evaluation"][
            "diagnostic_threshold"
        ],
    )
    error_path = artifact_directory / "validation_error_examples.csv"
    error_examples.to_csv(error_path, index=False)
    artifact_paths["validation_errors"] = error_path

    for model_name, model in result.models.items():
        if "preprocessor" not in model.named_steps:
            continue

        coefficient_path = (
            artifact_directory / f"{model_name}_coefficients.csv"
        )
        logistic_coefficient_frame(model).to_csv(
            coefficient_path,
            index=False,
        )
        artifact_paths[f"coefficients:{model_name}"] = coefficient_path

    metadata = {
        "phase": 3,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "random_seed": int(result.config["project"]["random_seed"]),
        "target_column": result.config["data"]["target_column"],
        "excluded_features": result.config["feature_policy"][
            "excluded_from_model"
        ],
        "train_periods": result.config["split"]["train_periods"],
        "calibration_periods": result.config["split"][
            "calibration_periods"
        ],
        "validation_periods": result.config["split"][
            "validation_periods"
        ],
        "reserved_test_periods": result.config["split"]["test_periods"],
        "reserved_test_rows": result.development_data.reserved_test_rows,
        "test_evaluated": False,
        "training_seconds": result.training_seconds,
        "model_raw_features": {
            model_name: list(
                model.named_steps["feature_selection"].selected_columns_
            )
            for model_name, model in result.models.items()
        },
    }
    metadata_path = artifact_directory / "baseline_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["metadata"] = metadata_path
    result.artifact_paths = artifact_paths

    return artifact_paths


def run_baseline_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    save_artifacts: bool = True,
) -> BaselineExperimentResult:
    """Run the complete Phase 3 baseline experiment from configuration."""
    config = load_config(config_path)
    raw_dataset_path = resolve_project_path(
        config["data"]["raw_path"],
        config_path=config_path,
    )
    dataframe = load_base_dataset(raw_dataset_path, validate=True)
    development_data = prepare_development_data(dataframe, config)
    del dataframe
    models = build_baseline_models(
        development_data.train_features,
        config,
    )
    training_seconds = fit_baseline_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )
    metrics = evaluate_baseline_models(
        models,
        development_data,
        config,
    )
    result = BaselineExperimentResult(
        config=config,
        development_data=development_data,
        models=models,
        training_seconds=training_seconds,
        metrics=metrics,
    )

    if save_artifacts:
        save_baseline_artifacts(result, config_path=config_path)

    return result


def _print_validation_summary(result: BaselineExperimentResult) -> None:
    """Print concise development metrics without touching the test split."""
    print("Phase 3 baseline experiment completed.")
    print(
        "Reserved test rows (not evaluated): "
        f"{result.development_data.reserved_test_rows:,}"
    )

    for model_name, split_metrics in result.metrics.items():
        validation_metrics = split_metrics["validation"]
        print(
            f"{model_name}: "
            "validation AP="
            f"{validation_metrics['pr_auc_average_precision']:.6f}, "
            "ROC-AUC="
            f"{validation_metrics['roc_auc']:.6f}, "
            "Brier="
            f"{validation_metrics['brier_score']:.6f}"
        )


def main() -> None:
    """Run the Phase 3 command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Train leakage-safe FraudShield baseline models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the project YAML configuration.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the experiment without writing local model artifacts.",
    )
    arguments = parser.parse_args()
    result = run_baseline_experiment(
        arguments.config,
        save_artifacts=not arguments.no_save,
    )
    _print_validation_summary(result)


if __name__ == "__main__":
    main()
