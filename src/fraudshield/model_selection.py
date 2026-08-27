"""Leakage-safe XGBoost training and model selection for Phase 4."""

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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraudshield.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
)
from fraudshield.data import load_base_dataset
from fraudshield.evaluate import (
    build_error_analysis_frame,
    evaluate_probabilities,
)
from fraudshield.features import (
    DataFrameColumnSelector,
    build_preprocessor,
    get_preprocessed_feature_names,
)
from fraudshield.train import (
    BaselineDevelopmentData,
    build_baseline_models,
    positive_class_probability,
    prepare_development_data,
)

_SAFE_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_CANDIDATE_PARAMETER_NAMES = (
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
)


@dataclass(frozen=True, slots=True)
class ModelSelectionDecision:
    """Auditable decision comparing the best tree model with the baseline."""

    baseline_model_name: str
    best_xgboost_model_name: str
    selected_model_name: str
    xgboost_promoted: bool
    validation_average_precision_baseline: float
    validation_average_precision_challenger: float
    absolute_average_precision_improvement: float
    minimum_absolute_average_precision_improvement: float
    operational_review_rate: float
    validation_recall_at_capacity_baseline: float
    validation_recall_at_capacity_challenger: float
    recall_at_capacity_change: float
    maximum_recall_at_capacity_drop: float
    average_precision_guardrail_passed: bool
    recall_guardrail_passed: bool
    ranked_xgboost_candidates: tuple[str, ...]
    reason: str


@dataclass(slots=True)
class Phase4ExperimentResult:
    """Models, development metrics, and decision from one Phase 4 run."""

    config: dict[str, Any]
    development_data: BaselineDevelopmentData
    models: dict[str, Pipeline]
    xgboost_candidate_names: tuple[str, ...]
    scale_pos_weight: float
    training_seconds: dict[str, float]
    metrics: dict[str, dict[str, dict[str, object]]]
    comparison: pd.DataFrame
    decision: ModelSelectionDecision
    artifact_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def selected_model(self) -> Pipeline:
        """Return the model chosen by the locked development policy."""
        return self.models[self.decision.selected_model_name]


def calculate_scale_pos_weight(target: pd.Series | np.ndarray) -> float:
    """Return negative-to-positive ratio using training labels only."""
    target_array = np.asarray(target)

    if target_array.ndim != 1 or len(target_array) == 0:
        raise ValueError("Training target must be a non-empty 1D array.")

    if not np.isin(target_array, [0, 1]).all():
        raise ValueError("Training target must contain only zero and one.")

    positive_count = int(target_array.sum())
    negative_count = int(len(target_array) - positive_count)

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "scale_pos_weight requires both target classes in training data."
        )

    return float(negative_count / positive_count)


def _candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return the deliberately small candidate list."""
    candidates = config["xgboost"].get("candidates")

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("xgboost.candidates must be a non-empty list.")

    maximum_candidates = int(config["xgboost"]["maximum_candidates"])

    if maximum_candidates < 1 or len(candidates) > maximum_candidates:
        raise ValueError(
            "The configured XGBoost shortlist exceeds maximum_candidates."
        )

    normalized: list[dict[str, Any]] = []
    names: list[str] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Every XGBoost candidate must be a mapping.")

        missing_parameters = sorted(
            {"name", *_CANDIDATE_PARAMETER_NAMES} - set(candidate)
        )

        if missing_parameters:
            raise KeyError(
                "XGBoost candidate is missing parameters: "
                f"{missing_parameters}"
            )

        name = candidate["name"]

        if not isinstance(name, str) or not _SAFE_MODEL_NAME.fullmatch(name):
            raise ValueError(
                "XGBoost candidate names must use lowercase letters, "
                "numbers, and underscores."
            )

        names.append(name)

        if int(candidate["n_estimators"]) < 1:
            raise ValueError("n_estimators must be positive.")

        if float(candidate["learning_rate"]) <= 0:
            raise ValueError("learning_rate must be positive.")

        if int(candidate["max_depth"]) < 1:
            raise ValueError("max_depth must be positive.")

        if float(candidate["min_child_weight"]) < 0:
            raise ValueError("min_child_weight must be non-negative.")

        for rate_parameter in ("subsample", "colsample_bytree"):
            rate = float(candidate[rate_parameter])

            if not 0 < rate <= 1:
                raise ValueError(
                    f"{rate_parameter} must lie within the interval (0, 1]."
                )

        for regularization_parameter in (
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ):
            if float(candidate[regularization_parameter]) < 0:
                raise ValueError(
                    f"{regularization_parameter} must be non-negative."
                )

        normalized.append(dict(candidate))

    duplicate_names = sorted(
        name for name in set(names) if names.count(name) > 1
    )

    if duplicate_names:
        raise ValueError(
            f"XGBoost candidate names must be unique: {duplicate_names}"
        )

    return normalized


def _build_xgboost_pipeline(
    train_features: pd.DataFrame,
    *,
    candidate: dict[str, Any],
    scale_pos_weight: float,
    config: dict[str, Any],
) -> Pipeline:
    """Build one preprocessing-plus-XGBoost pipeline without fitting it."""
    preprocessing_config = config["preprocessing"]
    xgboost_config = config["xgboost"]
    selected_features = train_features.columns.tolist()
    categorical_features = [
        column
        for column in preprocessing_config["categorical_features"]
        if column in selected_features
    ]
    semantic_missing_rules = {
        column: rule
        for column, rule in preprocessing_config[
            "semantic_missing_rules"
        ].items()
        if column in selected_features
    }
    preprocessor = build_preprocessor(
        train_features.loc[:, selected_features],
        categorical_features=categorical_features,
        semantic_missing_rules=semantic_missing_rules,
        add_missing_indicators=preprocessing_config[
            "add_missing_indicators"
        ],
        numeric_imputer_strategy=preprocessing_config[
            "numeric_imputer_strategy"
        ],
        categorical_imputer_strategy=preprocessing_config[
            "categorical_imputer_strategy"
        ],
        scale_numeric=bool(xgboost_config["scale_numeric"]),
    )
    candidate_parameters = {
        name: candidate[name] for name in _CANDIDATE_PARAMETER_NAMES
    }
    classifier = XGBClassifier(
        objective=xgboost_config["objective"],
        eval_metric=xgboost_config["eval_metric"],
        tree_method=xgboost_config["tree_method"],
        importance_type=xgboost_config["importance_type"],
        n_jobs=int(xgboost_config["n_jobs"]),
        verbosity=int(xgboost_config["verbosity"]),
        random_state=int(config["project"]["random_seed"]),
        scale_pos_weight=scale_pos_weight,
        **candidate_parameters,
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


def build_xgboost_models(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    config: dict[str, Any],
) -> tuple[dict[str, Pipeline], float]:
    """Build all configured candidates with train-only class weighting."""
    scale_pos_weight = calculate_scale_pos_weight(train_target)
    baseline_name = config["model_selection"]["baseline_model_name"]
    candidates = _candidate_specs(config)

    if baseline_name in {candidate["name"] for candidate in candidates}:
        raise ValueError(
            "An XGBoost candidate name collides with the baseline name."
        )

    models = {
        candidate["name"]: _build_xgboost_pipeline(
            train_features,
            candidate=candidate,
            scale_pos_weight=scale_pos_weight,
            config=config,
        )
        for candidate in candidates
    }

    return models, scale_pos_weight


def fit_phase4_models(
    models: dict[str, Pipeline],
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> dict[str, float]:
    """Fit models sequentially on train and return wall-clock durations."""
    training_seconds: dict[str, float] = {}

    for model_name, model in models.items():
        started_at = time.perf_counter()
        model.fit(train_features, train_target)
        training_seconds[model_name] = float(
            time.perf_counter() - started_at
        )

    return training_seconds


def evaluate_phase4_models(
    models: dict[str, Pipeline],
    development_data: BaselineDevelopmentData,
    config: dict[str, Any],
) -> dict[str, dict[str, dict[str, object]]]:
    """Evaluate calibration and validation only; test is unavailable here."""
    evaluation_config = config["evaluation"]
    development_splits = {
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
        split_metrics: dict[str, dict[str, object]] = {}

        for split_name, (features, target) in development_splits.items():
            probabilities = positive_class_probability(model, features)
            split_metrics[split_name] = evaluate_probabilities(
                target,
                probabilities,
                review_rates=evaluation_config["review_rates"],
                diagnostic_threshold=evaluation_config[
                    "diagnostic_threshold"
                ],
                calibration_bins=evaluation_config["calibration_bins"],
            )

        metrics[model_name] = split_metrics

    return metrics


def _selection_review_key(config: dict[str, Any]) -> str:
    review_rate = float(
        config["model_selection"]["operational_review_rate"]
    )
    configured_review_rates = {
        float(rate) for rate in config["evaluation"]["review_rates"]
    }

    if not 0 < review_rate <= 1:
        raise ValueError(
            "model_selection.operational_review_rate must lie within (0, 1]."
        )

    if review_rate not in configured_review_rates:
        raise ValueError(
            "model_selection.operational_review_rate must also appear in "
            "evaluation.review_rates."
        )

    return f"{review_rate:.4f}"


def build_model_comparison_frame(
    metrics: dict[str, dict[str, dict[str, object]]],
    training_seconds: dict[str, float],
    *,
    xgboost_candidate_names: tuple[str, ...],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the development-only table used by the selection decision."""
    review_key = _selection_review_key(config)
    baseline_name = config["model_selection"]["baseline_model_name"]
    expected_names = {baseline_name, *xgboost_candidate_names}

    if set(metrics) != expected_names:
        raise ValueError(
            "Metrics must contain exactly the baseline and XGBoost "
            f"candidates. Expected={sorted(expected_names)}; "
            f"received={sorted(metrics)}."
        )

    if set(training_seconds) != expected_names:
        raise ValueError("Training durations do not match evaluated models.")

    records: list[dict[str, Any]] = []

    for model_name, split_metrics in metrics.items():
        if set(split_metrics) != {"calibration", "validation"}:
            raise ValueError(
                "Phase 4 selection accepts calibration and validation "
                "metrics only."
            )

        calibration_metrics = split_metrics["calibration"]
        validation_metrics = split_metrics["validation"]
        capacity_metrics = validation_metrics["review_rate_metrics"][
            review_key
        ]
        records.append(
            {
                "model": model_name,
                "model_family": (
                    "baseline_logistic"
                    if model_name == baseline_name
                    else "xgboost"
                ),
                "calibration_average_precision": calibration_metrics[
                    "pr_auc_average_precision"
                ],
                "validation_average_precision": validation_metrics[
                    "pr_auc_average_precision"
                ],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_brier_score": validation_metrics[
                    "brier_score"
                ],
                "validation_precision_at_capacity": capacity_metrics[
                    "precision_at_capacity"
                ],
                "validation_recall_at_capacity": capacity_metrics[
                    "recall_at_capacity"
                ],
                "training_seconds": training_seconds[model_name],
            }
        )

    comparison = pd.DataFrame.from_records(records)

    return comparison.sort_values(
        [
            "validation_average_precision",
            "validation_recall_at_capacity",
            "validation_brier_score",
            "training_seconds",
            "model",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
        ignore_index=True,
    )


def select_development_model(
    comparison: pd.DataFrame,
    *,
    xgboost_candidate_names: tuple[str, ...],
    config: dict[str, Any],
) -> ModelSelectionDecision:
    """Select a candidate without consulting calibration or test outcomes."""
    selection_config = config["model_selection"]

    if selection_config["primary_metric"] != "pr_auc_average_precision":
        raise ValueError(
            "Phase 4 currently supports pr_auc_average_precision as the "
            "locked primary selection metric."
        )

    expected_tie_breakers = (
        "recall_at_capacity",
        "brier_score",
        "training_seconds",
        "model_name",
    )

    if tuple(selection_config["tie_breakers"]) != expected_tie_breakers:
        raise ValueError(
            "Configured tie_breakers do not match the implemented locked "
            "selection order."
        )

    validation_periods = tuple(config["split"]["validation_periods"])
    selection_periods = tuple(selection_config["selection_periods"])

    if selection_periods != validation_periods:
        raise ValueError(
            "Model selection periods must exactly match validation periods."
        )

    baseline_name = selection_config["baseline_model_name"]
    baseline_rows = comparison.loc[comparison["model"] == baseline_name]
    challenger_rows = comparison.loc[
        comparison["model"].isin(xgboost_candidate_names)
    ].sort_values(
        [
            "validation_average_precision",
            "validation_recall_at_capacity",
            "validation_brier_score",
            "training_seconds",
            "model",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )

    if len(baseline_rows) != 1 or challenger_rows.empty:
        raise ValueError(
            "Selection requires one baseline and at least one XGBoost row."
        )

    baseline = baseline_rows.iloc[0]
    challenger = challenger_rows.iloc[0]
    ap_improvement = float(
        challenger["validation_average_precision"]
        - baseline["validation_average_precision"]
    )
    recall_change = float(
        challenger["validation_recall_at_capacity"]
        - baseline["validation_recall_at_capacity"]
    )
    minimum_ap_improvement = float(
        selection_config["minimum_absolute_ap_improvement"]
    )
    maximum_recall_drop = float(
        selection_config["maximum_recall_at_capacity_drop"]
    )

    if minimum_ap_improvement < 0:
        raise ValueError(
            "minimum_absolute_ap_improvement must be non-negative."
        )

    if not 0 <= maximum_recall_drop <= 1:
        raise ValueError(
            "maximum_recall_at_capacity_drop must lie within [0, 1]."
        )
    ap_guardrail_passed = ap_improvement >= minimum_ap_improvement
    recall_guardrail_passed = recall_change >= -maximum_recall_drop
    xgboost_promoted = ap_guardrail_passed and recall_guardrail_passed
    challenger_name = str(challenger["model"])
    selected_name = challenger_name if xgboost_promoted else baseline_name

    if xgboost_promoted:
        reason = (
            "The best XGBoost candidate cleared both the locked average-"
            "precision improvement and review-capacity recall guardrails."
        )
    else:
        failed_guardrails = []

        if not ap_guardrail_passed:
            failed_guardrails.append("average-precision improvement")

        if not recall_guardrail_passed:
            failed_guardrails.append("review-capacity recall")

        reason = (
            "The logistic baseline was retained because the best XGBoost "
            "candidate did not clear: " + ", ".join(failed_guardrails) + "."
        )

    return ModelSelectionDecision(
        baseline_model_name=baseline_name,
        best_xgboost_model_name=challenger_name,
        selected_model_name=selected_name,
        xgboost_promoted=xgboost_promoted,
        validation_average_precision_baseline=float(
            baseline["validation_average_precision"]
        ),
        validation_average_precision_challenger=float(
            challenger["validation_average_precision"]
        ),
        absolute_average_precision_improvement=ap_improvement,
        minimum_absolute_average_precision_improvement=(
            minimum_ap_improvement
        ),
        operational_review_rate=float(
            selection_config["operational_review_rate"]
        ),
        validation_recall_at_capacity_baseline=float(
            baseline["validation_recall_at_capacity"]
        ),
        validation_recall_at_capacity_challenger=float(
            challenger["validation_recall_at_capacity"]
        ),
        recall_at_capacity_change=recall_change,
        maximum_recall_at_capacity_drop=maximum_recall_drop,
        average_precision_guardrail_passed=ap_guardrail_passed,
        recall_guardrail_passed=recall_guardrail_passed,
        ranked_xgboost_candidates=tuple(challenger_rows["model"]),
        reason=reason,
    )


def xgboost_importance_frame(model: Pipeline) -> pd.DataFrame:
    """Return sorted built-in gain importances for a fitted XGBoost model."""
    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")

    if not isinstance(classifier, XGBClassifier) or preprocessor is None:
        raise TypeError("Expected a fitted preprocessing-plus-XGBoost pipeline.")

    feature_names = get_preprocessed_feature_names(preprocessor)
    feature_importance = np.asarray(classifier.feature_importances_)

    if len(feature_names) != len(feature_importance):
        raise RuntimeError(
            "Preprocessed feature names and XGBoost importances differ in "
            "length."
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": feature_importance,
        }
    )

    return importance.sort_values(
        "gain_importance",
        ascending=False,
        ignore_index=True,
    )


def save_phase4_artifacts(
    result: Phase4ExperimentResult,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Path]:
    """Persist development-only models, metrics, and the selection record."""
    artifact_directory = resolve_project_path(
        result.config["artifacts"]["phase4_directory"],
        config_path=config_path,
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}

    for model_name, model in result.models.items():
        model_path = artifact_directory / f"{model_name}.joblib"
        joblib.dump(model, model_path, compress=3)
        artifact_paths[f"model:{model_name}"] = model_path

    artifact_paths["selected_model"] = artifact_paths[
        f"model:{result.decision.selected_model_name}"
    ]
    metrics_path = artifact_directory / "development_metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["metrics"] = metrics_path

    comparison_path = artifact_directory / "model_comparison.csv"
    result.comparison.to_csv(comparison_path, index=False)
    artifact_paths["comparison"] = comparison_path

    decision_path = artifact_directory / "selection_decision.json"
    decision_path.write_text(
        json.dumps(asdict(result.decision), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["decision"] = decision_path

    for model_name in result.xgboost_candidate_names:
        importance_path = (
            artifact_directory / f"{model_name}_gain_importance.csv"
        )
        xgboost_importance_frame(result.models[model_name]).to_csv(
            importance_path,
            index=False,
        )
        artifact_paths[f"importance:{model_name}"] = importance_path

    selected_probabilities = positive_class_probability(
        result.selected_model,
        result.development_data.validation_features,
    )
    error_examples = build_error_analysis_frame(
        result.development_data.validation_features,
        result.development_data.validation_target,
        selected_probabilities,
        diagnostic_threshold=result.config["evaluation"][
            "diagnostic_threshold"
        ],
    )
    error_path = artifact_directory / "selected_validation_errors.csv"
    error_examples.to_csv(error_path, index=False)
    artifact_paths["validation_errors"] = error_path

    metadata = {
        "phase": 4,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "random_seed": int(result.config["project"]["random_seed"]),
        "python_packages": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "train_periods": result.config["split"]["train_periods"],
        "calibration_periods": result.config["split"][
            "calibration_periods"
        ],
        "validation_periods": result.config["split"][
            "validation_periods"
        ],
        "reserved_test_periods": result.config["split"]["test_periods"],
        "reserved_test_rows": result.development_data.reserved_test_rows,
        "test_features_exposed": False,
        "test_evaluated": False,
        "preprocessing_fitted_inside_model_pipeline": True,
        "scale_pos_weight": result.scale_pos_weight,
        "scale_pos_weight_source": "training_target_only",
        "fixed_boosting_rounds": True,
        "early_stopping_used": False,
        "model_selected_on": "validation_periods_only",
        "selected_model_name": result.decision.selected_model_name,
        "probability_calibrated": False,
        "business_threshold_selected": False,
        "training_seconds": result.training_seconds,
        "candidate_parameters": result.config["xgboost"]["candidates"],
    }
    metadata_path = artifact_directory / "phase4_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["metadata"] = metadata_path
    result.artifact_paths = artifact_paths

    return artifact_paths


def run_phase4_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    save_artifacts: bool = True,
) -> Phase4ExperimentResult:
    """Run baseline comparison, XGBoost candidates, and model selection."""
    config = load_config(config_path)
    raw_dataset_path = resolve_project_path(
        config["data"]["raw_path"],
        config_path=config_path,
    )
    dataframe = load_base_dataset(raw_dataset_path, validate=True)
    development_data = prepare_development_data(dataframe, config)
    del dataframe

    baseline_name = config["model_selection"]["baseline_model_name"]
    baseline_models = build_baseline_models(
        development_data.train_features,
        config,
    )

    if baseline_name not in baseline_models:
        raise KeyError(
            f"Configured model-selection baseline does not exist: {baseline_name}"
        )

    xgboost_models, scale_pos_weight = build_xgboost_models(
        development_data.train_features,
        development_data.train_target,
        config,
    )
    models = {baseline_name: baseline_models[baseline_name], **xgboost_models}
    candidate_names = tuple(xgboost_models)
    training_seconds = fit_phase4_models(
        models,
        development_data.train_features,
        development_data.train_target,
    )
    metrics = evaluate_phase4_models(models, development_data, config)
    comparison = build_model_comparison_frame(
        metrics,
        training_seconds,
        xgboost_candidate_names=candidate_names,
        config=config,
    )
    decision = select_development_model(
        comparison,
        xgboost_candidate_names=candidate_names,
        config=config,
    )
    comparison = comparison.assign(
        selected=lambda frame: frame["model"].eq(
            decision.selected_model_name
        ),
        best_xgboost_candidate=lambda frame: frame["model"].eq(
            decision.best_xgboost_model_name
        ),
    )
    result = Phase4ExperimentResult(
        config=config,
        development_data=development_data,
        models=models,
        xgboost_candidate_names=candidate_names,
        scale_pos_weight=scale_pos_weight,
        training_seconds=training_seconds,
        metrics=metrics,
        comparison=comparison,
        decision=decision,
    )

    if save_artifacts:
        save_phase4_artifacts(result, config_path=config_path)

    return result


def _print_selection_summary(result: Phase4ExperimentResult) -> None:
    """Print the locked development decision without test metrics."""
    decision = result.decision
    print("Phase 4 XGBoost model selection completed.")
    print(
        "Reserved test rows (not evaluated): "
        f"{result.development_data.reserved_test_rows:,}"
    )
    print(f"Train-only scale_pos_weight: {result.scale_pos_weight:.6f}")
    print(
        "Best XGBoost candidate: "
        f"{decision.best_xgboost_model_name}"
    )
    print(f"Selected model: {decision.selected_model_name}")
    print(
        "Validation AP improvement versus baseline: "
        f"{decision.absolute_average_precision_improvement:+.6f}"
    )
    print(decision.reason)


def main() -> None:
    """Run the Phase 4 command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run leakage-safe FraudShield XGBoost model selection."
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
    result = run_phase4_experiment(
        arguments.config,
        save_artifacts=not arguments.no_save,
    )
    _print_selection_summary(result)


if __name__ == "__main__":
    main()
