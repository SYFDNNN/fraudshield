"""Leakage-safe feature preprocessing for FraudShield models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

SUPPORTED_MISSING_OPERATORS = frozenset({"equals", "less_than"})


def _require_dataframe(dataframe: pd.DataFrame, *, name: str) -> None:
    """Raise a clear error when a transformer receives a non-dataframe input."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if dataframe.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate column names.")


class DataFrameColumnSelector(BaseEstimator, TransformerMixin):
    """Select a stable ordered feature set from a pandas DataFrame."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = columns

    def fit(
        self,
        dataframe: pd.DataFrame,
        target: Any = None,
    ) -> DataFrameColumnSelector:
        """Validate the requested columns and remember the input schema."""
        del target
        _require_dataframe(dataframe, name="dataframe")

        selected_columns = tuple(self.columns)

        if not selected_columns:
            raise ValueError("At least one feature column must be selected.")

        if len(selected_columns) != len(set(selected_columns)):
            raise ValueError("Selected feature columns contain duplicates.")

        missing_columns = sorted(
            set(selected_columns) - set(dataframe.columns)
        )

        if missing_columns:
            raise KeyError(
                f"Selected feature columns do not exist: {missing_columns}"
            )

        self.feature_names_in_ = np.asarray(
            dataframe.columns,
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        self.selected_columns_ = selected_columns

        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return selected columns in the same order used during fitting."""
        check_is_fitted(self, "selected_columns_")
        _require_dataframe(dataframe, name="dataframe")

        missing_columns = sorted(
            set(self.selected_columns_) - set(dataframe.columns)
        )

        if missing_columns:
            raise KeyError(
                f"Required model features are missing: {missing_columns}"
            )

        return dataframe.loc[:, self.selected_columns_].copy()

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return the selected raw feature names."""
        del input_features
        check_is_fitted(self, "selected_columns_")
        return np.asarray(self.selected_columns_, dtype=object)


class SemanticMissingTransformer(BaseEstimator, TransformerMixin):
    """Convert documented sentinel values to missing values.

    The rules are fixed domain definitions rather than statistics learned from
    future data. Imputation statistics are learned later by the numeric and
    categorical pipelines using training rows only.
    """

    def __init__(
        self,
        rules: Mapping[str, Mapping[str, Any]],
        *,
        add_indicators: bool = True,
    ) -> None:
        self.rules = rules
        self.add_indicators = add_indicators

    def fit(
        self,
        dataframe: pd.DataFrame,
        target: Any = None,
    ) -> SemanticMissingTransformer:
        """Validate sentinel rules without learning distribution statistics."""
        del target
        _require_dataframe(dataframe, name="dataframe")

        missing_rule_columns = sorted(
            set(self.rules) - set(dataframe.columns)
        )

        if missing_rule_columns:
            raise KeyError(
                "Semantic-missing rule columns do not exist: "
                f"{missing_rule_columns}"
            )

        normalized_rules: dict[str, dict[str, Any]] = {}

        for column, rule in self.rules.items():
            if not isinstance(rule, Mapping):
                raise TypeError(
                    f"Semantic-missing rule for {column!r} must be a mapping."
                )

            operator = rule.get("operator")

            if operator not in SUPPORTED_MISSING_OPERATORS:
                raise ValueError(
                    f"Unsupported semantic-missing operator for {column!r}: "
                    f"{operator!r}."
                )

            if "value" not in rule:
                raise ValueError(
                    f"Semantic-missing rule for {column!r} has no value."
                )

            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                raise TypeError(
                    "Semantic-missing rules currently require numeric "
                    f"columns: {column!r}."
                )

            normalized_rules[column] = {
                "operator": operator,
                "value": rule["value"],
            }

        self.feature_names_in_ = np.asarray(
            dataframe.columns,
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        self.rules_ = normalized_rules
        self.indicator_columns_ = tuple(
            f"{column}__missing" for column in normalized_rules
        )

        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Replace sentinels with NaN and optionally add missing indicators."""
        check_is_fitted(self, "rules_")
        _require_dataframe(dataframe, name="dataframe")

        expected_columns = tuple(self.feature_names_in_)
        missing_columns = sorted(
            set(expected_columns) - set(dataframe.columns)
        )
        unexpected_columns = sorted(
            set(dataframe.columns) - set(expected_columns)
        )

        if missing_columns or unexpected_columns:
            raise ValueError(
                "Preprocessing input schema differs from training schema. "
                f"Missing columns={missing_columns}; "
                f"unexpected columns={unexpected_columns}."
            )

        transformed = dataframe.loc[:, expected_columns].copy()

        for column, rule in self.rules_.items():
            if rule["operator"] == "equals":
                missing_mask = transformed[column].eq(rule["value"])
            else:
                missing_mask = transformed[column].lt(rule["value"])

            if self.add_indicators:
                transformed[f"{column}__missing"] = missing_mask.astype(
                    np.int8
                )

            transformed[column] = transformed[column].mask(
                missing_mask,
                np.nan,
            )

        return transformed

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return original feature names plus semantic-missing indicators."""
        del input_features
        check_is_fitted(self, "rules_")
        output_features = list(self.feature_names_in_)

        if self.add_indicators:
            output_features.extend(self.indicator_columns_)

        return np.asarray(output_features, dtype=object)


def build_preprocessor(
    feature_frame: pd.DataFrame,
    *,
    categorical_features: Sequence[str],
    semantic_missing_rules: Mapping[str, Mapping[str, Any]],
    add_missing_indicators: bool = True,
    numeric_imputer_strategy: str = "median",
    categorical_imputer_strategy: str = "most_frequent",
    scale_numeric: bool = True,
) -> Pipeline:
    """Build an unfitted preprocessing pipeline for one raw feature schema.

    The returned pipeline must be placed inside the estimator pipeline and fit
    only on the training partition. Unknown categories are ignored at inference
    time so later temporal periods cannot break transformation.
    """
    _require_dataframe(feature_frame, name="feature_frame")

    feature_columns = tuple(feature_frame.columns)
    categorical_columns = tuple(categorical_features)

    if len(categorical_columns) != len(set(categorical_columns)):
        raise ValueError("Categorical feature names contain duplicates.")

    missing_categorical_columns = sorted(
        set(categorical_columns) - set(feature_columns)
    )

    if missing_categorical_columns:
        raise KeyError(
            "Configured categorical features do not exist: "
            f"{missing_categorical_columns}"
        )

    categorical_column_set = set(categorical_columns)
    numeric_columns = tuple(
        column
        for column in feature_columns
        if column not in categorical_column_set
    )

    non_numeric_columns = [
        column
        for column in numeric_columns
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]

    if non_numeric_columns:
        raise TypeError(
            "Features not configured as categorical must be numeric: "
            f"{non_numeric_columns}"
        )

    indicator_columns = tuple(
        f"{column}__missing" for column in semantic_missing_rules
    )
    transformed_numeric_columns = list(numeric_columns)

    if add_missing_indicators:
        transformed_numeric_columns.extend(indicator_columns)

    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy=numeric_imputer_strategy,
                keep_empty_features=True,
            ),
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(with_mean=False),
            )
        )

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=categorical_imputer_strategy,
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                transformed_numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                list(categorical_columns),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "semantic_missing",
                SemanticMissingTransformer(
                    rules=semantic_missing_rules,
                    add_indicators=add_missing_indicators,
                ),
            ),
            ("columns", column_transformer),
        ]
    )


def get_preprocessed_feature_names(preprocessor: Pipeline) -> np.ndarray:
    """Return output feature names from a fitted preprocessing pipeline."""
    check_is_fitted(preprocessor, "n_features_in_")

    if "columns" not in preprocessor.named_steps:
        raise KeyError("Pipeline does not contain a 'columns' step.")

    return preprocessor.named_steps["columns"].get_feature_names_out()
