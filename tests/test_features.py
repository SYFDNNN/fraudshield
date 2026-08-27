"""Tests for leakage-safe feature preprocessing."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fraudshield.features import (
    DataFrameColumnSelector,
    SemanticMissingTransformer,
    build_preprocessor,
    get_preprocessed_feature_names,
)


@pytest.fixture
def train_features() -> pd.DataFrame:
    """Create a small mixed-type training feature frame."""
    return pd.DataFrame(
        {
            "income": [0.2, 0.5, 0.8],
            "prev_address_months_count": [-1, 10, 20],
            "intended_balcon_amount": [-5.0, 100.0, 200.0],
            "payment_type": ["AA", "AB", "AA"],
        }
    )


@pytest.fixture
def semantic_missing_rules() -> dict[str, dict[str, float | str]]:
    """Return sentinel rules used by the preprocessing tests."""
    return {
        "prev_address_months_count": {
            "operator": "equals",
            "value": -1,
        },
        "intended_balcon_amount": {
            "operator": "less_than",
            "value": 0,
        },
    }


def test_semantic_missing_values_become_nan_with_indicators(
    train_features: pd.DataFrame,
    semantic_missing_rules: dict[str, dict[str, float | str]],
) -> None:
    """Documented sentinel values should be represented explicitly."""
    transformer = SemanticMissingTransformer(
        semantic_missing_rules,
        add_indicators=True,
    )

    transformed = transformer.fit_transform(train_features)

    assert pd.isna(transformed.loc[0, "prev_address_months_count"])
    assert pd.isna(transformed.loc[0, "intended_balcon_amount"])
    assert transformed.loc[0, "prev_address_months_count__missing"] == 1
    assert transformed.loc[1, "prev_address_months_count__missing"] == 0
    assert transformed.loc[0, "intended_balcon_amount__missing"] == 1


def test_preprocessor_handles_unknown_future_category(
    train_features: pd.DataFrame,
    semantic_missing_rules: dict[str, dict[str, float | str]],
) -> None:
    """A category absent from training must not break later periods."""
    preprocessor = build_preprocessor(
        train_features,
        categorical_features=["payment_type"],
        semantic_missing_rules=semantic_missing_rules,
    )
    future_features = train_features.iloc[[0]].copy()
    future_features.loc[:, "payment_type"] = "UNSEEN"

    transformed_train = preprocessor.fit_transform(train_features)
    transformed_future = preprocessor.transform(future_features)

    assert sparse.issparse(transformed_train)
    assert sparse.issparse(transformed_future)
    assert transformed_future.shape[1] == transformed_train.shape[1]
    assert not np.isnan(transformed_future.data).any()


def test_numeric_imputer_statistics_are_learned_from_train_only(
    train_features: pd.DataFrame,
    semantic_missing_rules: dict[str, dict[str, float | str]],
) -> None:
    """Future values must not influence a fitted training median."""
    preprocessor = build_preprocessor(
        train_features,
        categorical_features=["payment_type"],
        semantic_missing_rules=semantic_missing_rules,
    )
    preprocessor.fit(train_features)
    column_transformer = preprocessor.named_steps["columns"]
    numeric_columns = column_transformer.transformers_[0][2]
    numeric_imputer = column_transformer.named_transformers_[
        "numeric"
    ].named_steps["imputer"]
    statistics = dict(
        zip(numeric_columns, numeric_imputer.statistics_, strict=True)
    )

    assert statistics["prev_address_months_count"] == pytest.approx(15.0)
    assert statistics["intended_balcon_amount"] == pytest.approx(150.0)

    future_features = train_features.iloc[[0]].copy()
    future_features.loc[:, "prev_address_months_count"] = 999
    preprocessor.transform(future_features)

    assert statistics["prev_address_months_count"] == pytest.approx(15.0)


def test_preprocessor_returns_traceable_feature_names(
    train_features: pd.DataFrame,
    semantic_missing_rules: dict[str, dict[str, float | str]],
) -> None:
    """Transformed columns should remain available for interpretation."""
    preprocessor = build_preprocessor(
        train_features,
        categorical_features=["payment_type"],
        semantic_missing_rules=semantic_missing_rules,
    )
    preprocessor.fit(train_features)

    feature_names = set(get_preprocessed_feature_names(preprocessor))

    assert "prev_address_months_count__missing" in feature_names
    assert "intended_balcon_amount__missing" in feature_names
    assert "payment_type_AA" in feature_names
    assert "payment_type_AB" in feature_names


def test_column_selector_rejects_missing_inference_feature(
    train_features: pd.DataFrame,
) -> None:
    """A saved model should fail clearly when a required field is absent."""
    selector = DataFrameColumnSelector(train_features.columns.tolist())
    selector.fit(train_features)

    with pytest.raises(KeyError, match="Required model features"):
        selector.transform(train_features.drop(columns="income"))
