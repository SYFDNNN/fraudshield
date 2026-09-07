"""Tests for local, grouped XGBoost TreeSHAP explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraudshield.explain import (
    TREE_SHAP_METHOD,
    TREE_SHAP_SCOPE,
    XGBoostTreeShapExplainer,
    aggregate_transformed_contributions,
    raw_feature_for_transformed_name,
    reason_contributions,
)
from fraudshield.features import DataFrameColumnSelector, build_preprocessor


def test_transformed_names_map_back_to_raw_contract_features() -> None:
    raw_features = ("income", "device_os", "bank_months_count")

    assert raw_feature_for_transformed_name("income", raw_features) == "income"
    assert (
        raw_feature_for_transformed_name(
            "bank_months_count__missing",
            raw_features,
        )
        == "bank_months_count"
    )
    assert (
        raw_feature_for_transformed_name("device_os_windows", raw_features)
        == "device_os"
    )


def test_grouping_and_reason_selection_are_directional() -> None:
    grouped = aggregate_transformed_contributions(
        (
            "income",
            "device_os_windows",
            "device_os_linux",
            "bank_months_count__missing",
        ),
        np.asarray([0.2, 0.35, -0.05, -0.4]),
        ("income", "device_os", "bank_months_count"),
    )
    reasons = reason_contributions(grouped, maximum_reasons=3)

    assert grouped["device_os"] == 0.30
    assert {reason.direction for reason in reasons} == {
        "increases_risk",
        "decreases_risk",
    }
    assert all(reason.code.startswith("SHAP_") for reason in reasons)
    assert all(0.0 <= reason.importance_share <= 1.0 for reason in reasons)


def test_native_tree_shap_explains_fitted_pipeline_without_refitting() -> None:
    frame = pd.DataFrame(
        {
            "income": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "device_os": [
                "linux",
                "linux",
                "windows",
                "linux",
                "windows",
                "windows",
                "other",
                "windows",
            ],
            "bank_months_count": [-1, 2, 3, 5, 10, 15, 20, 30],
        }
    )
    target = np.asarray([1, 1, 1, 0, 0, 0, 0, 0])
    preprocessor = build_preprocessor(
        frame,
        categorical_features=("device_os",),
        semantic_missing_rules={
            "bank_months_count": {"operator": "equals", "value": -1}
        },
        add_missing_indicators=True,
        scale_numeric=False,
    )
    pipeline = Pipeline(
        steps=[
            ("feature_selection", DataFrameColumnSelector(tuple(frame.columns))),
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_estimators=8,
                    max_depth=2,
                    min_child_weight=0.0,
                    learning_rate=0.2,
                    reg_lambda=0.0,
                    n_jobs=1,
                    random_state=42,
                    verbosity=0,
                ),
            ),
        ]
    )
    pipeline.fit(frame, target)
    explainer = XGBoostTreeShapExplainer(
        pipeline,
        maximum_reasons=3,
    )

    explanations = explainer.explain(frame.iloc[:2])

    assert len(explanations) == 2
    assert all(item.method == TREE_SHAP_METHOD for item in explanations)
    assert all(item.scope == TREE_SHAP_SCOPE for item in explanations)
    assert all(item.reason_codes for item in explanations)
