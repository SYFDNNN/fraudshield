"""Local XGBoost TreeSHAP explanations for prediction-time review support.

The implementation deliberately explains the frozen base XGBoost margin. It
does not fit an explainer, read labels, or claim that a feature caused fraud.
The later sigmoid calibrator changes probability scale but preserves ranking,
so its output is not presented as the direct target of the SHAP decomposition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraudshield.features import get_preprocessed_feature_names

TREE_SHAP_METHOD = "xgboost_native_tree_shap"
TREE_SHAP_SCOPE = "base_xgboost_raw_margin_before_probability_calibration"

FEATURE_LABELS: dict[str, str] = {
    "income": "Pendapatan pada skala dataset",
    "name_email_similarity": "Kemiripan nama dan email",
    "prev_address_months_count": "Lama di alamat sebelumnya",
    "current_address_months_count": "Lama di alamat saat ini",
    "customer_age": "Usia pemohon",
    "intended_balcon_amount": "Rencana saldo awal",
    "payment_type": "Kategori metode pembayaran",
    "zip_count_4w": "Aktivitas area/ZIP dalam 4 minggu",
    "velocity_6h": "Laju aktivitas agregat 6 jam",
    "velocity_24h": "Laju aktivitas agregat 24 jam",
    "velocity_4w": "Laju aktivitas agregat 4 minggu",
    "bank_branch_count_8w": "Aktivitas cabang bank dalam 8 minggu",
    "date_of_birth_distinct_emails_4w": (
        "Email berbeda terkait tanggal lahir dalam 4 minggu"
    ),
    "employment_status": "Kategori status pekerjaan",
    "email_is_free": "Penggunaan penyedia email gratis",
    "housing_status": "Kategori status tempat tinggal",
    "phone_home_valid": "Validasi format telepon rumah",
    "phone_mobile_valid": "Validasi format nomor ponsel",
    "bank_months_count": "Lama hubungan dengan bank",
    "has_other_cards": "Kepemilikan kartu lain",
    "proposed_credit_limit": "Limit kredit yang diajukan",
    "foreign_request": "Sinyal permintaan lintas negara",
    "source": "Kanal pengajuan",
    "session_length_in_minutes": "Durasi sesi",
    "device_os": "Sistem operasi perangkat",
    "keep_alive_session": "Status keep-alive sesi",
    "device_distinct_emails_8w": (
        "Email berbeda pada perangkat dalam 8 minggu"
    ),
}


class ExplainabilityError(RuntimeError):
    """Raised when a frozen model cannot produce a trustworthy explanation."""


@dataclass(frozen=True, slots=True)
class ReasonContribution:
    """One grouped local contribution suitable for a public reason code."""

    code: str
    feature: str
    feature_label: str
    direction: str
    shap_value: float
    importance_share: float


@dataclass(frozen=True, slots=True)
class LocalExplanation:
    """One locally additive explanation of a base-model raw margin."""

    method: str
    scope: str
    base_value: float
    raw_margin: float
    reason_codes: tuple[ReasonContribution, ...]


def raw_feature_for_transformed_name(
    transformed_name: str,
    raw_feature_names: tuple[str, ...],
) -> str:
    """Map numeric, missing-indicator, or one-hot names to one raw feature."""
    if transformed_name in raw_feature_names:
        return transformed_name

    if transformed_name.endswith("__missing"):
        candidate = transformed_name.removesuffix("__missing")

        if candidate in raw_feature_names:
            return candidate

    prefix_matches = [
        feature
        for feature in raw_feature_names
        if transformed_name.startswith(f"{feature}_")
    ]

    if prefix_matches:
        return max(prefix_matches, key=len)

    raise ExplainabilityError(
        "Preprocessed feature cannot be mapped to the public raw schema: "
        f"{transformed_name!r}."
    )


def aggregate_transformed_contributions(
    transformed_feature_names: tuple[str, ...],
    contributions: np.ndarray,
    raw_feature_names: tuple[str, ...],
) -> dict[str, float]:
    """Sum one-hot and missing-indicator SHAP values into raw features."""
    values = np.asarray(contributions, dtype=float)

    if values.ndim != 1 or len(values) != len(transformed_feature_names):
        raise ValueError(
            "Contributions must be one-dimensional and match feature names."
        )

    if not np.isfinite(values).all():
        raise ValueError("Contributions must contain only finite values.")

    grouped = {feature: 0.0 for feature in raw_feature_names}

    for transformed_name, contribution in zip(
        transformed_feature_names,
        values,
        strict=True,
    ):
        raw_feature = raw_feature_for_transformed_name(
            transformed_name,
            raw_feature_names,
        )
        grouped[raw_feature] += float(contribution)

    return grouped


def reason_contributions(
    grouped_contributions: dict[str, float],
    *,
    maximum_reasons: int,
    minimum_absolute_contribution: float = 1e-8,
) -> tuple[ReasonContribution, ...]:
    """Select stable, directional local reasons without inventing semantics."""
    if not 1 <= maximum_reasons <= 10:
        raise ValueError("maximum_reasons must lie between 1 and 10.")

    eligible = [
        (feature, float(value))
        for feature, value in grouped_contributions.items()
        if abs(float(value)) >= minimum_absolute_contribution
    ]
    eligible.sort(key=lambda item: (-abs(item[1]), item[0]))
    selected = eligible[:maximum_reasons]
    total_absolute = sum(abs(value) for _, value in eligible)
    reasons: list[ReasonContribution] = []

    for feature, value in selected:
        direction = "increases_risk" if value > 0 else "decreases_risk"
        direction_code = "UP" if value > 0 else "DOWN"
        normalized_feature = re.sub(r"[^A-Z0-9]+", "_", feature.upper()).strip(
            "_"
        )
        reasons.append(
            ReasonContribution(
                code=f"SHAP_{direction_code}_{normalized_feature}",
                feature=feature,
                feature_label=FEATURE_LABELS.get(
                    feature,
                    feature.replace("_", " ").title(),
                ),
                direction=direction,
                shap_value=value,
                importance_share=(
                    abs(value) / total_absolute if total_absolute else 0.0
                ),
            )
        )

    return tuple(reasons)


class XGBoostTreeShapExplainer:
    """Explain a fitted preprocessing-plus-XGBoost pipeline natively."""

    def __init__(
        self,
        pipeline: Any,
        *,
        maximum_reasons: int = 5,
    ) -> None:
        if not 1 <= maximum_reasons <= 10:
            raise ValueError("maximum_reasons must lie between 1 and 10.")

        named_steps = getattr(pipeline, "named_steps", {})
        selector = named_steps.get("feature_selection")
        preprocessor = named_steps.get("preprocessor")
        classifier = named_steps.get("classifier")

        if selector is None or preprocessor is None or classifier is None:
            raise ExplainabilityError(
                "Explainability requires feature_selection, preprocessor, "
                "and classifier pipeline steps."
            )

        raw_feature_names = tuple(
            getattr(selector, "selected_columns_", ())
        )

        if not raw_feature_names:
            raise ExplainabilityError(
                "Feature selector does not expose its fitted raw schema."
            )

        if not callable(getattr(classifier, "get_booster", None)):
            raise ExplainabilityError(
                "The selected classifier does not expose an XGBoost booster."
            )

        try:
            transformed_names = tuple(
                str(name)
                for name in get_preprocessed_feature_names(preprocessor)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ExplainabilityError(
                "Fitted preprocessor does not expose stable feature names."
            ) from error

        for transformed_name in transformed_names:
            raw_feature_for_transformed_name(
                transformed_name,
                raw_feature_names,
            )

        self.pipeline = pipeline
        self.selector = selector
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.raw_feature_names = raw_feature_names
        self.transformed_feature_names = transformed_names
        self.maximum_reasons = maximum_reasons

    def explain(self, features: pd.DataFrame) -> list[LocalExplanation]:
        """Return grouped exact TreeSHAP contributions for every input row."""
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise TypeError("features must be a non-empty pandas DataFrame.")

        try:
            from xgboost import DMatrix

            selected = self.selector.transform(features)
            transformed = self.preprocessor.transform(selected)
            matrix = DMatrix(transformed)
            booster = self.classifier.get_booster()
            shap_matrix = np.asarray(
                booster.predict(
                    matrix,
                    pred_contribs=True,
                    validate_features=False,
                ),
                dtype=float,
            )
            raw_margin = np.asarray(
                booster.predict(
                    matrix,
                    output_margin=True,
                    validate_features=False,
                ),
                dtype=float,
            ).reshape(-1)
        except Exception as error:
            raise ExplainabilityError(
                "XGBoost TreeSHAP failed for this prediction request."
            ) from error

        expected_width = len(self.transformed_feature_names) + 1

        if shap_matrix.ndim == 3 and shap_matrix.shape[1] == 1:
            shap_matrix = shap_matrix[:, 0, :]

        if (
            shap_matrix.ndim != 2
            or shap_matrix.shape != (len(features), expected_width)
        ):
            raise ExplainabilityError(
                "TreeSHAP returned an unexpected contribution shape."
            )

        if len(raw_margin) != len(features):
            raise ExplainabilityError("Raw-margin output has an invalid shape.")

        reconstructed_margin = shap_matrix.sum(axis=1)

        if not np.allclose(
            reconstructed_margin,
            raw_margin,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ExplainabilityError(
                "TreeSHAP local-accuracy validation failed."
            )

        explanations: list[LocalExplanation] = []

        for row_index in range(len(features)):
            grouped = aggregate_transformed_contributions(
                self.transformed_feature_names,
                shap_matrix[row_index, :-1],
                self.raw_feature_names,
            )
            explanations.append(
                LocalExplanation(
                    method=TREE_SHAP_METHOD,
                    scope=TREE_SHAP_SCOPE,
                    base_value=float(shap_matrix[row_index, -1]),
                    raw_margin=float(raw_margin[row_index]),
                    reason_codes=reason_contributions(
                        grouped,
                        maximum_reasons=self.maximum_reasons,
                    ),
                )
            )

        return explanations
