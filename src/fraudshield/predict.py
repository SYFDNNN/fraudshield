"""Strict, artifact-locked inference contract for FraudShield Phase 7."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudshield.calibrate import CalibratedFraudModel
from fraudshield.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
)
from fraudshield.thresholds import (
    assign_risk_bands,
    capacity_review_flags,
    threshold_review_flags,
)

CONTRACT_VERSION = "1.0.0"
MAXIMUM_CONTRACT_BATCH_SIZE = 5_000
MODEL_FEATURE_NAMES: tuple[str, ...] = (
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
)
EXCLUDED_INFERENCE_FIELDS: tuple[str, ...] = (
    "fraud_bool",
    "month",
    "device_fraud_count",
    "days_since_request",
    "credit_risk_score",
)

StrictBinary = Annotated[int, Field(strict=True, ge=0, le=1)]
StrictFiniteFloat = Annotated[float, Field(strict=True)]
CategoryCode = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]


class ApplicationInput(BaseModel):
    """One prediction-time application with no label or split fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    application_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    income: Annotated[float, Field(strict=True, ge=0.1, le=0.9)]
    name_email_similarity: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0),
    ]
    prev_address_months_count: Annotated[int, Field(strict=True, ge=-1)]
    current_address_months_count: Annotated[int, Field(strict=True, ge=-1)]
    customer_age: Annotated[int, Field(strict=True, ge=10, le=90)]
    intended_balcon_amount: StrictFiniteFloat
    payment_type: CategoryCode
    zip_count_4w: Annotated[int, Field(strict=True, ge=1)]
    velocity_6h: StrictFiniteFloat
    velocity_24h: StrictFiniteFloat
    velocity_4w: StrictFiniteFloat
    bank_branch_count_8w: Annotated[int, Field(strict=True, ge=0)]
    date_of_birth_distinct_emails_4w: Annotated[
        int,
        Field(strict=True, ge=0),
    ]
    employment_status: CategoryCode
    email_is_free: StrictBinary
    housing_status: CategoryCode
    phone_home_valid: StrictBinary
    phone_mobile_valid: StrictBinary
    bank_months_count: Annotated[int, Field(strict=True, ge=-1, le=32)]
    has_other_cards: StrictBinary
    proposed_credit_limit: Annotated[
        float,
        Field(strict=True, ge=190.0, le=2100.0),
    ]
    foreign_request: StrictBinary
    source: CategoryCode
    session_length_in_minutes: Annotated[float, Field(strict=True, ge=-1.0)]
    device_os: CategoryCode
    keep_alive_session: StrictBinary
    device_distinct_emails_8w: Annotated[
        int,
        Field(strict=True, ge=-1, le=2),
    ]

    @field_validator("application_id")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        """Keep the stable tie-break key safe for logs and transport."""
        if any(ord(character) < 32 for character in value):
            raise ValueError("application_id contains control characters.")

        return value


class BatchPredictionRequest(BaseModel):
    """A bounded batch used for exact-capacity queue assignment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    batch_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    complete_decision_window: Literal[True]
    applications: list[ApplicationInput] = Field(
        min_length=1,
        max_length=MAXIMUM_CONTRACT_BATCH_SIZE,
    )


class PredictionOutput(BaseModel):
    """Stable public response for one application."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_band: str
    fixed_threshold_review: bool
    exact_capacity_review: bool | None
    review_rank: int | None = Field(default=None, ge=1)
    review_policy: Literal[
        "fixed_threshold_signal_only",
        "exact_batch_capacity",
    ]
    model_version: str
    threshold_policy_version: str
    calibrator: str
    automated_rejection_allowed: Literal[False] = False
    explanation_status: Literal["not_available_in_phase7"] = (
        "not_available_in_phase7"
    )
    reason_codes: list[str] = Field(default_factory=list)


class SinglePredictionResponse(BaseModel):
    """API envelope for one independently scored application."""

    request_id: str
    contract_version: str
    prediction: PredictionOutput


class BatchPredictionResponse(BaseModel):
    """API envelope for one exact-capacity scoring batch."""

    request_id: str
    batch_id: str
    contract_version: str
    row_count: int = Field(ge=1)
    requested_review_rate: float = Field(gt=0.0, le=1.0)
    review_count: int = Field(ge=1)
    predictions: list[PredictionOutput]


@dataclass(frozen=True, slots=True)
class InferenceArtifactIdentity:
    """Version identity derived from evaluated immutable artifacts."""

    model_sha256: str
    phase6_policy_sha256: str
    model_version: str
    threshold_policy_version: str
    calibrator_name: str


class InferenceContractError(ValueError):
    """Raised when a request cannot satisfy the production contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain a mapping: {path}")

    return payload


def _artifact_path(
    config: dict[str, Any],
    key: str,
    *,
    config_path: str | Path,
) -> Path:
    return resolve_project_path(
        config["inference"][key],
        config_path=config_path,
    )


class InferenceRuntime:
    """Thread-safe scorer that never fits, recalibrates, or reads labels."""

    def __init__(
        self,
        *,
        model: CalibratedFraudModel,
        identity: InferenceArtifactIdentity,
        maximum_batch_size: int,
    ) -> None:
        if not isinstance(model, CalibratedFraudModel):
            raise TypeError("Inference model must be a CalibratedFraudModel.")

        if maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be positive.")

        if maximum_batch_size > MAXIMUM_CONTRACT_BATCH_SIZE:
            raise ValueError(
                "maximum_batch_size exceeds the public contract limit."
            )

        try:
            selected_columns = tuple(
                model.base_model.named_steps[
                    "feature_selection"
                ].selected_columns_
            )
        except (AttributeError, KeyError) as error:
            raise TypeError(
                "Frozen model does not expose its fitted feature selector."
            ) from error

        if selected_columns != MODEL_FEATURE_NAMES:
            raise ValueError(
                "Frozen model feature schema differs from API contract. "
                f"Model={selected_columns}; contract={MODEL_FEATURE_NAMES}."
            )

        if model.calibrator_name != identity.calibrator_name:
            raise ValueError("Model calibrator differs from artifact identity.")

        self.model = model
        self.identity = identity
        self.maximum_batch_size = int(maximum_batch_size)
        self._prediction_lock = threading.RLock()

    @classmethod
    def load(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> InferenceRuntime:
        """Load only the exact model that passed the Phase 6 final gate."""
        config = load_config(config_path)
        inference_config = config["inference"]

        if inference_config["contract_version"] != CONTRACT_VERSION:
            raise ValueError("Configured inference contract version is unsupported.")

        if inference_config["automated_rejection_allowed"] is not False:
            raise ValueError("Inference must forbid automated rejection.")

        if config["review_policy"]["automated_rejection_allowed"] is not False:
            raise ValueError("Review policy must forbid automated rejection.")

        paths = {
            name: _artifact_path(config, key, config_path=config_path)
            for name, key in {
                "model": "model_artifact",
                "phase5_metadata": "phase5_metadata_artifact",
                "phase6_metadata": "phase6_metadata_artifact",
                "phase6_completion": "phase6_completion_artifact",
            }.items()
        }
        missing = sorted(name for name, path in paths.items() if not path.is_file())

        if missing:
            raise FileNotFoundError(
                "Required inference artifacts are missing: " + ", ".join(missing)
            )

        model_sha256 = _sha256_file(paths["model"])
        expected_model_sha256 = inference_config["expected_model_sha256"]

        if model_sha256 != expected_model_sha256:
            raise RuntimeError("Model artifact hash differs from deployment lock.")

        phase5_metadata = _read_json_mapping(paths["phase5_metadata"])
        phase6_metadata = _read_json_mapping(paths["phase6_metadata"])
        completion = _read_json_mapping(paths["phase6_completion"])
        expected_policy_sha256 = inference_config[
            "expected_phase6_policy_sha256"
        ]

        if completion.get("status") != "completed":
            raise RuntimeError("Phase 6 final evaluation is not completed.")

        if completion.get("test_evaluation_count") != 1:
            raise RuntimeError("Final test evaluation count must equal one.")

        if completion.get("locked_policy_sha256") != expected_policy_sha256:
            raise RuntimeError("Phase 6 policy hash differs from deployment lock.")

        if phase6_metadata.get("locked_policy_sha256") != expected_policy_sha256:
            raise RuntimeError("Phase 6 metadata uses a different policy lock.")

        completion_phase5_hashes = completion.get(
            "phase5_artifact_sha256",
            {},
        )
        metadata_phase5_hashes = phase6_metadata.get(
            "phase5_artifact_sha256",
            {},
        )

        if completion_phase5_hashes != metadata_phase5_hashes:
            raise RuntimeError("Phase 5 evidence hashes differ across Phase 6.")

        for phase5_hashes in (
            completion_phase5_hashes,
            metadata_phase5_hashes,
        ):

            if phase5_hashes.get("model") != model_sha256:
                raise RuntimeError(
                    "Final-evaluation evidence refers to a different model."
                )

            if phase5_hashes.get("metadata") != _sha256_file(
                paths["phase5_metadata"]
            ):
                raise RuntimeError(
                    "Phase 5 metadata differs from final-evaluation evidence."
                )

        phase6_metadata_hash = _sha256_file(paths["phase6_metadata"])
        result_hashes = completion.get("result_artifact_sha256", {})

        if result_hashes.get("metadata") != phase6_metadata_hash:
            raise RuntimeError("Phase 6 metadata hash differs from completion.")

        phase5_guardrails = {
            "test_evaluated": False,
            "test_features_exposed": False,
            "business_threshold_selected": True,
            "automated_rejection_allowed": False,
        }
        phase6_guardrails = {
            "test_evaluated": True,
            "test_evaluation_count": 1,
            "base_model_refit_after_test": False,
            "calibrator_refit_after_test": False,
            "threshold_reselected_after_test": False,
            "risk_bands_reselected_after_test": False,
            "automated_rejection_allowed": False,
        }

        for key, expected in phase5_guardrails.items():
            if phase5_metadata.get(key) != expected:
                raise RuntimeError(f"Phase 5 guardrail failed: {key}")

        for key, expected in phase6_guardrails.items():
            if phase6_metadata.get(key) != expected:
                raise RuntimeError(f"Phase 6 guardrail failed: {key}")

        if phase5_metadata.get("selected_calibrator_name") != (
            phase6_metadata.get("selected_calibrator_name")
        ):
            raise RuntimeError("Calibrator identity differs across evidence.")

        model = joblib.load(paths["model"])

        if not isinstance(model, CalibratedFraudModel):
            raise TypeError("Model artifact is not a CalibratedFraudModel.")

        if not np.isclose(
            model.capacity_policy.requested_review_rate,
            float(config["review_policy"]["capacity_rate"]),
        ):
            raise RuntimeError("Model review capacity differs from configuration.")

        policy_payload = asdict(model.capacity_policy)
        policy_digest = _payload_sha256(policy_payload)[:12]
        configured_policy_version = inference_config[
            "threshold_policy_version"
        ]
        identity = InferenceArtifactIdentity(
            model_sha256=model_sha256,
            phase6_policy_sha256=expected_policy_sha256,
            model_version=(
                f"{inference_config['model_version_prefix']}-{model_sha256[:12]}"
            ),
            threshold_policy_version=(
                f"{configured_policy_version}-{policy_digest}"
            ),
            calibrator_name=model.calibrator_name,
        )
        return cls(
            model=model,
            identity=identity,
            maximum_batch_size=int(inference_config["maximum_batch_size"]),
        )

    @property
    def requested_review_rate(self) -> float:
        """Return the frozen exact-batch capacity rate."""
        return float(self.model.capacity_policy.requested_review_rate)

    def contract_document(self) -> dict[str, Any]:
        """Return the machine-readable public prediction contract."""
        return {
            "contract_version": CONTRACT_VERSION,
            "model_version": self.identity.model_version,
            "threshold_policy_version": (
                self.identity.threshold_policy_version
            ),
            "required_model_features": list(MODEL_FEATURE_NAMES),
            "excluded_fields": list(EXCLUDED_INFERENCE_FIELDS),
            "maximum_batch_size": self.maximum_batch_size,
            "requested_review_rate": self.requested_review_rate,
            "single_prediction_policy": "fixed_threshold_signal_only",
            "batch_prediction_policy": "exact_batch_capacity",
            "single_prediction_limitation": (
                "An exact 5% queue decision requires a complete batch."
            ),
            "batch_precondition": (
                "complete_decision_window must be true; applications must "
                "represent the complete immutable operational queue window."
            ),
            "unknown_categories": "accepted_and_ignored_by_fitted_encoder",
            "semantic_missing_values": {
                "prev_address_months_count": -1,
                "current_address_months_count": -1,
                "bank_months_count": -1,
                "session_length_in_minutes": -1,
                "device_distinct_emails_8w": -1,
                "intended_balcon_amount": "any_negative_value",
            },
            "reason_codes_available": False,
            "automated_rejection_allowed": False,
            "input_payload_logged": False,
        }

    def _feature_frame(
        self,
        applications: list[ApplicationInput],
    ) -> pd.DataFrame:
        records = [
            {
                feature_name: getattr(application, feature_name)
                for feature_name in MODEL_FEATURE_NAMES
            }
            for application in applications
        ]
        return pd.DataFrame.from_records(records, columns=MODEL_FEATURE_NAMES)

    @staticmethod
    def _validate_probabilities(
        values: Any,
        *,
        expected_rows: int,
        name: str,
    ) -> np.ndarray:
        probabilities = np.asarray(values, dtype=float)

        if probabilities.ndim != 1 or len(probabilities) != expected_rows:
            raise RuntimeError(f"{name} probabilities have an invalid shape.")

        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"{name} probabilities contain non-finite values.")

        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise RuntimeError(f"{name} probabilities lie outside [0, 1].")

        return probabilities

    def _score(
        self,
        applications: list[ApplicationInput],
        *,
        assign_exact_capacity: bool,
    ) -> list[PredictionOutput]:
        if not applications:
            raise InferenceContractError("At least one application is required.")

        if len(applications) > self.maximum_batch_size:
            raise InferenceContractError(
                "Batch exceeds maximum_batch_size="
                f"{self.maximum_batch_size}."
            )

        application_ids = np.asarray(
            [application.application_id for application in applications],
            dtype=str,
        )

        if len(np.unique(application_ids)) != len(application_ids):
            raise InferenceContractError(
                "application_id values must be unique within a batch."
            )

        features = self._feature_frame(applications)

        with self._prediction_lock:
            raw_probability = self._validate_probabilities(
                self.model.raw_probability(features),
                expected_rows=len(features),
                name="Raw",
            )
            calibrated_probability = self._validate_probabilities(
                self.model.calibrator.transform(raw_probability),
                expected_rows=len(features),
                name="Calibrated",
            )

        risk_bands = assign_risk_bands(
            calibrated_probability,
            self.model.risk_band_policy,
        )
        fixed_threshold_flags = threshold_review_flags(
            calibrated_probability,
            self.model.capacity_policy,
        )
        exact_capacity_flags: np.ndarray | None = None
        review_ranks: np.ndarray | None = None

        if assign_exact_capacity:
            exact_capacity_flags = capacity_review_flags(
                calibrated_probability,
                review_rate=self.requested_review_rate,
                secondary_scores=raw_probability,
                stable_keys=application_ids,
            )
            ranking = np.lexsort(
                (application_ids, -raw_probability, -calibrated_probability)
            )
            review_ranks = np.empty(len(applications), dtype=int)
            review_ranks[ranking] = np.arange(1, len(applications) + 1)

        outputs = []

        for index, application in enumerate(applications):
            outputs.append(
                PredictionOutput(
                    application_id=application.application_id,
                    fraud_probability=float(calibrated_probability[index]),
                    risk_band=str(risk_bands[index]),
                    fixed_threshold_review=bool(
                        fixed_threshold_flags[index]
                    ),
                    exact_capacity_review=(
                        bool(exact_capacity_flags[index])
                        if exact_capacity_flags is not None
                        else None
                    ),
                    review_rank=(
                        int(review_ranks[index])
                        if review_ranks is not None
                        else None
                    ),
                    review_policy=(
                        "exact_batch_capacity"
                        if assign_exact_capacity
                        else "fixed_threshold_signal_only"
                    ),
                    model_version=self.identity.model_version,
                    threshold_policy_version=(
                        self.identity.threshold_policy_version
                    ),
                    calibrator=self.identity.calibrator_name,
                    automated_rejection_allowed=False,
                    reason_codes=[],
                )
            )

        return outputs

    def score_single(self, application: ApplicationInput) -> PredictionOutput:
        """Score one row without pretending an exact-capacity rank exists."""
        return self._score(
            [application],
            assign_exact_capacity=False,
        )[0]

    def score_batch(
        self,
        applications: list[ApplicationInput],
    ) -> list[PredictionOutput]:
        """Score and rank a complete batch under the locked 5% policy."""
        return self._score(
            applications,
            assign_exact_capacity=True,
        )
