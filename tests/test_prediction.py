"""Tests for the artifact-locked Phase 7 inference runtime."""

import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

import fraudshield.predict as prediction_module
from fraudshield.calibrate import (
    CalibratedFraudModel,
    IdentityProbabilityCalibrator,
)
from fraudshield.config import load_config
from fraudshield.predict import (
    MODEL_FEATURE_NAMES,
    ApplicationInput,
    InferenceArtifactIdentity,
    InferenceContractError,
    InferenceRuntime,
)
from fraudshield.thresholds import (
    CapacityThresholdPolicy,
    RiskBandBoundary,
    RiskBandPolicy,
)


class FakeFeatureSelector:
    """Expose the exact fitted column contract expected by the runtime."""

    selected_columns_ = MODEL_FEATURE_NAMES


class FakeClassifier:
    """Expose class order used by positive_class_probability."""

    classes_ = np.asarray([0, 1], dtype=np.int8)


class FakeProbabilityPipeline:
    """Small picklable probability model for runtime contract tests."""

    def __init__(self) -> None:
        self.named_steps = {
            "feature_selection": FakeFeatureSelector(),
            "classifier": FakeClassifier(),
        }

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = np.clip(
            0.02
            + (0.75 * (1.0 - features["name_email_similarity"].to_numpy()))
            + (0.10 * features["foreign_request"].to_numpy()),
            0.001,
            0.999,
        )
        return np.column_stack([1.0 - probability, probability])


def valid_application(
    application_id: str = "application-001",
    *,
    similarity: float = 0.75,
    foreign_request: int = 0,
) -> dict[str, object]:
    """Return one strict, prediction-time-only request payload."""
    return {
        "application_id": application_id,
        "income": 0.6,
        "name_email_similarity": similarity,
        "prev_address_months_count": -1,
        "current_address_months_count": 24,
        "customer_age": 40,
        "intended_balcon_amount": 20.0,
        "payment_type": "AA",
        "zip_count_4w": 1200,
        "velocity_6h": 4500.0,
        "velocity_24h": 5200.0,
        "velocity_4w": 5000.0,
        "bank_branch_count_8w": 10,
        "date_of_birth_distinct_emails_4w": 2,
        "employment_status": "CA",
        "email_is_free": 1,
        "housing_status": "BA",
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "bank_months_count": 12,
        "has_other_cards": 0,
        "proposed_credit_limit": 1000.0,
        "foreign_request": foreign_request,
        "source": "INTERNET",
        "session_length_in_minutes": 8.0,
        "device_os": "windows",
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
    }


def frozen_model() -> CalibratedFraudModel:
    """Return a deterministic calibrated-model bundle without fitting."""
    calibrator = IdentityProbabilityCalibrator()
    calibrator.is_fitted_ = True
    capacity_policy = CapacityThresholdPolicy(
        source_split="validation",
        requested_review_rate=0.05,
        row_count=100,
        target_review_count=5,
        score_threshold=0.40,
        strictly_above_threshold_count=5,
        boundary_tie_count=1,
        boundary_review_slots=1,
        observed_at_or_above_threshold_count=5,
        observed_at_or_above_threshold_rate=0.05,
        expected_captured_fraud_count=3.0,
        expected_precision_at_capacity=0.60,
        expected_recall_at_capacity=0.50,
        tie_break_required=False,
        tie_break_policy=(
            "calibrated_score_desc_then_raw_score_desc_then_"
            "stable_application_key"
        ),
    )
    risk_policy = RiskBandPolicy(
        source_split="validation",
        boundaries=(
            RiskBandBoundary(
                label="sangat_tinggi",
                cumulative_review_rate=0.01,
                score_threshold=0.60,
                observed_at_or_above_count=1,
                observed_at_or_above_rate=0.01,
                boundary_tie_count=1,
            ),
            RiskBandBoundary(
                label="tinggi",
                cumulative_review_rate=0.05,
                score_threshold=0.40,
                observed_at_or_above_count=5,
                observed_at_or_above_rate=0.05,
                boundary_tie_count=1,
            ),
            RiskBandBoundary(
                label="menengah",
                cumulative_review_rate=0.10,
                score_threshold=0.20,
                observed_at_or_above_count=10,
                observed_at_or_above_rate=0.10,
                boundary_tie_count=1,
            ),
        ),
        default_label="rendah",
        tie_break_policy=(
            "calibrated_score_desc_then_raw_score_desc_then_"
            "stable_application_key"
        ),
    )
    return CalibratedFraudModel(
        base_model=FakeProbabilityPipeline(),
        calibrator=calibrator,
        calibrator_name="sigmoid",
        capacity_policy=capacity_policy,
        risk_band_policy=risk_policy,
    )


def runtime(*, maximum_batch_size: int = 100) -> InferenceRuntime:
    """Build a runtime around the deterministic test bundle."""
    return InferenceRuntime(
        model=frozen_model(),
        identity=InferenceArtifactIdentity(
            model_sha256="a" * 64,
            phase6_policy_sha256="b" * 64,
            model_version="fraudshield-test-aaaaaaaaaaaa",
            threshold_policy_version="capacity-test-bbbbbbbbbbbb",
            calibrator_name="sigmoid",
        ),
        maximum_batch_size=maximum_batch_size,
    )


def test_application_schema_rejects_leakage_and_unknown_fields() -> None:
    """Target, temporal, and other undeclared fields must fail closed."""
    payload = valid_application()
    payload["fraud_bool"] = 0

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationInput.model_validate(payload)


def test_application_schema_is_strict_and_requires_every_feature() -> None:
    """The API must not invent absent fields or coerce numeric strings."""
    missing = valid_application()
    missing.pop("velocity_4w")

    with pytest.raises(ValidationError, match="Field required"):
        ApplicationInput.model_validate(missing)

    wrong_type = valid_application()
    wrong_type["income"] = "0.6"

    with pytest.raises(ValidationError, match="valid number"):
        ApplicationInput.model_validate(wrong_type)


def test_application_schema_accepts_well_formed_unknown_category() -> None:
    """The fitted encoder, rather than the API, handles unseen codes."""
    payload = valid_application()
    payload["payment_type"] = "NEW_PAYMENT_CODE"

    application = ApplicationInput.model_validate(payload)

    assert application.payment_type == "NEW_PAYMENT_CODE"


def test_single_scoring_does_not_claim_exact_capacity() -> None:
    """One row cannot receive a pretend exact 5% queue decision."""
    prediction = runtime().score_single(
        ApplicationInput.model_validate(valid_application())
    )

    assert 0.0 <= prediction.fraud_probability <= 1.0
    assert prediction.exact_capacity_review is None
    assert prediction.review_rank is None
    assert prediction.review_policy == "fixed_threshold_signal_only"
    assert prediction.automated_rejection_allowed is False
    assert prediction.reason_codes == []


def test_batch_scoring_assigns_exact_capacity_and_unique_ranks() -> None:
    """A complete batch should receive ceil(5%) deterministic review flags."""
    applications = [
        ApplicationInput.model_validate(
            valid_application(
                f"application-{index:03d}",
                similarity=0.05 + (index * 0.04),
                foreign_request=int(index % 7 == 0),
            )
        )
        for index in range(20)
    ]
    predictions = runtime().score_batch(applications)

    assert sum(item.exact_capacity_review is True for item in predictions) == 1
    assert sorted(item.review_rank for item in predictions) == list(range(1, 21))
    assert all(item.review_policy == "exact_batch_capacity" for item in predictions)


def test_batch_scoring_is_order_invariant_for_stable_ids() -> None:
    """Stable application IDs must make ties deterministic across row order."""
    applications = [
        ApplicationInput.model_validate(
            valid_application(f"application-{index:03d}", similarity=0.5)
        )
        for index in range(20)
    ]
    scorer = runtime()
    forward = scorer.score_batch(applications)
    reverse = scorer.score_batch(list(reversed(applications)))
    forward_map = {
        item.application_id: (item.review_rank, item.exact_capacity_review)
        for item in forward
    }
    reverse_map = {
        item.application_id: (item.review_rank, item.exact_capacity_review)
        for item in reverse
    }

    assert forward_map == reverse_map


def test_batch_rejects_duplicate_ids_and_oversized_payloads() -> None:
    """Capacity ranking requires unique keys and a bounded request size."""
    application = ApplicationInput.model_validate(valid_application())

    with pytest.raises(InferenceContractError, match="unique"):
        runtime().score_batch([application, application])

    second = ApplicationInput.model_validate(
        valid_application("application-002")
    )

    with pytest.raises(InferenceContractError, match="maximum_batch_size"):
        runtime(maximum_batch_size=1).score_batch([application, second])


def test_contract_documents_human_review_boundaries() -> None:
    """The public contract must make operational limitations explicit."""
    contract = runtime().contract_document()

    assert contract["required_model_features"] == list(MODEL_FEATURE_NAMES)
    assert contract["automated_rejection_allowed"] is False
    assert contract["reason_codes_available"] is False
    assert contract["batch_prediction_policy"] == "exact_batch_capacity"
    assert "complete_decision_window" in contract["batch_precondition"]
    assert contract["unknown_categories"].startswith("accepted")
    assert contract["input_payload_logged"] is False


def test_inference_source_never_fits_or_loads_raw_dataset() -> None:
    """Serving code must be prediction-only and label-independent."""
    source = inspect.getsource(prediction_module)

    assert ".fit(" not in source
    assert "load_base_dataset" not in source
    assert "data/raw" not in source


def _write_runtime_artifacts(tmp_path: Path) -> Path:
    """Write one complete synthetic Phase 5/6 deployment gate."""
    config = load_config("configs/base.yaml")
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    model_path = artifact_directory / "model.joblib"
    phase5_path = artifact_directory / "phase5_metadata.json"
    phase6_path = artifact_directory / "phase6_metadata.json"
    completion_path = artifact_directory / "completion.json"
    joblib.dump(frozen_model(), model_path)
    model_hash = prediction_module._sha256_file(model_path)
    policy_hash = "c" * 64
    phase5_metadata = {
        "test_evaluated": False,
        "test_features_exposed": False,
        "business_threshold_selected": True,
        "automated_rejection_allowed": False,
        "selected_calibrator_name": "sigmoid",
    }
    phase6_metadata = {
        "test_evaluated": True,
        "test_evaluation_count": 1,
        "base_model_refit_after_test": False,
        "calibrator_refit_after_test": False,
        "threshold_reselected_after_test": False,
        "risk_bands_reselected_after_test": False,
        "automated_rejection_allowed": False,
        "selected_calibrator_name": "sigmoid",
        "locked_policy_sha256": policy_hash,
        "phase5_artifact_sha256": {"model": model_hash},
    }
    phase5_path.write_text(json.dumps(phase5_metadata), encoding="utf-8")
    phase5_hashes = {
        "model": model_hash,
        "metadata": prediction_module._sha256_file(phase5_path),
    }
    phase6_metadata["phase5_artifact_sha256"] = phase5_hashes
    phase6_path.write_text(json.dumps(phase6_metadata), encoding="utf-8")
    completion = {
        "status": "completed",
        "test_evaluation_count": 1,
        "locked_policy_sha256": policy_hash,
        "phase5_artifact_sha256": phase5_hashes,
        "result_artifact_sha256": {
            "metadata": prediction_module._sha256_file(phase6_path)
        },
    }
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    config["inference"].update(
        {
            "model_artifact": str(model_path),
            "phase5_metadata_artifact": str(phase5_path),
            "phase6_metadata_artifact": str(phase6_path),
            "phase6_completion_artifact": str(completion_path),
            "expected_model_sha256": model_hash,
            "expected_phase6_policy_sha256": policy_hash,
            "maximum_batch_size": 100,
        }
    )
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    config_path = config_directory / "base.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_runtime_loads_only_evaluated_hash_locked_model(tmp_path: Path) -> None:
    """Deployment startup should verify both model and final evidence."""
    config_path = _write_runtime_artifacts(tmp_path)
    loaded = InferenceRuntime.load(config_path)

    assert loaded.identity.calibrator_name == "sigmoid"
    assert loaded.identity.model_sha256[:12] in loaded.identity.model_version
    assert loaded.maximum_batch_size == 100


def test_runtime_rejects_model_tampering(tmp_path: Path) -> None:
    """Any post-evaluation model change must block service startup."""
    config_path = _write_runtime_artifacts(tmp_path)
    config = load_config(config_path)
    model_path = Path(config["inference"]["model_artifact"])

    with model_path.open("ab") as model_file:
        model_file.write(b"tampered")

    with pytest.raises(RuntimeError, match="deployment lock"):
        InferenceRuntime.load(config_path)


def test_runtime_rejects_evidence_tampering(tmp_path: Path) -> None:
    """Serving must reject metadata changed after the final evaluation."""
    config_path = _write_runtime_artifacts(tmp_path)
    config = load_config(config_path)
    metadata_path = Path(
        config["inference"]["phase5_metadata_artifact"]
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["selected_calibrator_name"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata differs"):
        InferenceRuntime.load(config_path)
