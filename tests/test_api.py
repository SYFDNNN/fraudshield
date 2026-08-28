"""Contract tests for the Phase 7 FastAPI service."""

from __future__ import annotations

import logging
from math import ceil
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from fraudshield.predict import (
    CONTRACT_VERSION,
    ApplicationInput,
    InferenceArtifactIdentity,
    InferenceContractError,
    InferenceRuntime,
    PredictionOutput,
)


def valid_application(
    application_id: str = "application-001",
) -> dict[str, object]:
    """Return one request containing all prediction-time fields."""
    return {
        "application_id": application_id,
        "income": 0.6,
        "name_email_similarity": 0.75,
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
        "foreign_request": 0,
        "source": "INTERNET",
        "session_length_in_minutes": 8.0,
        "device_os": "windows",
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
    }


class StubRuntime(InferenceRuntime):
    """Small runtime double that still satisfies API type guards."""

    def __init__(self, *, maximum_batch_size: int = 100) -> None:
        self.identity = InferenceArtifactIdentity(
            model_sha256="a" * 64,
            phase6_policy_sha256="b" * 64,
            model_version="fraudshield-api-test-aaaaaaaaaaaa",
            threshold_policy_version="capacity-api-test-bbbbbbbbbbbb",
            calibrator_name="sigmoid",
        )
        self.maximum_batch_size = maximum_batch_size

    @property
    def requested_review_rate(self) -> float:
        return 0.05

    def contract_document(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "model_version": self.identity.model_version,
            "threshold_policy_version": (
                self.identity.threshold_policy_version
            ),
            "maximum_batch_size": self.maximum_batch_size,
            "requested_review_rate": self.requested_review_rate,
            "automated_rejection_allowed": False,
            "input_payload_logged": False,
        }

    def _output(
        self,
        application: ApplicationInput,
        *,
        rank: int | None,
        selected: bool | None,
    ) -> PredictionOutput:
        return PredictionOutput(
            application_id=application.application_id,
            fraud_probability=0.12,
            risk_band="menengah",
            fixed_threshold_review=True,
            exact_capacity_review=selected,
            review_rank=rank,
            review_policy=(
                "exact_batch_capacity"
                if rank is not None
                else "fixed_threshold_signal_only"
            ),
            model_version=self.identity.model_version,
            threshold_policy_version=(
                self.identity.threshold_policy_version
            ),
            calibrator=self.identity.calibrator_name,
        )

    def score_single(self, application: ApplicationInput) -> PredictionOutput:
        return self._output(application, rank=None, selected=None)

    def score_batch(
        self,
        applications: list[ApplicationInput],
    ) -> list[PredictionOutput]:
        if len(applications) > self.maximum_batch_size:
            raise InferenceContractError("Batch exceeds maximum_batch_size.")

        if len({item.application_id for item in applications}) != len(
            applications
        ):
            raise InferenceContractError("application_id values must be unique.")

        review_count = ceil(len(applications) * self.requested_review_rate)
        return [
            self._output(
                application,
                rank=index,
                selected=index <= review_count,
            )
            for index, application in enumerate(applications, start=1)
        ]


@pytest.fixture
def client() -> TestClient:
    """Start the application with a preloaded deterministic runtime."""
    with TestClient(create_app(runtime=StubRuntime())) as test_client:
        yield test_client


def test_health_and_contract_expose_active_versions(client: TestClient) -> None:
    """Readiness should prove that the frozen runtime is loaded."""
    health = client.get("/health/ready")
    contract = client.get("/v1/contract")

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert health.json()["calibrator"] == "sigmoid"
    assert contract.status_code == 200
    assert contract.json()["contract_version"] == CONTRACT_VERSION
    assert contract.json()["automated_rejection_allowed"] is False


def test_single_prediction_never_claims_exact_capacity(
    client: TestClient,
) -> None:
    """The single endpoint may expose a signal but not a queue rank."""
    response = client.post(
        "/v1/predict",
        json=valid_application(),
        headers={"X-Request-ID": "request-single-001"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-single-001"
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.json()
    assert payload["request_id"] == "request-single-001"
    assert payload["prediction"]["exact_capacity_review"] is None
    assert payload["prediction"]["review_rank"] is None
    assert payload["prediction"]["automated_rejection_allowed"] is False


def test_schema_rejects_target_and_unknown_fields(client: TestClient) -> None:
    """HTTP validation must reject labels and undeclared request fields."""
    payload = valid_application()
    payload["fraud_bool"] = 1
    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422
    assert "fraud_bool" in response.text


def test_batch_requires_complete_window_and_returns_exact_count(
    client: TestClient,
) -> None:
    """A complete 20-row batch should select exactly one row at 5%."""
    applications = [
        valid_application(f"application-{index:03d}")
        for index in range(20)
    ]
    incomplete = client.post(
        "/v1/predict/batch",
        json={
            "batch_id": "window-001",
            "complete_decision_window": False,
            "applications": applications,
        },
    )
    response = client.post(
        "/v1/predict/batch",
        json={
            "batch_id": "window-001",
            "complete_decision_window": True,
            "applications": applications,
        },
    )

    assert incomplete.status_code == 422
    assert response.status_code == 200
    payload = response.json()
    assert payload["batch_id"] == "window-001"
    assert payload["row_count"] == 20
    assert payload["review_count"] == 1
    assert sum(
        item["exact_capacity_review"] is True
        for item in payload["predictions"]
    ) == 1


def test_api_rejects_duplicate_batch_identifiers(client: TestClient) -> None:
    """Duplicate application IDs make deterministic ranking ambiguous."""
    application = valid_application()
    response = client.post(
        "/v1/predict/batch",
        json={
            "batch_id": "window-duplicate",
            "complete_decision_window": True,
            "applications": [application, application],
        },
    )

    assert response.status_code == 422
    assert "unique" in response.json()["detail"]
    assert response.json()["request_id"]


def test_telemetry_is_aggregate_and_payload_free(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operational logging must never retain application-level inputs."""
    secret_identifier = "do-not-log-this-application"

    with caplog.at_level(logging.INFO, logger="fraudshield.api"):
        response = client.post(
            "/v1/predict",
            json=valid_application(secret_identifier),
        )

    metrics = client.get("/v1/metrics").json()
    assert response.status_code == 200
    assert metrics["requests_total"] >= 1
    assert metrics["scored_rows_total"] >= 1
    assert secret_identifier not in caplog.text
    assert '"payload_logged": false' in caplog.text


def test_openapi_lists_both_prediction_modes(client: TestClient) -> None:
    """The generated schema is the integration source of truth."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "/v1/predict" in paths
    assert "/v1/predict/batch" in paths
