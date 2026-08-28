"""FastAPI service for artifact-locked FraudShield inference."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from fraudshield.monitoring import InferenceTelemetry
from fraudshield.predict import (
    CONTRACT_VERSION,
    ApplicationInput,
    BatchPredictionRequest,
    BatchPredictionResponse,
    InferenceContractError,
    InferenceRuntime,
    SinglePredictionResponse,
)

LOGGER = logging.getLogger("fraudshield.api")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID", "")

    if REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied

    return uuid.uuid4().hex


def create_app(
    *,
    runtime: InferenceRuntime | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Create an API whose startup fails closed on invalid artifacts."""
    configured_log_level = os.getenv("FRAUDSHIELD_LOG_LEVEL", "INFO").upper()
    LOGGER.setLevel(getattr(logging, configured_log_level, logging.INFO))
    resolved_config_path = Path(
        config_path
        or os.getenv("FRAUDSHIELD_CONFIG_PATH", "configs/base.yaml")
    )
    telemetry = InferenceTelemetry()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.runtime = runtime or InferenceRuntime.load(
            resolved_config_path
        )
        yield

    application = FastAPI(
        title="FraudShield Inference API",
        description=(
            "Probabilitas fraud terkalibrasi untuk prioritas human review. "
            "Automated rejection tidak diizinkan."
        ),
        version=CONTRACT_VERSION,
        lifespan=lifespan,
    )
    application.state.telemetry = telemetry

    @application.middleware("http")
    async def request_observability(request: Request, call_next):
        request.state.request_id = _request_id(request)
        request.state.scored_rows = 0
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            duration = (time.perf_counter() - started) * 1000.0
            telemetry.record(
                status_code=500,
                scored_rows=int(request.state.scored_rows),
                latency_milliseconds=duration,
            )
            LOGGER.exception(
                json.dumps(
                    {
                        "event": "inference_http_request",
                        "request_id": request.state.request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": 500,
                        "scored_rows": request.state.scored_rows,
                        "duration_ms": round(duration, 3),
                        "payload_logged": False,
                    }
                )
            )
            raise

        duration = (time.perf_counter() - started) * 1000.0
        telemetry.record(
            status_code=status_code,
            scored_rows=int(request.state.scored_rows),
            latency_milliseconds=duration,
        )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        LOGGER.info(
            json.dumps(
                {
                    "event": "inference_http_request",
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "scored_rows": request.state.scored_rows,
                    "duration_ms": round(duration, 3),
                    "payload_logged": False,
                }
            )
        )
        return response

    @application.exception_handler(InferenceContractError)
    async def inference_contract_error(
        request: Request,
        error: InferenceContractError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(error),
                "request_id": request.state.request_id,
            },
        )

    def active_runtime(request: Request) -> InferenceRuntime:
        loaded_runtime = getattr(request.app.state, "runtime", None)

        if not isinstance(loaded_runtime, InferenceRuntime):
            raise HTTPException(status_code=503, detail="Model is not ready.")

        return loaded_runtime

    @application.get("/", tags=["service"])
    async def service_index() -> dict[str, object]:
        return {
            "service": "fraudshield-inference",
            "contract_version": CONTRACT_VERSION,
            "documentation": "/docs",
            "automated_rejection_allowed": False,
        }

    @application.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready", tags=["health"])
    async def readiness(request: Request) -> dict[str, object]:
        loaded_runtime = active_runtime(request)
        return {
            "status": "ready",
            "model_version": loaded_runtime.identity.model_version,
            "threshold_policy_version": (
                loaded_runtime.identity.threshold_policy_version
            ),
            "calibrator": loaded_runtime.identity.calibrator_name,
            "automated_rejection_allowed": False,
        }

    @application.get("/v1/contract", tags=["contract"])
    async def prediction_contract(request: Request) -> dict[str, object]:
        return active_runtime(request).contract_document()

    @application.get("/v1/metrics", tags=["operations"])
    async def process_metrics() -> dict[str, str | int | float]:
        return telemetry.as_dict()

    @application.post(
        "/v1/predict",
        response_model=SinglePredictionResponse,
        tags=["prediction"],
    )
    def predict_single(
        payload: ApplicationInput,
        request: Request,
    ) -> SinglePredictionResponse:
        prediction = active_runtime(request).score_single(payload)
        request.state.scored_rows = 1
        return SinglePredictionResponse(
            request_id=request.state.request_id,
            contract_version=CONTRACT_VERSION,
            prediction=prediction,
        )

    @application.post(
        "/v1/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["prediction"],
    )
    def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
    ) -> BatchPredictionResponse:
        loaded_runtime = active_runtime(request)
        predictions = loaded_runtime.score_batch(payload.applications)
        request.state.scored_rows = len(predictions)
        review_count = sum(
            prediction.exact_capacity_review is True
            for prediction in predictions
        )
        return BatchPredictionResponse(
            request_id=request.state.request_id,
            batch_id=payload.batch_id,
            contract_version=CONTRACT_VERSION,
            row_count=len(predictions),
            requested_review_rate=loaded_runtime.requested_review_rate,
            review_count=review_count,
            predictions=predictions,
        )

    return application


app = create_app()
