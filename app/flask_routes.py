"""Flask route handlers and API proxy endpoints."""

from __future__ import annotations

from typing import Any

import httpx
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
)

from app.ui_helpers import (
    SCENARIO_PRESETS,
    normalize_batch_payload,
    validate_batch_applications,
)

# Blueprint for main page routes
main_bp = Blueprint("main", __name__)

# Blueprint for API proxy routes
api_bp = Blueprint("api", __name__)


# ============================================================================
# Main Page Routes
# ============================================================================


@main_bp.route("/")
def index():
    """Landing page / Overview."""
    return render_template("overview.html")


@main_bp.route("/simulation")
def simulation():
    """Simulation Playground for single application testing."""
    return render_template("simulation.html", presets=SCENARIO_PRESETS)


@main_bp.route("/review-queue")
def review_queue():
    """Review Queue for batch operations."""
    return render_template("review_queue.html")


@main_bp.route("/system")
def system_info():
    """Model & System information."""
    return render_template("system.html")


# ============================================================================
# API Proxy Routes
# ============================================================================


def _get_api_url() -> str:
    """Get configured API URL."""
    return str(current_app.config.get("API_URL", "http://localhost:8000"))


def _get_api_timeout() -> float:
    """Get configured API timeout."""
    return float(current_app.config.get("API_TIMEOUT", 15.0))


def _api_get(path: str) -> dict[str, Any]:
    """Make GET request to inference API."""
    api_url = _get_api_url()
    timeout = _get_api_timeout()
    
    try:
        response = httpx.get(
            f"{api_url.rstrip('/')}{path}",
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as error:
        raise ApiError("API tidak merespons sebelum batas waktu.", 504) from error
    except httpx.ConnectError as error:
        raise ApiError(
            "API tidak dapat dihubungi. Pastikan layanan aktif.",
            503,
        ) from error
    except httpx.HTTPStatusError as e:
        detail = None
        try:
            detail = e.response.json()
        except ValueError:
            detail = None
        raise ApiError(
            f"API error: {e.response.status_code}",
            e.response.status_code,
            detail,
        ) from e
    except httpx.RequestError as error:
        raise ApiError(f"API request failed: {error!s}", 502) from error


def _api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Make POST request to inference API."""
    api_url = _get_api_url()
    timeout = _get_api_timeout()
    
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}{path}",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as error:
        raise ApiError("API tidak merespons sebelum batas waktu.", 504) from error
    except httpx.ConnectError as error:
        raise ApiError(
            "API tidak dapat dihubungi. Pastikan layanan aktif.",
            503,
        ) from error
    except httpx.HTTPStatusError as e:
        detail = None
        try:
            detail = e.response.json()
        except ValueError:
            detail = None
        raise ApiError(
            f"API error: {e.response.status_code}",
            e.response.status_code,
            detail,
        ) from e
    except httpx.RequestError as error:
        raise ApiError(f"API request failed: {error!s}", 502) from error


class ApiError(Exception):
    """API communication error."""
    
    def __init__(self, message: str, status_code: int = 500, detail: Any = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


@api_bp.errorhandler(ApiError)
def handle_api_error(error: ApiError):
    """Handle API proxy errors."""
    response = {
        "error": error.message,
        "status_code": error.status_code,
    }
    if error.detail:
        response["detail"] = error.detail
    return jsonify(response), error.status_code


@api_bp.route("/health")
def health_check():
    """Check API health and return combined status."""
    try:
        health = _api_get("/health/ready")
        contract = _api_get("/v1/contract")
        
        return jsonify({
            "status": "ready",
            "health": health,
            "contract": contract,
        })
    except ApiError as e:
        return jsonify({
            "status": "offline",
            "error": e.message,
        }), e.status_code


@api_bp.route("/contract")
def get_contract():
    """Get prediction contract from API."""
    contract = _api_get("/v1/contract")
    return jsonify(contract)


@api_bp.route("/metrics")
def get_metrics():
    """Get API metrics."""
    metrics = _api_get("/v1/metrics")
    return jsonify(metrics)


@api_bp.route("/predict", methods=["POST"])
def predict_single():
    """Proxy single prediction request to API."""
    payload = request.get_json()
    
    if not payload:
        return jsonify({"error": "Request body harus berupa JSON"}), 400
    
    result = _api_post("/v1/predict", payload)
    return jsonify(result)


@api_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Proxy batch prediction request to API."""
    payload = request.get_json()
    
    if not payload:
        return jsonify({"error": "Request body harus berupa JSON"}), 400
    
    result = _api_post("/v1/predict/batch", payload)
    return jsonify(result)


@api_bp.route("/validate-batch", methods=["POST"])
def validate_batch():
    """Validate batch payload locally before sending to API."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body harus berupa JSON"}), 400
    
    try:
        # Parse batch payload
        parsed = normalize_batch_payload(
            data.get("payload"),
            fallback_batch_id=data.get("batch_id", "batch-upload"),
        )
        
        # Validate applications
        validation = validate_batch_applications(
            parsed.applications,
            maximum_batch_size=5000,
        )
        
        return jsonify({
            "valid": validation.is_valid,
            "row_count": validation.row_count,
            "unique_id_count": validation.unique_id_count,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "batch_id": parsed.batch_id,
            "declared_complete": parsed.declared_complete,
            "source_format": parsed.source_format,
        })
    except (TypeError, ValueError, KeyError) as e:
        return jsonify({
            "valid": False,
            "error": str(e),
        }), 400


@api_bp.route("/presets/<preset_name>")
def get_preset(preset_name: str):
    """Get scenario preset data."""
    if preset_name not in SCENARIO_PRESETS:
        return jsonify({"error": f"Preset '{preset_name}' tidak ditemukan"}), 404
    
    return jsonify(SCENARIO_PRESETS[preset_name])
