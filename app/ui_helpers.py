"""Pure helpers shared by the FraudShield web client.

This module intentionally has no web-framework or inference-runtime dependency so
that input preparation can be tested without loading model artifacts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

APPLICATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

REQUIRED_MODEL_FEATURES: tuple[str, ...] = (
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

DEFAULT_APPLICATION: dict[str, Any] = {
    "application_id": "demo-application-001",
    "income": 0.6,
    "name_email_similarity": 0.65,
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

PRIMARY_EDITABLE_FEATURES: tuple[str, ...] = (
    "income",
    "current_address_months_count",
    "customer_age",
    "intended_balcon_amount",
    "phone_mobile_valid",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
)
SYSTEM_SIGNAL_FEATURES: tuple[str, ...] = tuple(
    feature
    for feature in REQUIRED_MODEL_FEATURES
    if feature not in PRIMARY_EDITABLE_FEATURES
)

SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "rendah": {
        **DEFAULT_APPLICATION,
        "application_id": "demo-rendah-001",
        "income": 0.4,
        "name_email_similarity": 0.92,
        "prev_address_months_count": 48,
        "current_address_months_count": 72,
        "customer_age": 50,
        "intended_balcon_amount": 100.0,
        "zip_count_4w": 400,
        "velocity_6h": 1200.0,
        "velocity_24h": 1800.0,
        "velocity_4w": 3000.0,
        "bank_branch_count_8w": 2,
        "date_of_birth_distinct_emails_4w": 1,
        "email_is_free": 0,
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "bank_months_count": 30,
        "has_other_cards": 1,
        "proposed_credit_limit": 500.0,
        "foreign_request": 0,
        "session_length_in_minutes": 15.0,
        "device_os": "macintosh",
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
    },
    "menengah": {
        **DEFAULT_APPLICATION,
        "application_id": "demo-menengah-001",
    },
    "tinggi": {
        **DEFAULT_APPLICATION,
        "application_id": "demo-tinggi-001",
        "income": 0.9,
        "name_email_similarity": 0.10,
        "prev_address_months_count": -1,
        "current_address_months_count": 1,
        "customer_age": 20,
        "intended_balcon_amount": -1.0,
        "payment_type": "AC",
        "zip_count_4w": 3000,
        "velocity_6h": 10000.0,
        "velocity_24h": 12000.0,
        "velocity_4w": 9000.0,
        "bank_branch_count_8w": 30,
        "date_of_birth_distinct_emails_4w": 5,
        "employment_status": "CC",
        "email_is_free": 1,
        "housing_status": "BA",
        "phone_home_valid": 0,
        "phone_mobile_valid": 0,
        "bank_months_count": -1,
        "has_other_cards": 0,
        "proposed_credit_limit": 2000.0,
        "foreign_request": 1,
        "source": "INTERNET",
        "session_length_in_minutes": 1.0,
        "device_os": "windows",
        "keep_alive_session": 0,
        "device_distinct_emails_8w": 2,
    },
}

RISK_BAND_ORDER: tuple[str, ...] = (
    "sangat_tinggi",
    "tinggi",
    "menengah",
    "rendah",
)


@dataclass(frozen=True, slots=True)
class ParsedBatch:
    """Normalized representation of either supported upload shape."""

    batch_id: str
    applications: tuple[Any, ...]
    declared_complete: bool | None
    source_format: str


@dataclass(frozen=True, slots=True)
class BatchValidation:
    """Human-readable local validation result for a batch upload."""

    row_count: int
    unique_id_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the upload can be sent to the API."""
        return not self.errors


def default_application() -> dict[str, Any]:
    """Return a fresh complete request using synthetic demonstration values."""
    return DEFAULT_APPLICATION.copy()


def scenario_preset(name: str) -> dict[str, Any]:
    """Return one fresh illustrative low, medium, or high signal profile."""
    try:
        preset = SCENARIO_PRESETS[name]
    except KeyError as error:
        raise KeyError(f"Unknown scenario preset: {name}") from error

    return preset.copy()


def reason_summary(prediction: dict[str, Any], *, limit: int = 3) -> str:
    """Create a compact, non-causal summary for result tables and exports."""
    reasons = prediction.get("reason_codes")

    if not isinstance(reasons, list):
        return "Penjelasan tidak tersedia"

    labels: list[str] = []

    for reason in reasons[:limit]:
        if not isinstance(reason, dict):
            continue

        label = str(reason.get("feature_label", reason.get("feature", "Sinyal")))
        direction = reason.get("direction")
        arrow = "↑" if direction == "increases_risk" else "↓"
        labels.append(f"{arrow} {label}")

    return "; ".join(labels) if labels else "Penjelasan tidak tersedia"


def flatten_batch_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten nested explanation/action fields for analyst tables and CSV."""
    flattened: list[dict[str, Any]] = []

    for prediction in predictions:
        action = prediction.get("analyst_action")
        action_mapping = action if isinstance(action, dict) else {}
        reasons = prediction.get("reason_codes")
        reason_items = reasons if isinstance(reasons, list) else []
        flattened.append(
            {
                **{
                    key: value
                    for key, value in prediction.items()
                    if key not in {"reason_codes", "analyst_action"}
                },
                "analyst_action_code": action_mapping.get("code", ""),
                "analyst_action_label": action_mapping.get("label", ""),
                "top_local_signals": reason_summary(prediction),
                "reason_code_count": len(reason_items),
            }
        )

    return flattened


def normalize_batch_payload(
    payload: Any,
    *,
    fallback_batch_id: str,
) -> ParsedBatch:
    """Accept a raw application array or the documented API envelope."""
    if isinstance(payload, list):
        return ParsedBatch(
            batch_id=fallback_batch_id.strip(),
            applications=tuple(payload),
            declared_complete=None,
            source_format="raw_array",
        )

    if not isinstance(payload, dict):
        raise TypeError(
            "Isi file harus berupa array aplikasi atau objek envelope API."
        )

    if "applications" not in payload:
        raise ValueError("Envelope JSON tidak memiliki field 'applications'.")

    applications = payload["applications"]

    if not isinstance(applications, list):
        raise TypeError("Field 'applications' harus berupa JSON array.")

    supplied_batch_id = payload.get("batch_id", fallback_batch_id)

    if not isinstance(supplied_batch_id, str):
        raise TypeError("Field 'batch_id' harus berupa teks.")

    declared_complete = payload.get("complete_decision_window")

    if declared_complete is not None and not isinstance(
        declared_complete,
        bool,
    ):
        raise ValueError(
            "Field 'complete_decision_window' harus true atau false."
        )

    return ParsedBatch(
        batch_id=supplied_batch_id.strip(),
        applications=tuple(applications),
        declared_complete=declared_complete,
        source_format="api_envelope",
    )


def validate_batch_applications(
    applications: tuple[Any, ...] | list[Any],
    *,
    maximum_batch_size: int,
    required_features: tuple[str, ...] = REQUIRED_MODEL_FEATURES,
) -> BatchValidation:
    """Validate deterministic-ranking preconditions before an API request."""
    errors: list[str] = []
    warnings: list[str] = []
    row_count = len(applications)

    if row_count == 0:
        errors.append("Batch kosong. Tambahkan minimal satu pengajuan.")

    if row_count > maximum_batch_size:
        errors.append(
            "Jumlah pengajuan melebihi batas kontrak: "
            f"{row_count:,} > {maximum_batch_size:,}."
        )

    expected_fields = {"application_id", *required_features}
    identifiers: list[str] = []
    invalid_object_rows: list[int] = []
    invalid_id_rows: list[int] = []
    missing_field_rows: list[tuple[int, tuple[str, ...]]] = []
    extra_field_rows: list[tuple[int, tuple[str, ...]]] = []

    for index, application in enumerate(applications, start=1):
        if not isinstance(application, dict):
            invalid_object_rows.append(index)
            continue

        application_id = application.get("application_id")

        if not isinstance(application_id, str) or not (
            APPLICATION_ID_PATTERN.fullmatch(application_id)
        ):
            invalid_id_rows.append(index)
        else:
            identifiers.append(application_id)

        missing_fields = tuple(sorted(expected_fields - application.keys()))
        extra_fields = tuple(sorted(application.keys() - expected_fields))

        if missing_fields:
            missing_field_rows.append((index, missing_fields))

        if extra_fields:
            extra_field_rows.append((index, extra_fields))

    if invalid_object_rows:
        errors.append(
            "Setiap item harus berupa objek JSON. Baris bermasalah: "
            + _summarize_indexes(invalid_object_rows)
            + "."
        )

    if invalid_id_rows:
        errors.append(
            "application_id wajib unik, 1–128 karakter, dan hanya boleh "
            "memakai huruf, angka, titik, garis bawah, titik dua, atau "
            "tanda minus. Baris bermasalah: "
            + _summarize_indexes(invalid_id_rows)
            + "."
        )

    identifier_counts = Counter(identifiers)
    duplicates = sorted(
        identifier
        for identifier, count in identifier_counts.items()
        if count > 1
    )

    if duplicates:
        errors.append(
            "application_id duplikat ditemukan: "
            + ", ".join(duplicates[:5])
            + (" …" if len(duplicates) > 5 else "")
            + "."
        )

    if missing_field_rows:
        errors.append(
            "Ada field wajib yang hilang. "
            + _summarize_field_issues(missing_field_rows)
        )

    if extra_field_rows:
        errors.append(
            "Ada field di luar kontrak. "
            + _summarize_field_issues(extra_field_rows)
        )

    if 0 < row_count < 20:
        warnings.append(
            "Pada batch kecil, pembulatan ke atas membuat proporsi review "
            "aktual dapat melebihi 5%."
        )

    return BatchValidation(
        row_count=row_count,
        unique_id_count=len(set(identifiers)),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def risk_band_distribution(
    predictions: list[dict[str, Any]],
) -> list[dict[str, int | float | str]]:
    """Summarize response risk bands in a stable display order."""
    counts: dict[str, int] = {}

    for prediction in predictions:
        label = str(prediction.get("risk_band", "tidak_diketahui"))
        counts[label] = counts.get(label, 0) + 1

    total = len(predictions)
    ordered_labels = [label for label in RISK_BAND_ORDER if label in counts]
    ordered_labels.extend(sorted(set(counts) - set(ordered_labels)))

    return [
        {
            "risk_band": label,
            "row_count": counts[label],
            "row_share": counts[label] / total if total else 0.0,
        }
        for label in ordered_labels
    ]


def format_api_error_detail(status_code: int, payload: Any) -> str:
    """Create a concise API error without echoing submitted field values."""
    detail = payload.get("detail") if isinstance(payload, dict) else None

    if isinstance(detail, str):
        description = detail
    elif isinstance(detail, list):
        issues: list[str] = []

        for item in detail[:5]:
            if not isinstance(item, dict):
                continue

            location = ".".join(
                str(part) for part in item.get("loc", []) if part != "body"
            )
            message = str(item.get("msg", "input tidak valid"))
            issues.append(f"{location}: {message}" if location else message)

        description = "; ".join(issues) or "Permintaan tidak valid."
    else:
        description = "API mengembalikan respons error yang tidak dikenali."

    return f"API {status_code}: {description}"


def _summarize_indexes(indexes: list[int], *, limit: int = 8) -> str:
    visible = ", ".join(str(index) for index in indexes[:limit])
    return visible + (" …" if len(indexes) > limit else "")


def _summarize_field_issues(
    issues: list[tuple[int, tuple[str, ...]]],
    *,
    limit: int = 3,
) -> str:
    visible = [
        f"baris {index}: {', '.join(fields[:5])}"
        + (" …" if len(fields) > 5 else "")
        for index, fields in issues[:limit]
    ]
    suffix = " …" if len(issues) > limit else ""
    return "; ".join(visible) + suffix + "."
