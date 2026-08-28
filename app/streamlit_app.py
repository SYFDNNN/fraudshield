"""Portfolio demo client for the FraudShield inference API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

DEFAULT_API_URL = os.getenv("FRAUDSHIELD_API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = float(
    os.getenv("FRAUDSHIELD_API_TIMEOUT_SECONDS", "15")
)


def _api_get(api_url: str, path: str) -> dict[str, Any]:
    response = httpx.get(
        f"{api_url.rstrip('/')}{path}",
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _api_post(
    api_url: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = httpx.post(
        f"{api_url.rstrip('/')}{path}",
        json=payload,
        timeout=API_TIMEOUT_SECONDS,
    )

    if response.is_error:
        try:
            detail = response.json()
        except json.JSONDecodeError:
            detail = response.text

        raise RuntimeError(f"API {response.status_code}: {detail}")

    return response.json()


def _binary_input(label: str, *, key: str, value: int = 0) -> int:
    return int(
        st.selectbox(
            label,
            options=[0, 1],
            index=value,
            format_func=lambda item: "Ya" if item else "Tidak",
            key=key,
        )
    )


def _single_application_form() -> dict[str, Any] | None:
    with st.form("single_application"):
        st.subheader("Identitas aplikasi")
        application_id = st.text_input(
            "Application ID anonim",
            value="demo-application-001",
            help="Gunakan identifier teknis, bukan nama, email, atau nomor telepon.",
        )

        left, middle, right = st.columns(3)

        with left:
            st.markdown("**Profil dan finansial**")
            income = st.slider("Income", 0.1, 0.9, 0.6, 0.1)
            name_email_similarity = st.slider(
                "Name–email similarity",
                0.0,
                1.0,
                0.65,
                0.01,
            )
            customer_age = st.number_input(
                "Customer age",
                min_value=10,
                max_value=90,
                value=40,
                step=10,
            )
            proposed_credit_limit = st.number_input(
                "Proposed credit limit",
                min_value=190.0,
                max_value=2100.0,
                value=1000.0,
                step=10.0,
            )
            intended_balcon_amount = st.number_input(
                "Intended balance amount",
                value=20.0,
                step=1.0,
                help="Nilai negatif mengikuti semantic-missing contract.",
            )
            employment_status = st.selectbox(
                "Employment status",
                ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
            )
            housing_status = st.selectbox(
                "Housing status",
                ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
            )

        with middle:
            st.markdown("**Alamat, kontak, dan sesi**")
            prev_address_months_count = st.number_input(
                "Previous address months",
                min_value=-1,
                value=-1,
                step=1,
            )
            current_address_months_count = st.number_input(
                "Current address months",
                min_value=-1,
                value=24,
                step=1,
            )
            bank_months_count = st.number_input(
                "Bank months",
                min_value=-1,
                max_value=32,
                value=12,
                step=1,
            )
            session_length_in_minutes = st.number_input(
                "Session length (minutes)",
                min_value=-1.0,
                value=8.0,
                step=0.5,
            )
            phone_home_valid = _binary_input(
                "Home phone valid",
                key="home_phone",
                value=1,
            )
            phone_mobile_valid = _binary_input(
                "Mobile phone valid",
                key="mobile_phone",
                value=1,
            )
            email_is_free = _binary_input(
                "Free email provider",
                key="free_email",
                value=1,
            )

        with right:
            st.markdown("**Kanal, perangkat, dan agregat historis**")
            payment_type = st.selectbox(
                "Payment type",
                ["AA", "AB", "AC", "AD", "AE"],
            )
            source = st.selectbox("Application source", ["INTERNET", "TELEAPP"])
            device_os = st.selectbox(
                "Device OS",
                ["windows", "macintosh", "linux", "x11", "other"],
            )
            zip_count_4w = st.number_input(
                "ZIP count 4w",
                min_value=1,
                value=1200,
                step=1,
            )
            velocity_6h = st.number_input(
                "Velocity 6h",
                value=4500.0,
                step=10.0,
            )
            velocity_24h = st.number_input(
                "Velocity 24h",
                value=5200.0,
                step=10.0,
            )
            velocity_4w = st.number_input(
                "Velocity 4w",
                value=5000.0,
                step=10.0,
            )
            bank_branch_count_8w = st.number_input(
                "Bank branch count 8w",
                min_value=0,
                value=10,
                step=1,
            )
            date_of_birth_distinct_emails_4w = st.number_input(
                "DOB distinct emails 4w",
                min_value=0,
                value=2,
                step=1,
            )
            device_distinct_emails_8w = st.number_input(
                "Device distinct emails 8w",
                min_value=-1,
                max_value=2,
                value=1,
                step=1,
            )

        flags_left, flags_middle, flags_right = st.columns(3)

        with flags_left:
            has_other_cards = _binary_input(
                "Has other cards",
                key="other_cards",
            )

        with flags_middle:
            foreign_request = _binary_input(
                "Foreign request",
                key="foreign_request",
            )

        with flags_right:
            keep_alive_session = _binary_input(
                "Keep-alive session",
                key="keep_alive",
                value=1,
            )

        submitted = st.form_submit_button(
            "Hitung skor risiko",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    return {
        "application_id": application_id,
        "income": float(income),
        "name_email_similarity": float(name_email_similarity),
        "prev_address_months_count": int(prev_address_months_count),
        "current_address_months_count": int(current_address_months_count),
        "customer_age": int(customer_age),
        "intended_balcon_amount": float(intended_balcon_amount),
        "payment_type": payment_type,
        "zip_count_4w": int(zip_count_4w),
        "velocity_6h": float(velocity_6h),
        "velocity_24h": float(velocity_24h),
        "velocity_4w": float(velocity_4w),
        "bank_branch_count_8w": int(bank_branch_count_8w),
        "date_of_birth_distinct_emails_4w": int(
            date_of_birth_distinct_emails_4w
        ),
        "employment_status": employment_status,
        "email_is_free": email_is_free,
        "housing_status": housing_status,
        "phone_home_valid": phone_home_valid,
        "phone_mobile_valid": phone_mobile_valid,
        "bank_months_count": int(bank_months_count),
        "has_other_cards": has_other_cards,
        "proposed_credit_limit": float(proposed_credit_limit),
        "foreign_request": foreign_request,
        "source": source,
        "session_length_in_minutes": float(session_length_in_minutes),
        "device_os": device_os,
        "keep_alive_session": keep_alive_session,
        "device_distinct_emails_8w": int(device_distinct_emails_8w),
    }


def _render_prediction(prediction: dict[str, Any]) -> None:
    probability = float(prediction["fraud_probability"])
    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Fraud probability", f"{probability:.2%}")
    metric_middle.metric("Risk band", prediction["risk_band"])
    metric_right.metric(
        "Fixed-threshold signal",
        "Review" if prediction["fixed_threshold_review"] else "Tidak",
    )

    st.progress(min(max(probability, 0.0), 1.0))
    st.info(
        "Single scoring tidak menentukan exact-capacity review. Keputusan "
        "antrean 5% hanya sah pada endpoint batch."
    )
    st.caption(
        f"Model: {prediction['model_version']} · "
        f"Policy: {prediction['threshold_policy_version']}"
    )


def _single_tab(api_url: str) -> None:
    payload = _single_application_form()

    if payload is None:
        return

    try:
        response = _api_post(api_url, "/v1/predict", payload)
    except (httpx.HTTPError, RuntimeError) as error:
        st.error(str(error))
        return

    _render_prediction(response["prediction"])

    with st.expander("Response JSON"):
        st.json(response)


def _batch_tab(api_url: str, contract: dict[str, Any] | None) -> None:
    st.markdown(
        "Upload JSON berisi list objek aplikasi. Setiap objek wajib memiliki "
        "`application_id` unik dan seluruh fitur kontrak."
    )
    batch_id = st.text_input(
        "Batch ID",
        value="demo-review-window-001",
        help="Identifier stabil untuk jendela antrean ini.",
    )
    complete_window = st.checkbox(
        "Batch ini merupakan jendela keputusan yang lengkap",
        help=(
            "Exact-capacity 5% tidak sah untuk potongan batch atau batch "
            "yang sengaja diperkecil."
        ),
    )
    uploaded = st.file_uploader("Batch JSON", type=["json"])

    if contract:
        st.caption(
            f"Maksimum {contract['maximum_batch_size']:,} aplikasi per batch."
        )

    if uploaded is None:
        return

    try:
        applications = json.load(uploaded)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        st.error(f"JSON tidak valid: {error}")
        return

    if not isinstance(applications, list):
        st.error("File harus berisi JSON array.")
        return

    st.write(f"Jumlah aplikasi: **{len(applications):,}**")

    if contract and len(applications) > contract["maximum_batch_size"]:
        st.error("Jumlah aplikasi melebihi maximum batch size kontrak.")
        return

    if not st.button("Score dan ranking batch", type="primary"):
        return

    if not complete_window:
        st.error(
            "Konfirmasi complete decision window sebelum membuat antrean 5%."
        )
        return

    try:
        response = _api_post(
            api_url,
            "/v1/predict/batch",
            {
                "batch_id": batch_id,
                "complete_decision_window": True,
                "applications": applications,
            },
        )
    except (httpx.HTTPError, RuntimeError) as error:
        st.error(str(error))
        return

    st.success(
        f"{response['review_count']:,} dari {response['row_count']:,} "
        "aplikasi masuk exact-capacity review queue."
    )
    result_frame = pd.DataFrame(response["predictions"]).sort_values(
        "review_rank"
    )
    st.dataframe(result_frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Unduh hasil CSV",
        result_frame.to_csv(index=False).encode("utf-8"),
        file_name="fraudshield_batch_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="FraudShield",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("🛡️ FraudShield")
    st.markdown(
        "**Calibrated fraud-risk scoring untuk prioritas human review.** "
        "Demo ini tidak menerima label dan tidak dapat menolak aplikasi "
        "secara otomatis."
    )
    api_url = st.sidebar.text_input("Inference API URL", value=DEFAULT_API_URL)
    contract = None

    try:
        health = _api_get(api_url, "/health/ready")
        contract = _api_get(api_url, "/v1/contract")
        st.sidebar.success("API ready")
        st.sidebar.caption(health["model_version"])
        st.sidebar.caption(f"Calibrator: {health['calibrator']}")
    except httpx.HTTPError as error:
        st.sidebar.error(f"API belum tersedia: {error}")

    st.warning(
        "Score dan risk band bukan bukti fraud. Semua tindakan memerlukan "
        "investigasi manusia dan kontrol operasional tambahan."
    )
    single_tab, batch_tab, contract_tab = st.tabs(
        ["Single scoring", "Exact-capacity batch", "Production contract"]
    )

    with single_tab:
        _single_tab(api_url)

    with batch_tab:
        _batch_tab(api_url, contract)

    with contract_tab:
        if contract:
            st.json(contract)
        else:
            st.info("Hubungkan API untuk menampilkan kontrak aktif.")


if __name__ == "__main__":
    main()
