"""Tests for the raw Base dataset validation contract."""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from fraudshield import __version__
from fraudshield.validation import validate_base_dataset


def build_valid_dataframe() -> pd.DataFrame:
    """Build one valid synthetic row for schema testing."""
    record = {
        "fraud_bool": 0,
        "income": 0.5,
        "name_email_similarity": 0.8,
        "prev_address_months_count": -1,
        "current_address_months_count": 12,
        "customer_age": 30,
        "days_since_request": 1.5,
        "intended_balcon_amount": -1.0,
        "payment_type": "AB",
        "zip_count_4w": 10,
        "velocity_6h": 100.0,
        "velocity_24h": 2000.0,
        "velocity_4w": 4000.0,
        "bank_branch_count_8w": 2,
        "date_of_birth_distinct_emails_4w": 1,
        "employment_status": "CA",
        "credit_risk_score": 100,
        "email_is_free": 1,
        "housing_status": "BA",
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "bank_months_count": -1,
        "has_other_cards": 0,
        "proposed_credit_limit": 500.0,
        "foreign_request": 0,
        "source": "INTERNET",
        "session_length_in_minutes": 10.0,
        "device_os": "windows",
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 0,
        "device_fraud_count": 0,
        "month": 0,
    }

    return pd.DataFrame([record])


def test_package_version_is_defined() -> None:
    """Ensure the package version is defined."""
    assert __version__ == "0.1.0"


def test_valid_base_dataframe_passes_schema() -> None:
    """Ensure a valid raw-data example passes validation."""
    dataframe = build_valid_dataframe()

    validated_dataframe = validate_base_dataset(dataframe)

    assert validated_dataframe.shape == (1, 32)


def test_invalid_target_value_fails_schema() -> None:
    """Ensure a non-binary target value fails validation."""
    dataframe = build_valid_dataframe()
    dataframe.loc[0, "fraud_bool"] = 2

    with pytest.raises(SchemaErrors):
        validate_base_dataset(dataframe)