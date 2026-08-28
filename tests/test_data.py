"""Tests for raw data validation and leakage-safe temporal splitting."""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from fraudshield import __version__
from fraudshield.data import select_model_features, temporal_split
from fraudshield.validation import validate_base_dataset


def build_valid_dataframe() -> pd.DataFrame:
    """Create one valid example matching the raw Base dataset schema."""
    record = {
        "fraud_bool": 0,
        "income": 0.5,
        "name_email_similarity": 0.8,
        "prev_address_months_count": 12,
        "current_address_months_count": 24,
        "customer_age": 30,
        "days_since_request": 1.5,
        "intended_balcon_amount": 100.0,
        "payment_type": "AA",
        "zip_count_4w": 100,
        "velocity_6h": 100.0,
        "velocity_24h": 200.0,
        "velocity_4w": 300.0,
        "bank_branch_count_8w": 10,
        "date_of_birth_distinct_emails_4w": 1,
        "employment_status": "CA",
        "credit_risk_score": 100,
        "email_is_free": 1,
        "housing_status": "BA",
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "bank_months_count": 12,
        "has_other_cards": 0,
        "proposed_credit_limit": 500.0,
        "foreign_request": 0,
        "source": "INTERNET",
        "session_length_in_minutes": 10.0,
        "device_os": "windows",
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
        "device_fraud_count": 0,
        "month": 0,
    }

    return pd.DataFrame([record])


@pytest.fixture
def temporal_dataframe() -> pd.DataFrame:
    """Create a dataframe containing periods zero through seven."""
    return pd.DataFrame(
        {
            "fraud_bool": [0, 1] * 8,
            "feature": list(range(16)),
            "month": [
                month
                for month in range(8)
                for _ in range(2)
            ],
        }
    )


def test_package_version_is_defined() -> None:
    """Ensure the package version is defined."""
    assert __version__ == "0.2.0"


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


def test_temporal_split_is_complete_and_non_overlapping(
    temporal_dataframe: pd.DataFrame,
) -> None:
    """Every row should appear in exactly one temporal split."""
    splits = temporal_split(
        temporal_dataframe,
        time_column="month",
        train_periods=[0, 1, 2, 3, 4],
        calibration_periods=[5],
        validation_periods=[6],
        test_periods=[7],
    )

    assert len(splits.train) == 10
    assert len(splits.calibration) == 2
    assert len(splits.validation) == 2
    assert len(splits.test) == 2
    assert splits.total_rows == len(temporal_dataframe)

    split_indexes = [
        set(split.index)
        for split in splits.as_dict().values()
    ]
    combined_indexes = set().union(*split_indexes)

    assert combined_indexes == set(temporal_dataframe.index)

    for index, left_indexes in enumerate(split_indexes):
        for right_indexes in split_indexes[index + 1 :]:
            assert left_indexes.isdisjoint(right_indexes)


def test_temporal_split_rejects_overlapping_periods(
    temporal_dataframe: pd.DataFrame,
) -> None:
    """A period must not appear in multiple temporal splits."""
    with pytest.raises(ValueError, match="overlap"):
        temporal_split(
            temporal_dataframe,
            time_column="month",
            train_periods=[0, 1, 2, 3, 4],
            calibration_periods=[4, 5],
            validation_periods=[6],
            test_periods=[7],
        )


def test_temporal_split_rejects_unassigned_periods(
    temporal_dataframe: pd.DataFrame,
) -> None:
    """Every observed temporal period must be assigned."""
    with pytest.raises(
        ValueError,
        match="do not match observed periods",
    ):
        temporal_split(
            temporal_dataframe,
            time_column="month",
            train_periods=[0, 1, 2, 3],
            calibration_periods=[5],
            validation_periods=[6],
            test_periods=[7],
        )


def test_temporal_split_rejects_non_chronological_order(
    temporal_dataframe: pd.DataFrame,
) -> None:
    """Later temporal splits must not contain earlier periods."""
    with pytest.raises(
        ValueError,
        match="not strictly chronological",
    ):
        temporal_split(
            temporal_dataframe,
            time_column="month",
            train_periods=[1],
            calibration_periods=[0],
            validation_periods=[2, 3, 4, 5, 6],
            test_periods=[7],
        )


def test_select_model_features_applies_exclusion_policy(
    temporal_dataframe: pd.DataFrame,
) -> None:
    """Target and split-only columns should be removed."""
    feature_frame = select_model_features(
        temporal_dataframe,
        excluded_columns=["fraud_bool", "month"],
    )

    assert feature_frame.columns.tolist() == ["feature"]
    assert feature_frame.index.equals(temporal_dataframe.index)
