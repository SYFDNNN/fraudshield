"""Schema validation for the raw BAF Base dataset."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

EXPECTED_COLUMNS: tuple[str, ...] = (
    "fraud_bool",
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
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
    "device_fraud_count",
    "month",
)


def _is_numeric(series: pd.Series) -> bool:
    """Return whether a pandas Series uses a numeric dtype."""
    return pd.api.types.is_numeric_dtype(series.dtype)


def _numeric_column(*checks: pa.Check) -> pa.Column:
    """Create a required, non-null numeric column."""
    return pa.Column(
        dtype=None,
        checks=[
            pa.Check(
                _is_numeric,
                element_wise=False,
                error="Column must use a numeric dtype.",
            ),
            *checks,
        ],
        nullable=False,
        required=True,
    )


def _categorical_column(allowed_values: list[str]) -> pa.Column:
    """Create a required categorical column with known values."""
    return pa.Column(
        dtype=None,
        checks=pa.Check.isin(allowed_values),
        nullable=False,
        required=True,
    )


def _binary_column() -> pa.Column:
    """Create a required binary column containing only zero or one."""
    return _numeric_column(pa.Check.isin([0, 1]))


BASE_DATASET_SCHEMA = pa.DataFrameSchema(
    {
        "fraud_bool": _binary_column(),
        "income": _numeric_column(
            pa.Check.in_range(0.1, 0.9),
        ),
        "name_email_similarity": _numeric_column(
            pa.Check.in_range(0.0, 1.0),
        ),
        "prev_address_months_count": _numeric_column(
            # The observed Base.csv maximum is 383, while the
            # datasheet documents a maximum of 380.
            pa.Check.ge(-1),
        ),
        "current_address_months_count": _numeric_column(
            pa.Check.ge(-1),
        ),
        "customer_age": _numeric_column(
            pa.Check.in_range(10, 90),
        ),
        "days_since_request": _numeric_column(
            pa.Check.ge(0),
        ),
        "intended_balcon_amount": _numeric_column(),
        "payment_type": _categorical_column(
            ["AA", "AB", "AC", "AD", "AE"],
        ),
        "zip_count_4w": _numeric_column(
            pa.Check.ge(1),
        ),
        "velocity_6h": _numeric_column(),
        "velocity_24h": _numeric_column(),
        "velocity_4w": _numeric_column(),
        "bank_branch_count_8w": _numeric_column(
            pa.Check.ge(0),
        ),
        "date_of_birth_distinct_emails_4w": _numeric_column(
            pa.Check.ge(0),
        ),
        "employment_status": _categorical_column(
            ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
        ),
        "credit_risk_score": _numeric_column(),
        "email_is_free": _binary_column(),
        "housing_status": _categorical_column(
            ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
        ),
        "phone_home_valid": _binary_column(),
        "phone_mobile_valid": _binary_column(),
        "bank_months_count": _numeric_column(
            pa.Check.in_range(-1, 32),
        ),
        "has_other_cards": _binary_column(),
        "proposed_credit_limit": _numeric_column(
            # Base.csv v1 contains 190 and 2100 even though the
            # datasheet documents a range from 200 to 2000.
            pa.Check.in_range(190, 2100),
        ),
        "foreign_request": _binary_column(),
        "source": _categorical_column(
            ["INTERNET", "TELEAPP"],
        ),
        "session_length_in_minutes": _numeric_column(
            pa.Check.ge(-1),
        ),
        "device_os": _categorical_column(
            ["linux", "macintosh", "other", "windows", "x11"],
        ),
        "keep_alive_session": _binary_column(),
        "device_distinct_emails_8w": _numeric_column(
            pa.Check.in_range(-1, 2),
        ),
        "device_fraud_count": _binary_column(),
        "month": _numeric_column(
            pa.Check.in_range(0, 7),
        ),
    },
    strict=True,
    ordered=True,
    coerce=False,
    unique_column_names=True,
    name="base_dataset_schema",
)


def validate_base_dataset(
    dataframe: pd.DataFrame,
    *,
    check_duplicates: bool = True,
) -> pd.DataFrame:
    """Validate an unmodified BAF Base dataframe.

    Args:
        dataframe: Raw dataframe loaded from Base.csv.
        check_duplicates: Whether exact duplicate rows should be rejected.

    Returns:
        The validated dataframe.

    Raises:
        pandera.errors.SchemaErrors: If schema validation fails.
        ValueError: If exact duplicate rows are detected.
    """
    validated_dataframe = BASE_DATASET_SCHEMA.validate(
        dataframe,
        lazy=True,
    )

    if check_duplicates:
        duplicate_count = int(validated_dataframe.duplicated().sum())

        if duplicate_count:
            raise ValueError(
                "Raw Base dataset contains "
                f"{duplicate_count:,} exact duplicate rows."
            )

    return validated_dataframe