"""Data loading and leakage-safe temporal splitting utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pandas as pd

from fraudshield.validation import validate_base_dataset


@dataclass(frozen=True, slots=True)
class TemporalSplits:
    """Container for chronological dataset partitions."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return all splits as a name-to-dataframe mapping."""
        return {
            "train": self.train,
            "calibration": self.calibration,
            "validation": self.validation,
            "test": self.test,
        }

    @property
    def total_rows(self) -> int:
        """Return the total number of rows across all splits."""
        return sum(len(split) for split in self.as_dict().values())


def load_base_dataset(
    path: str | Path,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the raw BAF Base CSV without treating a column as an index.

    Args:
        path: Location of Base.csv.
        validate: Whether the complete raw schema should be validated.

    Returns:
        Loaded Base dataframe.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        pandera.errors.SchemaErrors: If raw schema validation fails.
    """
    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Base dataset was not found: {dataset_path.resolve()}"
        )

    dataframe = pd.read_csv(
        dataset_path,
        low_memory=False,
    )

    if validate:
        return validate_base_dataset(dataframe)

    return dataframe


def _normalize_periods(
    split_name: str,
    periods: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize one configured period collection."""
    normalized_periods = tuple(periods)

    if not normalized_periods:
        raise ValueError(f"{split_name} periods must not be empty.")

    if len(normalized_periods) != len(set(normalized_periods)):
        raise ValueError(
            f"{split_name} periods contain duplicate values: "
            f"{normalized_periods}."
        )

    if normalized_periods != tuple(sorted(normalized_periods)):
        raise ValueError(
            f"{split_name} periods must be ordered chronologically: "
            f"{normalized_periods}."
        )

    return normalized_periods


def temporal_split(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    train_periods: Sequence[int],
    calibration_periods: Sequence[int],
    validation_periods: Sequence[int],
    test_periods: Sequence[int],
) -> TemporalSplits:
    """Split a dataframe into non-overlapping chronological partitions.

    Every observed period must be assigned exactly once. The configured
    partitions must also follow strict chronological order.

    Args:
        dataframe: Validated dataframe to split.
        time_column: Column containing the temporal period.
        train_periods: Periods used for development and model fitting.
        calibration_periods: Periods reserved for probability calibration.
        validation_periods: Periods used for model and threshold selection.
        test_periods: Periods reserved for final evaluation.

    Returns:
        Train, calibration, validation, and test dataframes.

    Raises:
        KeyError: If the temporal column does not exist.
        ValueError: If periods are null, empty, overlapping, incomplete,
            unknown, or not chronological.
    """
    if time_column not in dataframe.columns:
        raise KeyError(f"Time column does not exist: {time_column}")

    if dataframe.empty:
        raise ValueError("Cannot split an empty dataframe.")

    if dataframe[time_column].isna().any():
        raise ValueError(
            f"Time column contains missing values: {time_column}"
        )

    period_groups = {
        "train": _normalize_periods("train", train_periods),
        "calibration": _normalize_periods(
            "calibration",
            calibration_periods,
        ),
        "validation": _normalize_periods(
            "validation",
            validation_periods,
        ),
        "test": _normalize_periods("test", test_periods),
    }

    configured_periods = [
        period
        for periods in period_groups.values()
        for period in periods
    ]

    if len(configured_periods) != len(set(configured_periods)):
        raise ValueError("Temporal split periods overlap.")

    ordered_groups = list(period_groups.items())

    for (
        (previous_name, previous_periods),
        (current_name, current_periods),
    ) in pairwise(ordered_groups):
        if max(previous_periods) >= min(current_periods):
            raise ValueError(
                "Temporal splits are not strictly chronological: "
                f"{previous_name}={previous_periods}, "
                f"{current_name}={current_periods}."
            )

    observed_periods = set(
        dataframe[time_column].unique().tolist()
    )
    configured_period_set = set(configured_periods)

    if observed_periods != configured_period_set:
        unassigned_periods = sorted(
            observed_periods - configured_period_set
        )
        unknown_periods = sorted(
            configured_period_set - observed_periods
        )

        raise ValueError(
            "Configured periods do not match observed periods. "
            f"Unassigned observed periods={unassigned_periods}; "
            f"configured periods absent from data={unknown_periods}."
        )

    split_frames = {
        split_name: dataframe.loc[
            dataframe[time_column].isin(periods)
        ].copy()
        for split_name, periods in period_groups.items()
    }

    for split_name, split_frame in split_frames.items():
        if split_frame.empty:
            raise ValueError(f"{split_name} split is empty.")

    splits = TemporalSplits(
        train=split_frames["train"],
        calibration=split_frames["calibration"],
        validation=split_frames["validation"],
        test=split_frames["test"],
    )

    if splits.total_rows != len(dataframe):
        raise RuntimeError(
            "Temporal split row count does not match source row count."
        )

    return splits


def select_model_features(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: Sequence[str],
) -> pd.DataFrame:
    """Return predictor columns after applying the feature exclusion policy."""
    unique_excluded_columns = tuple(dict.fromkeys(excluded_columns))

    missing_columns = sorted(
        set(unique_excluded_columns) - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Excluded columns do not exist: {missing_columns}"
        )

    excluded_column_set = set(unique_excluded_columns)
    selected_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_column_set
    ]

    if not selected_columns:
        raise ValueError("Feature exclusion removed every column.")

    return dataframe.loc[:, selected_columns].copy()