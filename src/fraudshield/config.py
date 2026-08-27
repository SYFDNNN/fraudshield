"""Centralized configuration loading for FraudShield."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load one YAML project configuration as a dictionary."""
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path.resolve()}"
        )

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, Mapping):
        raise TypeError("Project configuration must contain a YAML mapping.")

    required_sections = {
        "project",
        "data",
        "split",
        "feature_policy",
        "preprocessing",
        "baseline",
        "evaluation",
        "artifacts",
    }
    missing_sections = sorted(required_sections - set(config))

    if missing_sections:
        raise KeyError(
            f"Project configuration is missing sections: {missing_sections}"
        )

    return dict(config)


def project_root_from_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Return the repository root inferred from configs/<file>.yaml."""
    config_path = Path(path).resolve()

    if config_path.parent.name != "configs":
        raise ValueError(
            "Configuration file must be stored directly in a configs directory."
        )

    return config_path.parent.parent


def resolve_project_path(
    value: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Resolve a configuration path relative to the repository root."""
    configured_path = Path(value)

    if configured_path.is_absolute():
        return configured_path

    return project_root_from_config(config_path) / configured_path
