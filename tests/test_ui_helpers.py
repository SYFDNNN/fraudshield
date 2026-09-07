"""Unit tests for web-client input preparation and local validation."""

from __future__ import annotations

import unittest

from app.ui_helpers import (
    PRIMARY_EDITABLE_FEATURES,
    REQUIRED_MODEL_FEATURES,
    SYSTEM_SIGNAL_FEATURES,
    default_application,
    flatten_batch_predictions,
    format_api_error_detail,
    normalize_batch_payload,
    risk_band_distribution,
    scenario_preset,
    validate_batch_applications,
)


class UiHelperTests(unittest.TestCase):
    """Exercise helpers without starting Flask or loading model artifacts."""

    def test_default_application_is_complete_and_returns_fresh_copy(self) -> None:
        """The simple form must still produce all 27 model features."""
        first = default_application()
        second = default_application()

        first["income"] = 0.1

        self.assertEqual(set(second) - {"application_id"}, set(REQUIRED_MODEL_FEATURES))
        self.assertEqual(second["income"], 0.6)

    def test_presets_are_complete_fresh_and_keep_nineteen_system_signals(self) -> None:
        """Every scenario must satisfy the same 27-feature API contract."""
        low = scenario_preset("rendah")
        high = scenario_preset("tinggi")
        low["income"] = 0.1

        self.assertEqual(
            set(scenario_preset("rendah")) - {"application_id"},
            set(REQUIRED_MODEL_FEATURES),
        )
        self.assertEqual(len(PRIMARY_EDITABLE_FEATURES), 8)
        self.assertEqual(len(SYSTEM_SIGNAL_FEATURES), 19)
        self.assertEqual(scenario_preset("rendah")["income"], 0.4)
        self.assertEqual(high["foreign_request"], 1)

    def test_raw_array_uses_entered_batch_id(self) -> None:
        """A raw array should remain compatible with the original UI format."""
        parsed = normalize_batch_payload(
            [default_application()],
            fallback_batch_id="window-001",
        )

        self.assertEqual(parsed.batch_id, "window-001")
        self.assertEqual(parsed.source_format, "raw_array")
        self.assertIsNone(parsed.declared_complete)

    def test_api_envelope_preserves_batch_metadata(self) -> None:
        """The documented API request envelope should upload without editing."""
        parsed = normalize_batch_payload(
            {
                "batch_id": "window-envelope",
                "complete_decision_window": True,
                "applications": [default_application()],
            },
            fallback_batch_id="ignored",
        )

        self.assertEqual(parsed.batch_id, "window-envelope")
        self.assertTrue(parsed.declared_complete)
        self.assertEqual(parsed.source_format, "api_envelope")

    def test_invalid_envelope_is_rejected(self) -> None:
        """An envelope without an application array is not silently accepted."""
        with self.assertRaisesRegex(ValueError, "applications"):
            normalize_batch_payload(
                {"batch_id": "window-001"},
                fallback_batch_id="window-001",
            )

    def test_invalid_batch_types_raise_type_error(self) -> None:
        """Structural type mismatches should use Python's TypeError."""
        invalid_payloads = (
            "not-an-object",
            {"applications": "not-an-array"},
            {"applications": [], "batch_id": 123},
        )

        for payload in invalid_payloads:
            with (
                self.subTest(payload=payload),
                self.assertRaises(TypeError),
            ):
                normalize_batch_payload(
                        payload,
                        fallback_batch_id="window-001",
                    )

    def test_valid_batch_passes_local_contract_checks(self) -> None:
        """Complete unique rows should be ready for the API."""
        applications = [
            default_application(),
            {**default_application(), "application_id": "application-002"},
        ]

        validation = validate_batch_applications(
            applications,
            maximum_batch_size=5000,
        )

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.unique_id_count, 2)
        self.assertTrue(validation.warnings)

    def test_duplicate_and_missing_fields_are_reported(self) -> None:
        """Ranking ambiguity and incomplete rows must block processing."""
        incomplete = default_application()
        incomplete.pop("velocity_4w")
        validation = validate_batch_applications(
            [default_application(), incomplete],
            maximum_batch_size=5000,
        )

        self.assertFalse(validation.is_valid)
        self.assertIn("duplikat", " ".join(validation.errors))
        self.assertIn("velocity_4w", " ".join(validation.errors))

    def test_maximum_size_and_extra_fields_are_reported(self) -> None:
        """The UI should fail fast before sending oversized or leaky input."""
        application = default_application()
        application["fraud_bool"] = 1
        validation = validate_batch_applications(
            [application],
            maximum_batch_size=0,
        )

        self.assertFalse(validation.is_valid)
        self.assertIn("batas kontrak", " ".join(validation.errors))
        self.assertIn("fraud_bool", " ".join(validation.errors))

    def test_risk_distribution_is_stable_and_complete(self) -> None:
        """Known risk bands should use the operational display order."""
        distribution = risk_band_distribution(
            [
                {"risk_band": "rendah"},
                {"risk_band": "tinggi"},
                {"risk_band": "rendah"},
                {"risk_band": "sangat_tinggi"},
            ]
        )

        self.assertEqual(
            [row["risk_band"] for row in distribution],
            ["sangat_tinggi", "tinggi", "rendah"],
        )
        self.assertAlmostEqual(sum(row["row_share"] for row in distribution), 1.0)

    def test_batch_predictions_flatten_explanation_and_action(self) -> None:
        """Nested Phase 8 outputs should be usable in analyst tables and CSV."""
        flattened = flatten_batch_predictions(
            [
                {
                    "application_id": "application-001",
                    "fraud_probability": 0.4,
                    "reason_codes": [
                        {
                            "feature": "foreign_request",
                            "feature_label": "Permintaan lintas negara",
                            "direction": "increases_risk",
                        }
                    ],
                    "analyst_action": {
                        "code": "manual_review_queue",
                        "label": "Masukkan ke antrean pemeriksaan manusia",
                    },
                }
            ]
        )

        self.assertEqual(flattened[0]["reason_code_count"], 1)
        self.assertIn("↑", flattened[0]["top_local_signals"])
        self.assertEqual(
            flattened[0]["analyst_action_code"],
            "manual_review_queue",
        )

    def test_api_error_formatter_does_not_echo_input_values(self) -> None:
        """Pydantic input snapshots must not leak into a visible error."""
        message = format_api_error_detail(
            422,
            {
                "detail": [
                    {
                        "loc": ["body", "income"],
                        "msg": "Input should be a valid number",
                        "input": "private-value",
                    }
                ]
            },
        )

        self.assertIn("income", message)
        self.assertNotIn("private-value", message)

if __name__ == "__main__":
    unittest.main()
