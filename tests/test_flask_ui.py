"""Smoke tests for the Flask presentation layer and local UI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.flask_app import create_app
from app.ui_helpers import REQUIRED_MODEL_FEATURES


@pytest.fixture
def client():
    """Return a Flask test client with testing safeguards enabled."""
    app = create_app()
    app.config.update(TESTING=True)

    with app.test_client() as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("path", "heading", "active_href"),
    [
        ("/", "The fraud review system", 'href="/"'),
        ("/simulation", "Simulation Playground", 'href="/simulation"'),
        ("/review-queue", "Review Queue", 'href="/review-queue"'),
        ("/system", "Model &amp; System", 'href="/system"'),
    ],
)
def test_primary_pages_render_linear_shell(client, path, heading, active_href):
    response = client.get(path)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert heading in html
    assert active_href in html
    assert "linear-topbar" in html
    assert 'id="mainContent"' in html
    assert 'aria-current="page"' in html


def test_not_found_page_uses_product_shell(client):
    response = client.get("/route-yang-tidak-ada")

    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert "Halaman tidak ditemukan" in html
    assert "linear-topbar" in html


@pytest.mark.parametrize("preset_name", ["rendah", "menengah", "tinggi"])
def test_scenario_presets_match_prediction_contract(client, preset_name):
    response = client.get(f"/api/presets/{preset_name}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["application_id"].startswith(f"demo-{preset_name}")
    assert set(REQUIRED_MODEL_FEATURES).issubset(payload)


def test_unknown_preset_returns_clear_error(client):
    response = client.get("/api/presets/unknown")

    assert response.status_code == 404
    assert "tidak ditemukan" in response.get_json()["error"]


def test_batch_validation_rejects_non_json_body(client):
    response = client.post("/api/validate-batch", data="not-json")

    assert response.status_code == 415


def test_static_ui_assets_are_present_and_non_empty():
    project_root = Path(__file__).resolve().parents[1]
    expected_assets = (
        "app/static/css/styles.css",
        "app/static/css/linear-theme.css",
        "app/static/css/overview.css",
        "app/static/css/simulation.css",
        "app/static/css/review_queue.css",
        "app/static/css/system.css",
        "app/static/js/app.js",
        "app/static/js/overview.js",
        "app/static/js/simulation.js",
        "app/static/js/review_queue.js",
        "app/static/js/system.js",
    )

    for relative_path in expected_assets:
        asset = project_root / relative_path
        assert asset.is_file(), relative_path
        assert asset.stat().st_size > 100, relative_path
