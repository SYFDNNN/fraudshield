"""Flask web application for the FraudShield inference client."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template

DEFAULT_API_URL = os.getenv("FRAUDSHIELD_API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = float(
    os.getenv("FRAUDSHIELD_API_TIMEOUT_SECONDS", "15")
)


def create_app() -> Flask:
    """Create and configure Flask application."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    # Configuration
    app.config["API_URL"] = DEFAULT_API_URL
    app.config["API_TIMEOUT"] = API_TIMEOUT_SECONDS
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
    
    # Register blueprints
    from app.flask_routes import api_bp, main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.errorhandler(404)
    def not_found(error):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template("500.html"), 500

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
