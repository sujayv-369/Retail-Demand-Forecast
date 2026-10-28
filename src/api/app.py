"""
Flask application factory — serves the frontend, API endpoints,
and Gemini chat functionality.
"""

from __future__ import annotations
import logging
import os
import secrets
from pathlib import Path

from flask import Flask
from flask_cors import CORS
import os

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    project_root = Path(__file__).resolve().parent.parent.parent
    frontend_dir = project_root / "frontend"

    app = Flask(
        __name__,
        static_folder=str(frontend_dir),
        static_url_path="",
    )

    # CORS for development
    CORS(app)

    # Secret key for session management
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

    # Register Auth routes
    from src.api.auth_routes import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/api/auth")

    # Register API routes
    from src.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # Serve frontend
    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    # Direct root download routes for HTML anchors
    @app.route("/download-csv")
    @app.route("/reports/retail_demand_forecast.csv")
    @app.route("/reports/RetailForecaster_Forecast.csv")
    def root_download_csv():
        from src.api.routes import export_retail_report
        return export_retail_report()

    @app.route("/download-pdf")
    @app.route("/reports/executive_demand_report.pdf")
    @app.route("/reports/Executive_Demand_Report.pdf")
    def root_download_pdf():
        from src.api.routes import download_executive_pdf_report
        return download_executive_pdf_report()

    @app.route("/view-pdf")
    @app.route("/reports/view/executive_demand_report.pdf")
    @app.route("/reports/view/Executive_Demand_Report.pdf")
    def root_view_pdf():
        from src.api.routes import view_executive_pdf_inline
        return view_executive_pdf_inline()

    @app.route("/download-quality-report")
    @app.route("/reports/data_quality_report.md")
    @app.route("/reports/Data_Quality_Audit_Report.md")
    def root_download_quality():
        from src.api.routes import download_data_quality_report
        return download_data_quality_report()

    @app.route("/download-quality-csv")
    @app.route("/reports/data_quality_report.csv")
    def root_download_quality_csv():
        from src.api.routes import download_data_quality_csv
        return download_data_quality_csv()

    @app.route("/download-dataset-csv")
    @app.route("/reports/retail_dataset_clean.csv")
    def root_download_dataset_csv():
        from src.api.routes import export_clean_dataset_csv
        return export_clean_dataset_csv()

    @app.route("/download-evaluation-csv")
    @app.route("/reports/model_evaluation_report.csv")
    def root_download_evaluation_csv():
        from src.api.routes import export_evaluation_report_csv
        return export_evaluation_report_csv()

    @app.route("/download-comparison-csv")
    @app.route("/reports/model_comparison_leaderboard.csv")
    def root_download_comparison_csv():
        from src.api.routes import export_comparison_leaderboard_csv
        return export_comparison_leaderboard_csv()

    @app.route("/download-experiments-csv")
    @app.route("/reports/experiment_history_report.csv")
    def root_download_experiments_csv():
        from src.api.routes import export_experiments_history_csv
        return export_experiments_history_csv()

    # Prevent stale browser caching of frontend static assets
    @app.after_request
    def set_cache_headers(response):
        from flask import request
        if request.path.endswith((".html", ".js", ".css")) or request.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # Initialize database tables on first request
    @app.before_request
    def init_db():
        if not hasattr(app, "_db_initialized"):
            try:
                from src.db import create_tables
                create_tables()
                app._db_initialized = True
            except Exception as e:
                logger.warning("Database table creation skipped: %s", e)
                app._db_initialized = True

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    )
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
