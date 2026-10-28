"""
Database connection management and schema for evaluation/forecast results.
Uses the project's existing MySQL database.
"""

import json
import logging

import bcrypt
from datetime import datetime

from sqlalchemy import (
    create_engine, text, Column, Integer, String, Float,
    DateTime, Text, JSON, MetaData, Table, Index,
)
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_URL

logger = logging.getLogger(__name__)

_engine = None
_Session = None


def get_engine():
    """Return a singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session():
    """Return a new session."""
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine())
    return _Session()


# ── Additional tables for evaluation & forecasting ─────────────────
metadata_obj = MetaData()

evaluation_runs = Table(
    "evaluation_runs", metadata_obj,
    Column("run_id", String(64), primary_key=True),
    Column("model_name", String(128), nullable=False),
    Column("eval_cutoff_date", String(16), nullable=False),
    Column("test_start_date", String(16), nullable=False),
    Column("test_end_date", String(16), nullable=False),
    Column("config_json", Text),
    Column("created_at", DateTime, default=datetime.utcnow),
)

evaluation_metrics = Table(
    "evaluation_metrics", metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("model_type", String(32), nullable=False),  # catboost, naive, ma4
    Column("scope_type", String(32), nullable=False),   # overall, product, category, horizon
    Column("scope_value", String(128)),
    Column("horizon_week", Integer),
    Column("mae", Float),
    Column("wape", Float),
    Column("n_observations", Integer),
)

forecast_runs = Table(
    "forecast_runs", metadata_obj,
    Column("run_id", String(64), primary_key=True),
    Column("model_name", String(128), nullable=False),
    Column("forecast_origin_date", String(16), nullable=False),
    Column("horizon_weeks", Integer, default=4),
    Column("created_at", DateTime, default=datetime.utcnow),
)

forecast_results = Table(
    "forecast_results", metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("product_id", String(32), nullable=False),
    Column("store_id", String(32), nullable=False),
    Column("sales_channel", String(32), nullable=False),
    Column("forecast_week_date", String(16), nullable=False),
    Column("horizon_step", Integer, nullable=False),
    Column("predicted_units", Float),
    Column("product_name", String(256)),
    Column("category", String(128)),
)

reliability_flags = Table(
    "reliability_flags", metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("product_id", String(32), nullable=False),
    Column("store_id", String(32)),
    Column("sales_channel", String(32)),
    Column("flag_type", String(64), nullable=False),
    Column("flag_reason", Text),
    Column("severity", String(16)),  # low, medium, high
)

gemini_interactions = Table(
    "gemini_interactions", metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_question", Text),
    Column("tools_called", Text),
    Column("grounding_status", String(32)),
    Column("response_text", Text),
    Column("created_at", DateTime, default=datetime.utcnow),
)

gemini_feedback = Table(
    "gemini_feedback", metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("message_id", String(64)),
    Column("user_query", Text),
    Column("response_text", Text),
    Column("feedback_type", String(32)),
    Column("feedback_score", Integer),
    Column("comment", Text),
    Column("created_at", DateTime, default=datetime.utcnow),
)

app_settings = Table(
    "app_settings", metadata_obj,
    Column("key", String(64), primary_key=True),
    Column("value", Text, nullable=False),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

users = Table(
    "users", metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(128), nullable=False, unique=True),
    Column("password_hash", String(256), nullable=False),
    Column("role", String(32), nullable=False, default="user"),  # 'admin' or 'user'
    Column("phone", String(50), nullable=True),
    Column("email", String(120), nullable=True),
    Column("business_name", String(150), nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow),
)


def migrate_users_table():
    """Ensure phone, email, and business_name columns exist in users table."""
    try:
        engine = get_engine()
        cols = [
            ("phone", "VARCHAR(50)"),
            ("email", "VARCHAR(120)"),
            ("business_name", "VARCHAR(150)"),
        ]
        with engine.begin() as conn:
            for col_name, col_type in cols:
                try:
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type} NULL"))
                except Exception:
                    pass  # column already exists
    except Exception as e:
        logger.debug("Users table migration check: %s", e)


def create_tables():
    """Create all additional tables (idempotent)."""
    engine = get_engine()
    metadata_obj.create_all(engine)
    migrate_users_table()
    logger.info("Database tables created / verified.")
    seed_admin_user()


def seed_admin_user():
    """Ensure the Admin user exists with default credentials."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id FROM users WHERE username = :u"),
                {"u": "Admin"}
            ).fetchone()
            if row:
                return  # Admin already exists
        # Hash the default password
        pw_hash = bcrypt.hashpw("123456".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (:u, :p, 'admin', :now)
                """),
                {"u": "Admin", "p": pw_hash, "now": datetime.utcnow()}
            )
        logger.info("Admin user seeded successfully.")
    except Exception as e:
        logger.warning("Could not seed admin user: %s", e)


def get_app_setting(key: str, default: str | None = None) -> str | None:
    """Retrieve an application setting value by key."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT value FROM app_settings WHERE `key` = :k"),
                {"k": key}
            ).fetchone()
            if row:
                return row[0]
    except Exception as e:
        logger.debug("Could not read app_setting '%s': %s", key, e)
    return default


def set_app_setting(key: str, value: str):
    """Upsert an application setting value."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO app_settings (`key`, `value`, `updated_at`)
                VALUES (:k, :v, :now)
                ON DUPLICATE KEY UPDATE `value` = :v, `updated_at` = :now
            """), {"k": key, "v": value, "now": datetime.utcnow()})
    except Exception as e:
        logger.warning("Could not set app_setting '%s': %s", key, e)


def execute_read(query_str: str, params: dict | None = None):
    """Execute a read-only SQL query and return rows as list of dicts."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query_str), params or {})
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]

