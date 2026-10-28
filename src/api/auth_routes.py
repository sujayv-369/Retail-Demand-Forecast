"""
Authentication API routes — login, signup, session check, and logout.
Uses bcrypt for password hashing and Flask sessions for state.
"""

from __future__ import annotations
import logging
from datetime import datetime

import bcrypt
from flask import Blueprint, request, jsonify, session
from sqlalchemy import text

from src.db import get_engine

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/session", methods=["GET"])
def check_session():
    """Return current session info if user is logged in."""
    if "user_id" in session:
        return jsonify({
            "authenticated": True,
            "user_id": session["user_id"],
            "username": session["username"],
            "role": session["role"],
        })
    return jsonify({"authenticated": False})


@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate a user with username and password."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, username, password_hash, role FROM users WHERE username = :u"),
                {"u": username}
            ).fetchone()

        if not row:
            return jsonify({"error": "Invalid username or password."}), 401

        user_id, db_username, password_hash, role = row

        # Verify password
        if not bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
            return jsonify({"error": "Invalid username or password."}), 401

        # Set session
        session["user_id"] = user_id
        session["username"] = db_username
        session["role"] = role

        return jsonify({
            "success": True,
            "user_id": user_id,
            "username": db_username,
            "role": role,
        })

    except Exception as e:
        logger.error("Login error: %s", e)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


@auth_bp.route("/signup", methods=["POST"])
def signup():
    """Register a new user (always assigned 'user' role → Retailer View)."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400

    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400

    # Prevent creating another 'Admin'
    if username.lower() == "admin":
        return jsonify({"error": "This username is reserved."}), 400

    try:
        engine = get_engine()

        # Check for duplicate
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT id FROM users WHERE username = :u"),
                {"u": username}
            ).fetchone()

        if exists:
            return jsonify({"error": "Username already taken. Please choose another."}), 409

        # Hash password and insert
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (:u, :p, 'user', :now)
                """),
                {"u": username, "p": pw_hash, "now": datetime.utcnow()}
            )

        # Auto-login after signup
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, username, role FROM users WHERE username = :u"),
                {"u": username}
            ).fetchone()

        session["user_id"] = row[0]
        session["username"] = row[1]
        session["role"] = row[2]

        return jsonify({
            "success": True,
            "user_id": row[0],
            "username": row[1],
            "role": row[2],
        }), 201

    except Exception as e:
        logger.error("Signup error: %s", e)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """Clear the user session."""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."})


@auth_bp.route("/profile", methods=["GET"])
def get_profile():
    """Retrieve profile information for currently authenticated user."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT id, username, role, phone, email, business_name, created_at
                    FROM users WHERE id = :uid
                """),
                {"uid": session["user_id"]}
            ).fetchone()

        if not row:
            return jsonify({"error": "User not found."}), 404

        uid, username, role, phone, email, business_name, created_at = row
        return jsonify({
            "success": True,
            "profile": {
                "user_id": uid,
                "username": username,
                "role": role,
                "phone": phone or "",
                "email": email or "",
                "business_name": business_name or "",
                "created_at": created_at.strftime("%Y-%m-%d %H:%M") if created_at else None,
            }
        })
    except Exception as e:
        logger.error("Error fetching profile: %s", e)
        return jsonify({"error": "Could not fetch profile."}), 500


@auth_bp.route("/profile", methods=["POST"])
def update_profile():
    """Update profile information (phone, email, business name, and optional password change)."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    phone = (data.get("phone") or "").strip()
    email = (data.get("email") or "").strip()
    business_name = (data.get("business_name") or "").strip()
    current_password = data.get("current_password") or ""
    new_password = data.get("new_password") or ""

    user_id = session["user_id"]

    try:
        engine = get_engine()

        # Handle optional password change
        if new_password:
            if len(new_password) < 4:
                return jsonify({"error": "New password must be at least 4 characters long."}), 400
            if not current_password:
                return jsonify({"error": "Please provide your current password to set a new password."}), 400

            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT password_hash FROM users WHERE id = :uid"),
                    {"uid": user_id}
                ).fetchone()

            if not row or not bcrypt.checkpw(current_password.encode("utf-8"), row[0].encode("utf-8")):
                return jsonify({"error": "Incorrect current password."}), 400

            new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE users SET password_hash = :ph WHERE id = :uid"),
                    {"ph": new_hash, "uid": user_id}
                )

        # Update profile fields
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE users
                    SET phone = :phone, email = :email, business_name = :bn
                    WHERE id = :uid
                """),
                {"phone": phone, "email": email, "bn": business_name, "uid": user_id}
            )

        return jsonify({
            "success": True,
            "message": "Profile updated successfully.",
            "profile": {
                "user_id": user_id,
                "username": session["username"],
                "role": session["role"],
                "phone": phone,
                "email": email,
                "business_name": business_name,
            }
        })

    except Exception as e:
        logger.error("Error updating profile: %s", e)
        return jsonify({"error": "Failed to update profile."}), 500
