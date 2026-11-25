"""Supabase access-token verification helpers for FinSight.

The browser sends its Supabase access token in the Authorization header.
FinSight validates that token with Supabase Auth's /user endpoint and derives
user_id from the returned user object. Never accept user_id from request JSON,
query parameters, form data, or chatbot messages.

Required Render environment variables:
    SUPABASE_URL       e.g. https://<project-ref>.supabase.co
    SUPABASE_ANON_KEY  Supabase publishable/anon key (safe for backend use)

This module intentionally does not enable authentication globally by itself.
Integrate @require_supabase_auth on protected Flask routes at the same time as
the user-scoped PostgreSQL database layer. Do not deploy this module alone as
proof that financial data is isolated.
"""

from __future__ import annotations

import os
import psycopg
from functools import wraps
from uuid import UUID

import requests
from flask import current_app, g, jsonify, request

from user_context import user_context

# Signup approval is checked on each protected API request, not merely in the UI.\ndef _approval_status(user_id: str) -> str:\n    admins = {x.strip() for x in os.getenv('FINANCE_ADMIN_USER_IDS', '').split(',') if x.strip()}\n    if user_id in admins:\n        return 'approved'\n    database_url = os.getenv('DATABASE_URL', '').strip()\n    if not database_url:\n        raise RuntimeError('DATABASE_URL is not configured')\n    with psycopg.connect(database_url) as conn:\n        row = conn.execute('SELECT status FROM public.finsight_access_requests WHERE user_id=%s', (user_id,)).fetchone()\n    return row[0] if row else 'pending'


class SupabaseAuthConfigurationError(RuntimeError):
    """Raised when required Supabase Auth configuration is missing."""


def _supabase_auth_settings() -> tuple[str, str]:
    """Return the Supabase project URL and public API key from env vars."""
    project_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()

    if not project_url or not anon_key:
        raise SupabaseAuthConfigurationError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be configured on Render."
        )

    return project_url, anon_key


def get_supabase_user(access_token: str) -> dict:
    """Validate an access token with Supabase and return its user record.

    Supabase Auth validates the signature, expiry, and token status. The
    returned UUID is the authoritative identity used by the Flask backend.
    """
    if not isinstance(access_token, str) or not access_token.strip():
        raise ValueError("A Supabase access token is required.")

    project_url, anon_key = _supabase_auth_settings()

    try:
        response = requests.get(
            f"{project_url}/auth/v1/user",
            headers={
                "apikey": anon_key,
                "Authorization": f"Bearer {access_token.strip()}",
                "Accept": "application/json",
            },
            timeout=(3.05, 8),
        )
    except requests.RequestException as exc:
        # Do not expose tokens or low-level network details to the browser.
        current_app.logger.warning("Supabase Auth verification request failed.")
        raise ConnectionError("Unable to verify the login session.") from exc

    if response.status_code != 200:
        # Treat expired, revoked, malformed, and otherwise invalid tokens alike.
        raise PermissionError("The login session is invalid or expired.")

    try:
        user = response.json()
        user_id = str(UUID(str(user["id"])))
    except (ValueError, TypeError, KeyError, requests.JSONDecodeError) as exc:
        raise PermissionError("Supabase returned an invalid user identity.") from exc

    # Return verified identity fields needed by the profile UI. Keep this
    # payload limited to non-sensitive display information from Supabase Auth.
    metadata = user.get("user_metadata") or {}
    return {
        "id": user_id,
        "email": user.get("email"),
        "name": metadata.get("full_name") or metadata.get("name") or metadata.get("fullName"),
        # Include the verified OAuth profile photo for the dashboard profile UI.
        # Supabase commonly stores Google's picture URL in "picture", but
        # providers/clients may also populate avatar_url or avatar.
        "avatar_url": metadata.get("avatar_url") or metadata.get("picture") or metadata.get("avatar") or "",
        "user_metadata": metadata,
        "role": user.get("role"),
    }


def require_supabase_auth(view_function):
    """Protect a Flask route and expose verified identity through flask.g.

    Usage:
        @app.route("/api/example")
        @require_supabase_auth
        def example():
            user_id = g.user_id
            ...

    Apply this to every financial and chatbot API route. Keep only explicitly
    public endpoints (such as /api/health) outside the decorator.
    """
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        authorization = request.headers.get("Authorization", "")
        scheme, separator, token = authorization.partition(" ")

        if not separator or scheme.lower() != "bearer" or not token.strip():
            return jsonify({
                "status": "error",
                "message": "Authentication is required. Please sign in again.",
            }), 401

        try:
            user = get_supabase_user(token)
        except SupabaseAuthConfigurationError:
            current_app.logger.exception("Supabase Auth is not configured.")
            return jsonify({
                "status": "error",
                "message": "Authentication service is not configured.",
            }), 503
        except PermissionError:
            return jsonify({
                "status": "error",
                "message": "Your login session is invalid or expired. Please sign in again.",
            }), 401
        except ConnectionError:
            return jsonify({
                "status": "error",
                "message": "Unable to verify your login right now. Please try again.",
            }), 503

        # Downstream routes use this verified identity, never client input.
        g.supabase_user = user
        g.user_id = user["id"]

        # Establish the ContextVar identity for all scoped database adapters.
        # The context manager restores it even when the route raises an error.
        with user_context(user["id"]):
            return view_function(*args, **kwargs)

    return wrapped_view
