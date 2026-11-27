"""Secure, short-lived Supabase-to-Flask handoff for FinSight."""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request, make_response
import psycopg
from cryptography.fernet import Fernet, InvalidToken

from auth import get_supabase_user, admin_ids


handoff_bp = Blueprint("auth_handoff", __name__)

# Render pages and dashboard share an origin. Local testing is also retained.
LOCAL_FRONTEND_ORIGINS = {"http://localhost:5500", "http://127.0.0.1:5500"}
LOCAL_DASHBOARD_ORIGINS = {"http://localhost:5000", "http://127.0.0.1:5000"}

def _render_origin():
    return os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")

def _configured_origins():
    # Include Render's generated URL and any custom domain explicitly configured
    # by the operator. Do not trust arbitrary browser-supplied origins.
    origins = set()
    configured = os.getenv("FINSIGHT_PUBLIC_ORIGINS", "")
    origins.update(
        item.strip().rstrip("/")
        for item in configured.split(",")
        if item.strip()
    )
    render_origin = _render_origin()
    if render_origin:
        origins.add(render_origin)
    return origins

def _allowed_frontend_origins():
    return set(LOCAL_FRONTEND_ORIGINS) | _configured_origins()

def _allowed_dashboard_origins():
    return set(LOCAL_DASHBOARD_ORIGINS) | _configured_origins()


def _fernet():
    key = os.getenv("HANDOFF_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("HANDOFF_ENCRYPTION_KEY is not configured")
    return Fernet(key.encode())


def _db():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(url)


def _cors(response, origin=None):
    if origin in _allowed_frontend_origins() or origin in _allowed_dashboard_origins():
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization"
        )
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Vary"] = "Origin"
    return response


@handoff_bp.route("/auth/handoff/start", methods=["POST", "OPTIONS"])
def start_handoff():
    origin = request.headers.get("Origin", "")

    if request.method == "OPTIONS":
        return _cors(make_response("", 204), origin)

    if origin not in _allowed_frontend_origins():
        return _cors(jsonify(error="Origin not allowed"), origin), 403

    payload = request.get_json(silent=True) or {}
    access_token = payload.get("access_token")

    target_origin = payload.get(
        "return_origin",
        _render_origin() or "http://localhost:5000",
    )

    if (
        not isinstance(target_origin, str)
        or target_origin not in _allowed_dashboard_origins()
    ):
        return (
            _cors(
                jsonify(error="Dashboard destination not allowed"),
                origin,
            ),
            400,
        )

    try:
        user = get_supabase_user(access_token)

    except PermissionError as exc:
        return _cors(jsonify(error=str(exc)), origin), 401

    except (ConnectionError, RuntimeError):
        return (
            _cors(
                jsonify(error="Authentication verification unavailable"),
                origin,
            ),
            503,
        )

    except Exception:
        return (
            _cors(
                jsonify(error="Unable to verify authentication"),
                origin,
            ),
            400,
        )

    # Gate dashboard handoff until the account has explicitly been approved.
    if user["id"] not in admin_ids():
        try:
            with _db() as conn:
                status_row = conn.execute(
                    """
                    SELECT status
                    FROM public.finsight_access_requests
                    WHERE user_id=%s
                    """,
                    (user["id"],),
                ).fetchone()

                approval_status = (
                    status_row[0] if status_row else "pending"
                )

        except Exception as exc:
            # Log the server-side cause for diagnosis without exposing it to the client.
            import traceback
            print("Approval-status database query failed:", repr(exc), flush=True)
            traceback.print_exc()
            return (
                _cors(
                    jsonify(error="Approval service unavailable"),
                    origin,
                ),
                503,
            )

        if approval_status != "approved":
            return (
                _cors(
                    jsonify(
                        error=(
                            f"FinSight access is {approval_status}. "
                            "Please wait for administrator approval."
                        ),
                        access_status=approval_status,
                    ),
                    origin,
                ),
                403,
            )

    # Create a short-lived, single-use handoff.
    try:
        code = secrets.token_urlsafe(32)
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        encrypted_token = _fernet().encrypt(
            access_token.encode()
        ).decode()

        expiry = datetime.now(timezone.utc) + timedelta(seconds=90)

        with _db() as conn:
            conn.execute(
                "DELETE FROM finsight_auth_handoff WHERE expires_at < now()"
            )

            conn.execute(
                """
                INSERT INTO finsight_auth_handoff
                    (code_hash, user_id, encrypted_token, expires_at)
                VALUES (%s, %s, %s, %s)
                """,
                (code_hash, user["id"], encrypted_token, expiry),
            )

        return _cors(
            jsonify(
                redirect_url=(
                    f"{target_origin}/auth/handoff/consume?code={code}"
                )
            ),
            origin,
        )

    except Exception:
        import traceback
        traceback.print_exc()

        return (
            _cors(
                jsonify(
                    error="Sign-in handoff is temporarily unavailable"
                ),
                origin,
            ),
            503,
        )


@handoff_bp.route("/auth/handoff/exchange", methods=["POST", "OPTIONS"])
def exchange_handoff():
    origin = request.headers.get("Origin", "")

    if request.method == "OPTIONS":
        return _cors(make_response("", 204), origin)

    if origin not in _allowed_dashboard_origins():
        return _cors(jsonify(error="Origin not allowed"), origin), 403

    code = (request.get_json(silent=True) or {}).get("code", "")

    if not isinstance(code, str) or not code:
        return _cors(jsonify(error="Invalid handoff code"), origin), 400

    code_hash = hashlib.sha256(code.encode()).hexdigest()

    # DELETE ... RETURNING makes consumption atomic and strictly single-use.
    try:
        with _db() as conn:
            row = conn.execute(
                """
                DELETE FROM finsight_auth_handoff
                WHERE code_hash=%s AND expires_at > now()
                RETURNING encrypted_token, user_id
                """,
                (code_hash,),
            ).fetchone()

    except Exception:
        return (
            _cors(
                jsonify(
                    error="Sign-in handoff is temporarily unavailable"
                ),
                origin,
            ),
            503,
        )

    if not row:
        return (
            _cors(
                jsonify(error="Code expired or already used"),
                origin,
            ),
            401,
        )

    # Re-check approval when exchanging:
    # approval may have been revoked after handoff initiation.
    user_id = str(row[1])

    if user_id not in admin_ids():
        try:
            with _db() as conn:
                approval_row = conn.execute(
                    """
                    SELECT status
                    FROM public.finsight_access_requests
                    WHERE user_id=%s
                    """,
                    (user_id,),
                ).fetchone()

            if not approval_row or approval_row[0] != "approved":
                return (
                    _cors(
                        jsonify(error="Access is no longer approved"),
                        origin,
                    ),
                    403,
                )

        except Exception:
            return (
                _cors(
                    jsonify(error="Approval service unavailable"),
                    origin,
                ),
                503,
            )

    try:
        token = _fernet().decrypt(row[0].encode()).decode()

    except InvalidToken:
        return (
            _cors(
                jsonify(error="Handoff could not be completed"),
                origin,
            ),
            500,
        )

    return _cors(jsonify(access_token=token), origin)


@handoff_bp.route("/auth/handoff/consume", methods=["GET"])
def consume_page():
    # No token in URL; the page posts the one-time code
    # and then stores the token locally.
    page = '''<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>FinSight sign in</title>
<p id="status">Completing sign-in…</p>
<script>
(async()=>{
    const status=document.getElementById('status');
    const code=new URLSearchParams(location.search).get('code');

    history.replaceState({},'',location.pathname);

    if(!code){
        status.textContent='Missing sign-in code';
        return;
    }

    try{
        const r=await fetch('/auth/handoff/exchange',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({code})
        });

        const d=await r.json();

        if(!r.ok){
            throw new Error(d.error||'Sign-in failed');
        }

        sessionStorage.setItem(
            'finsight_access_token',
            d.access_token
        );

        location.replace('/dashboard');
    }catch(e){
        status.textContent=
            e.message+' — return to FinSight sign-in and try again.';
    }
})();
</script>'''

    response = make_response(page)

    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"

    return response