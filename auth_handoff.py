"""Secure, short-lived Supabase-to-Flask handoff for FinSight."""
from __future__ import annotations
import hashlib, os, secrets
from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, make_response
import psycopg
from cryptography.fernet import Fernet, InvalidToken
from auth import get_supabase_user

handoff_bp = Blueprint("auth_handoff", __name__)
CLOUDFLARE_ORIGIN = "https://finsight-website.pages.dev"
RENDER_ORIGIN = "https://finsight-dashboard-xtk6.onrender.com"
ALLOWED_DASHBOARD_ORIGINS = {
    RENDER_ORIGIN,
    "https://finsight-dashboard-pr-3.onrender.com",
}


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


def _cors(response, origin=CLOUDFLARE_ORIGIN):
    # Restrict browser access to the static authentication site.
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Vary"] = "Origin"
    return response


@handoff_bp.route("/auth/handoff/start", methods=["POST", "OPTIONS"])
def start_handoff():
    if request.method == "OPTIONS":
        return _cors(make_response("", 204))
    if request.headers.get("Origin") != CLOUDFLARE_ORIGIN:
        return _cors(jsonify(error="Origin not allowed")), 403
    payload = request.get_json(silent=True) or {}
    access_token = payload.get("access_token")
    target_origin = payload.get("return_origin", RENDER_ORIGIN)
    # Reject, rather than silently redirecting, untrusted destinations.
    if not isinstance(target_origin, str) or target_origin not in ALLOWED_DASHBOARD_ORIGINS:
        return _cors(jsonify(error="Dashboard destination not allowed")), 400
    try:
        user = get_supabase_user(access_token)
    except PermissionError as exc:
        return _cors(jsonify(error=str(exc))), 401
    except (ConnectionError, RuntimeError) as exc:
        return _cors(jsonify(error="Authentication verification unavailable")), 503
    except Exception:
        return _cors(jsonify(error="Unable to verify authentication")), 400

    # Gate dashboard handoff until the account has explicitly been approved.
    admins = {x.strip() for x in os.getenv('FINANCE_ADMIN_USER_IDS', '').split(',') if x.strip()}
    if user['id'] not in admins:
        try:
            with _db() as conn:
                status_row = conn.execute('SELECT status FROM public.finsight_access_requests WHERE user_id=%s', (user['id'],)).fetchone()
                approval_status = status_row[0] if status_row else 'pending'
        except Exception:
            return _cors(jsonify(error='Approval service unavailable')), 503
        if approval_status != 'approved':
            return _cors(jsonify(error='FinSight access is '+approval_status+'. Please wait for administrator approval.', access_status=approval_status)), 403


    # Code is random, short-lived and stored by hash; the token is encrypted at rest.
    # Persist a short-lived handoff only when server dependencies are available.
    try:
        code = secrets.token_urlsafe(32)
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        encrypted_token = _fernet().encrypt(access_token.encode()).decode()
        expiry = datetime.now(timezone.utc) + timedelta(seconds=90)
        with _db() as conn:
            conn.execute("DELETE FROM finsight_auth_handoff WHERE expires_at < now()")
            conn.execute(
                "INSERT INTO finsight_auth_handoff(code_hash,user_id,encrypted_token,expires_at) VALUES (%s,%s,%s,%s)",
                (code_hash, user["id"], encrypted_token, expiry),
            )
        return _cors(jsonify(redirect_url=f"{target_origin}/auth/handoff/consume?code={code}"))
    except Exception:
        return _cors(jsonify(error="Sign-in handoff is temporarily unavailable")), 503


@handoff_bp.route("/auth/handoff/exchange", methods=["POST", "OPTIONS"])
def exchange_handoff():
    if request.method == "OPTIONS":
        return _cors(make_response("", 204), request.headers.get("Origin", ""))
    if request.headers.get("Origin") not in ALLOWED_DASHBOARD_ORIGINS:
        return _cors(jsonify(error="Origin not allowed"), request.headers.get("Origin", "")), 403
    code = (request.get_json(silent=True) or {}).get("code", "")
    if not isinstance(code, str) or not code:
        return _cors(jsonify(error="Invalid handoff code"), request.headers.get("Origin", "")), 400
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    # DELETE ... RETURNING makes consumption atomic and strictly single-use.
    try:
        with _db() as conn:
            row = conn.execute(
                "DELETE FROM finsight_auth_handoff WHERE code_hash=%s AND expires_at > now() RETURNING encrypted_token,user_id",
                (code_hash,),
            ).fetchone()
    except Exception:
        return _cors(jsonify(error="Sign-in handoff is temporarily unavailable"), request.headers.get("Origin", "")), 503
    if not row:
        return _cors(jsonify(error="Code expired or already used"), request.headers.get("Origin", "")), 401
    # Re-check approval when exchanging: approval may have been revoked after start.
    user_id = str(row[1])
    admins = {x.strip() for x in os.getenv("FINANCE_ADMIN_USER_IDS", "").split(",") if x.strip()}
    if user_id not in admins:
        try:
            with _db() as conn:
                approval_row = conn.execute(
                    "SELECT status FROM public.finsight_access_requests WHERE user_id=%s",
                    (user_id,),
                ).fetchone()
            if not approval_row or approval_row[0] != "approved":
                return _cors(jsonify(error="Access is no longer approved"), request.headers.get("Origin", "")), 403
        except Exception:
            return _cors(jsonify(error="Approval service unavailable"), request.headers.get("Origin", "")), 503
    try:
        token = _fernet().decrypt(row[0].encode()).decode()
    except InvalidToken:
        return _cors(jsonify(error="Handoff could not be completed"), request.headers.get("Origin", "")), 500
    return _cors(jsonify(access_token=token), request.headers.get("Origin", ""))


@handoff_bp.route("/auth/handoff/consume", methods=["GET"])
def consume_page():
    # No token in URL; the page posts the one-time code and then stores token locally.
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>FinSight sign in</title><p id="status">Completing sign-in…</p><script>
(async()=>{const status=document.getElementById('status');const code=new URLSearchParams(location.search).get('code');history.replaceState({},'',location.pathname);if(!code){status.textContent='Missing sign-in code';return;}try{const r=await fetch('/auth/handoff/exchange',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});const d=await r.json();if(!r.ok)throw new Error(d.error||'Sign-in failed');sessionStorage.setItem('finsight_access_token',d.access_token);location.replace('/');}catch(e){status.textContent=e.message+' — return to FinSight sign-in and try again.';}})();
</script>'''
    response = make_response(page)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response
