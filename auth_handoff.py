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


def _cors(response):
    # Restrict browser access to the static authentication site.
    response.headers["Access-Control-Allow-Origin"] = CLOUDFLARE_ORIGIN
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
    try:
        user = get_supabase_user(access_token)
    except PermissionError as exc:
        return _cors(jsonify(error=str(exc))), 401
    except (ConnectionError, RuntimeError) as exc:
        return _cors(jsonify(error="Authentication verification unavailable")), 503
    except Exception:
        return _cors(jsonify(error="Unable to verify authentication")), 400

    # Code is random, short-lived and stored by hash; the token is encrypted at rest.
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
    return _cors(jsonify(redirect_url=f"{RENDER_ORIGIN}/auth/handoff/consume?code={code}"))


@handoff_bp.route("/auth/handoff/exchange", methods=["POST", "OPTIONS"])
def exchange_handoff():
    if request.method == "OPTIONS":
        return _cors(make_response("", 204))
    if request.headers.get("Origin") != RENDER_ORIGIN:
        return _cors(jsonify(error="Origin not allowed")), 403
    code = (request.get_json(silent=True) or {}).get("code", "")
    if not isinstance(code, str) or not code:
        return _cors(jsonify(error="Invalid handoff code")), 400
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    # DELETE ... RETURNING makes consumption atomic and strictly single-use.
    with _db() as conn:
        row = conn.execute(
            "DELETE FROM finsight_auth_handoff WHERE code_hash=%s AND expires_at > now() RETURNING encrypted_token",
            (code_hash,),
        ).fetchone()
    if not row:
        return _cors(jsonify(error="Code expired or already used")), 401
    try:
        token = _fernet().decrypt(row[0].encode()).decode()
    except InvalidToken:
        return _cors(jsonify(error="Handoff could not be completed")), 500
    return _cors(jsonify(access_token=token))


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
