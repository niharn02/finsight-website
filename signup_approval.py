"""Server-side signup approval and administrator endpoints for FinSight."""
import os
import psycopg
from uuid import UUID
from flask import Blueprint, request, jsonify
from auth import get_supabase_user, admin_ids, SupabaseAuthConfigurationError

approval_bp = Blueprint("signup_approval", __name__)

def _db():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(url)

def _admins():
    return admin_ids()

def _verified_user():
    scheme, sep, token = request.headers.get("Authorization", "").partition(" ")
    if not sep or scheme.lower() != "bearer":
        return None, (jsonify(error="Authentication required"), 401)
    try:
        return get_supabase_user(token.strip()), None
    except ConnectionError:
        return None, (jsonify(error="Authentication service unavailable"), 503)
    except SupabaseAuthConfigurationError:
        # A server misconfiguration is not a bad login; do not report it as one.
        return None, (jsonify(error="Authentication service is not configured"), 503)
    except Exception:
        return None, (jsonify(error="Invalid authentication session"), 401)

def _status(user_id):
    # Admin identities are explicitly configured on the server, not self-appointed.
    if user_id in _admins():
        return "approved"
    with _db() as conn:
        row = conn.execute("SELECT status FROM public.finsight_access_requests WHERE user_id=%s", (user_id,)).fetchone()
    return row[0] if row else "pending"

@approval_bp.get("/api/access/status")
def access_status():
    user, error = _verified_user()
    if error: return error
    try:
        status = _status(user["id"])
    except Exception:
        return jsonify(error="Access status could not be checked"), 503
    return jsonify(status=status, is_admin=user["id"] in _admins())

@approval_bp.post("/api/access/request")
def access_request():
    user, error = _verified_user()
    if error: return error
    if user["id"] in _admins():
        return jsonify(status="approved")
    try:
        with _db() as conn:
            conn.execute("""
                INSERT INTO public.finsight_access_requests(user_id,email,status)
                VALUES (%s,%s,'pending')
                ON CONFLICT(user_id) DO UPDATE SET email=EXCLUDED.email
                WHERE public.finsight_access_requests.status='pending'
            """, (user["id"], user.get("email")))
    except Exception:
        return jsonify(error="Signup request could not be saved"), 503
    return jsonify(status=_status(user["id"])), 200

@approval_bp.get("/api/admin/access-requests")
def admin_requests():
    user, error = _verified_user()
    if error: return error
    if user["id"] not in _admins(): return jsonify(error="Forbidden"), 403
    try:
        with _db() as conn:
            rows = conn.execute("SELECT user_id,email,status,created_at,updated_at FROM public.finsight_access_requests ORDER BY created_at DESC").fetchall()
        return jsonify(requests=[dict(user_id=str(r[0]),email=r[1],status=r[2],created_at=str(r[3]),updated_at=str(r[4])) for r in rows])
    except Exception:
        return jsonify(error="Access requests could not be loaded"), 503

@approval_bp.post("/api/admin/access-requests/<user_id>/<action>")
def admin_action(user_id, action):
    actor, error = _verified_user()
    if error: return error
    if actor["id"] not in _admins(): return jsonify(error="Forbidden"), 403
    if action not in {"approve","reject","revoke"}: return jsonify(error="Invalid action"), 400
    # Validate the route UUID before using it in database operations.
    try:
        target_id = str(UUID(user_id))
    except (ValueError, TypeError, AttributeError):
        return jsonify(error="Invalid user ID"), 400
    new_status = {"approve":"approved","reject":"rejected","revoke":"revoked"}[action]
    try:
        with _db() as conn:
            with conn.transaction():
                row=conn.execute("SELECT status FROM public.finsight_access_requests WHERE user_id=%s FOR UPDATE",(target_id,)).fetchone()
                if not row: return jsonify(error="Request not found"),404
                conn.execute("UPDATE public.finsight_access_requests SET status=%s,updated_at=now() WHERE user_id=%s",(new_status,target_id))
                conn.execute("INSERT INTO public.finsight_access_audit(actor_user_id,target_user_id,action,old_status,new_status) VALUES (%s,%s,%s,%s,%s)",(actor["id"],target_id,action,row[0],new_status))
                conn.execute("INSERT INTO public.finsight_access_notifications(user_id,message) VALUES (%s,%s)",(target_id,"Your FinSight access status is now "+new_status+"."))
        return jsonify(status=new_status)
    except Exception:
        return jsonify(error="The access status could not be updated"), 503

@approval_bp.get("/api/access/notifications")
def access_notifications():
    user, error = _verified_user()
    if error: return error
    try:
        with _db() as conn:
            rows=conn.execute("SELECT notification_id,message,created_at,read_at FROM public.finsight_access_notifications WHERE user_id=%s ORDER BY created_at DESC LIMIT 50",(user["id"],)).fetchall()
        return jsonify(notifications=[dict(id=r[0],message=r[1],created_at=str(r[2]),read_at=str(r[3]) if r[3] else None) for r in rows])
    except Exception:
        return jsonify(error="Notifications could not be loaded"), 503

# Permit requests from the static Cloudflare site and the dashboard previews.
_ALLOWED_ORIGINS = {
    "https://finsight-website.pages.dev",
    "https://finsight-dashboard-xtk6.onrender.com",
    "https://finsight-dashboard-pr-3.onrender.com",
}

# Local development origins are trusted only when FINSIGHT_ALLOW_LOCAL_ORIGINS=1
# (never set this on Render). It is read on every request.
_LOCAL_ORIGINS = {"http://localhost:5500", "http://127.0.0.1:5500"}


def _origin_allowed(origin):
    if origin in _ALLOWED_ORIGINS:
        return True
    return os.getenv("FINSIGHT_ALLOW_LOCAL_ORIGINS", "").strip() == "1" and origin in _LOCAL_ORIGINS


@approval_bp.after_request
def _approval_cors(response):
    origin = request.headers.get("Origin", "")
    if _origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Vary"] = "Origin"
    return response