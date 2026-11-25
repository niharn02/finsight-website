"""Request-scoped authenticated user identity for FinSight database adapters."""

from contextvars import ContextVar
from contextlib import contextmanager

_current_user_id = ContextVar("finsight_user_id", default=None)

def get_current_user_id() -> str:
    """Return the identity installed by the verified-auth request wrapper."""
    user_id = _current_user_id.get()
    if not user_id:
        raise RuntimeError("No verified FinSight user is active for this operation.")
    return user_id

@contextmanager
def user_context(user_id: str):
    """Set and reliably restore the authenticated identity for this context."""
    token = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(token)
