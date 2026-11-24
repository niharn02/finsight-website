"""PostgreSQL connection helpers for FinSight.

This module is the connection/security foundation for the fresh Supabase
PostgreSQL implementation. It deliberately does not migrate SQLite queries.
Every application query must run inside ``user_connection(user_id)`` so that
Supabase RLS policies evaluate ``auth.uid()`` for the authenticated user.

Required Render environment variable:
    DATABASE_URL=postgresql://... (Supabase Session Pooler, port 5432)

Never put DATABASE_URL in frontend code or commit it to Git.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator
from uuid import UUID

import psycopg
from psycopg.rows import dict_row


class DatabaseConfigurationError(RuntimeError):
    """Raised when the PostgreSQL connection is not configured."""


def _database_url() -> str:
    """Read the backend-only Supabase connection string."""
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. Add the Supabase Session Pooler "
            "connection string to the Render environment variables."
        )
    return value


def _normalise_user_id(user_id: str | UUID) -> str:
    """Validate the authenticated Supabase user UUID before opening a session."""
    try:
        return str(UUID(str(user_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("A valid authenticated user_id is required.") from exc


@contextmanager
def user_connection(user_id: str | UUID) -> Iterator[psycopg.Connection]:
    """Yield a transaction-scoped connection with Supabase RLS identity.

    The user ID must come from a verified Supabase JWT, never from request
    JSON, query parameters, chatbot text, or a browser-generated identifier.

    SET LOCAL ROLE and set_config(..., true) are transaction-scoped. This
    makes the existing ``auth.uid()``-based RLS policies apply to this user's
    queries when connecting with a database role permitted to assume
    ``authenticated`` (for example, the Supabase postgres backend role).
    """
    verified_user_id = _normalise_user_id(user_id)

    with psycopg.connect(_database_url(), row_factory=dict_row) as conn:
        # Use Supabase's authenticated database role so RLS is evaluated.
        conn.execute("SET LOCAL ROLE authenticated")

        # auth.uid() reads this transaction-local JWT subject claim.
        conn.execute(
            "SELECT set_config('request.jwt.claim.sub', %s, true)",
            (verified_user_id,),
        )

        # The connection context commits on success and rolls back on error.
        yield conn


@contextmanager
def health_connection() -> Iterator[psycopg.Connection]:
    """Yield a minimal connection for non-user-scoped health checks only.

    Do not use this helper for financial data or chatbot operations.
    """
    with psycopg.connect(_database_url(), row_factory=dict_row) as conn:
        yield conn
