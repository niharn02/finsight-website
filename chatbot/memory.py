"""PostgreSQL-backed LangGraph checkpoint storage for FinSight.

Checkpoint tables live in the dedicated ``langgraph`` schema. Create that
schema and grant the Render database role access using the supplied SQL setup
script before the first deployment. No SQLite checkpoints are imported.
"""

from __future__ import annotations
import os
import threading
import psycopg
from psycopg.rows import dict_row

# Require the checkpoint serializer to reject unapproved msgpack payload types.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
from langgraph.checkpoint.postgres import PostgresSaver

_lock = threading.Lock()
_connection = None
_checkpointer = None

def get_checkpointer() -> PostgresSaver:
    """Return one process-wide saver using a dedicated PostgreSQL connection."""
    global _connection, _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    with _lock:
        if _checkpointer is not None:
            return _checkpointer
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for PostgreSQL chat memory.")
        # Autocommit and dict_row are required by LangGraph's PostgresSaver.
        connection = psycopg.connect(
            database_url, autocommit=True, row_factory=dict_row
        )
        try:
            # The schema is created explicitly by the deployment SQL, not at app startup.
            connection.execute("SET search_path TO langgraph")
            saver = PostgresSaver(connection)
            saver.setup()
        except Exception:
            connection.close()
            raise
        _connection = connection
        _checkpointer = saver
        return _checkpointer



def initialise_checkpointer() -> None:
    """Initialize persistent checkpoint tables during backend startup."""
    # Fail startup if the configured database or schema cannot initialize memory.
    get_checkpointer()
