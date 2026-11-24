
"""
FinSight conversation memory.

Initialises a separate SQLite database for LangGraph checkpoints.
This database stores conversation state, not financial transactions.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


# Locate the FinSight project root directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Keep chatbot checkpoints separate from the financial database.
CHECKPOINT_DB_PATH = BASE_DIR / "chatbot_checkpoints.sqlite"

# Maintain one shared connection and checkpointer for the application.
_connection = None
_checkpointer = None


def get_checkpointer():
    """
    Return the shared SQLite checkpointer.

    Initialise the database and its checkpoint tables on first use.
    """

    global _connection, _checkpointer

    # Create the connection only once.
    if _checkpointer is None:

        # Allow the connection to be used across application threads.
        _connection = sqlite3.connect(
            str(CHECKPOINT_DB_PATH),
            check_same_thread=False
        )

        # Create the LangGraph checkpoint manager.
        _checkpointer = SqliteSaver(_connection)

        # Create the required checkpoint tables if they do not exist.
        _checkpointer.setup()

    return _checkpointer
