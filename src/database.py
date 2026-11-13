"""
database.py - SQLite storage for conversations, messages and tickets.

init_db()                  creates the tables (safe to run many times)
create_conversation()      starts a new conversation, returns its id
add_message(...)           saves one chat message
get_messages(id)           all messages of a conversation (for display)
get_history(id)            role/content pairs (for the LLM prompt)
list_conversations()       recent conversations (for the sidebar)

ticket_manager.py uses get_connection() to work with the tickets table.
Every function accepts an optional db_path so tests can use a temporary file.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id),
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    status           TEXT,
    sources_json     TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER REFERENCES conversations(id),
    requester_name   TEXT NOT NULL,
    contact          TEXT NOT NULL,
    category         TEXT NOT NULL,
    description      TEXT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""


def now():
    """Current local time as text, e.g. 2026-11-13T10:22:31."""
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_connection(db_path=None):
    """Open a database connection; commit on success, roll back on error, always close."""
    path = Path(db_path) if db_path is not None else config.DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path=None):
    """Create all tables if they do not exist yet."""
    with get_connection(db_path) as connection:
        connection.executescript(SCHEMA)


def create_conversation(db_path=None):
    """Start a new conversation and return its id."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO conversations (created_at) VALUES (?)", (now(),)
        )
        return cursor.lastrowid


def add_message(conversation_id, role, content, status=None, sources=None, db_path=None):
    """Save one message. Returns the new message id.

    role    : "user" or "assistant"
    status  : the pipeline status for assistant messages (optional)
    sources : the list of source dictionaries for assistant messages (optional)
    """
    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO messages "
            "(conversation_id, role, content, status, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, status, sources_json, now()),
        )
        return cursor.lastrowid


def get_messages(conversation_id, db_path=None):
    """Return all messages of a conversation, oldest first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT role, content, status, sources_json, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "status": row["status"],
            "sources": json.loads(row["sources_json"]) if row["sources_json"] else [],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_history(conversation_id, db_path=None):
    """Return [{"role", "content"}, ...] in the format prompt_builder expects."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in get_messages(conversation_id, db_path)
    ]


def list_conversations(limit=20, db_path=None):
    """Return recent conversations that have messages, newest first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT c.id, c.created_at, "
            "  (SELECT m.content FROM messages m "
            "   WHERE m.conversation_id = c.id AND m.role = 'user' "
            "   ORDER BY m.id LIMIT 1) AS first_question "
            "FROM conversations c "
            "WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id) "
            "ORDER BY c.id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "title": (row["first_question"] or "Conversation")[:50],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.database
# (uses a temporary database file, so it never touches the real one)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temp_folder:
        test_db = Path(temp_folder) / "test.db"

        init_db(test_db)
        init_db(test_db)  # running twice must be harmless

        with get_connection(test_db) as connection:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        print("Tables:", sorted(tables))
        assert {"conversations", "messages", "tickets"} <= tables

        first = create_conversation(test_db)
        empty = create_conversation(test_db)
        assert first != empty

        sources = [{"source": "a.txt", "page": 1, "label": "sample", "score": 0.5}]
        add_message(first, "user", "I can't connect to Wi-Fi caf\u00e9", db_path=test_db)
        add_message(first, "assistant", "Try these steps...", "answered", sources, test_db)

        messages = get_messages(first, test_db)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "I can't connect to Wi-Fi caf\u00e9"
        assert messages[0]["sources"] == []
        assert messages[1]["status"] == "answered"
        assert messages[1]["sources"] == sources

        history = get_history(first, test_db)
        assert history[1] == {"role": "assistant", "content": "Try these steps..."}

        conversations = list_conversations(db_path=test_db)
        print("Conversations:", conversations)
        assert [c["id"] for c in conversations] == [first]   # the empty one is hidden
        assert conversations[0]["title"].startswith("I can't connect")

        try:
            add_message(999, "user", "orphan", db_path=test_db)
            raise AssertionError("Foreign key should have rejected this")
        except sqlite3.IntegrityError as error:
            print("Foreign key enforced:", error)

    print("Database checks passed.")