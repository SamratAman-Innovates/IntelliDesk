"""
ticket_manager.py - Support ticket rules: create, read, update, statistics.

All SQL for the tickets table lives here. app.py and the admin dashboard
call these functions and never touch the database directly.
Every function accepts an optional db_path so tests can use a temporary file.
"""

from src import config
from src.database import get_connection, now


def create_ticket(
    requester_name, contact, category, description, conversation_id=None, db_path=None
):
    """Create a ticket with status "Open" and return its id."""
    requester_name = requester_name.strip()
    contact = contact.strip()
    description = description.strip()

    if not requester_name:
        raise ValueError("Name is required.")
    if not contact:
        raise ValueError("Contact (email or phone) is required.")
    if not description:
        raise ValueError("A description of the problem is required.")
    if category not in config.TICKET_CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    timestamp = now()
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO tickets "
            "(conversation_id, requester_name, contact, category, description, "
            " status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                requester_name,
                contact,
                category,
                description,
                config.DEFAULT_TICKET_STATUS,
                timestamp,
                timestamp,
            ),
        )
        return cursor.lastrowid


def get_ticket(ticket_id, db_path=None):
    """Return one ticket as a dictionary, or None if it does not exist."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
    return dict(row) if row else None


def list_tickets(status=None, category=None, db_path=None):
    """Return tickets, newest first. Optionally filter by status and/or category."""
    query = "SELECT * FROM tickets"
    conditions = []
    values = []

    if status:
        conditions.append("status = ?")
        values.append(status)
    if category:
        conditions.append("category = ?")
        values.append(category)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, values).fetchall()
    return [dict(row) for row in rows]


def update_ticket_status(ticket_id, new_status, db_path=None):
    """Change a ticket's status. Returns True if the ticket exists, else False."""
    if new_status not in config.TICKET_STATUSES:
        raise ValueError(f"Unknown status: {new_status}")

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, now(), ticket_id),
        )
        return cursor.rowcount > 0


def get_statistics(db_path=None):
    """Return counts for the admin dashboard.

    {"total": int,
     "by_status":   {status: count, ...}      (every configured status, 0 if none),
     "by_category": {category: count, ...}    (every configured category, 0 if none),
     "per_day":     [{"day": "YYYY-MM-DD", "count": int}, ...]}
    """
    with get_connection(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]
        status_rows = connection.execute(
            "SELECT status, COUNT(*) AS n FROM tickets GROUP BY status"
        ).fetchall()
        category_rows = connection.execute(
            "SELECT category, COUNT(*) AS n FROM tickets GROUP BY category"
        ).fetchall()
        day_rows = connection.execute(
            "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n "
            "FROM tickets GROUP BY day ORDER BY day"
        ).fetchall()

    by_status = {status: 0 for status in config.TICKET_STATUSES}
    for row in status_rows:
        by_status[row["status"]] = row["n"]

    by_category = {category: 0 for category in config.TICKET_CATEGORIES}
    for row in category_rows:
        by_category[row["category"]] = row["n"]

    return {
        "total": total,
        "by_status": by_status,
        "by_category": by_category,
        "per_day": [{"day": row["day"], "count": row["n"]} for row in day_rows],
    }


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.ticket_manager
# (uses a temporary database file, so it never touches the real one)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sqlite3
    import tempfile
    from pathlib import Path

    from src.database import create_conversation, init_db

    with tempfile.TemporaryDirectory() as temp_folder:
        test_db = Path(temp_folder) / "test.db"
        init_db(test_db)

        first = create_ticket(
            "Asha", "asha@example.com", "Wi-Fi", "No internet in lab 2", db_path=test_db
        )
        conversation = create_conversation(test_db)
        second = create_ticket(
            "Ravi", "9876543210", "University Email", "Account locked",
            conversation_id=conversation, db_path=test_db,
        )
        assert (first, second) == (1, 2)

        ticket = get_ticket(first, test_db)
        print("Ticket 1:", ticket)
        assert ticket["status"] == "Open"
        assert ticket["requester_name"] == "Asha"
        assert get_ticket(999, test_db) is None
        assert get_ticket(second, test_db)["conversation_id"] == conversation

        for bad_arguments in [
            ("", "x", "Wi-Fi", "desc"),
            ("A", "  ", "Wi-Fi", "desc"),
            ("A", "x", "Wi-Fi", "   "),
            ("A", "x", "Not a category", "desc"),
        ]:
            try:
                create_ticket(*bad_arguments, db_path=test_db)
                raise AssertionError(f"Should have rejected {bad_arguments}")
            except ValueError:
                pass
        assert len(list_tickets(db_path=test_db)) == 2   # nothing invalid was saved

        assert [t["id"] for t in list_tickets(db_path=test_db)] == [2, 1]
        assert [t["id"] for t in list_tickets(category="Wi-Fi", db_path=test_db)] == [1]

        assert update_ticket_status(first, "Resolved", test_db) is True
        assert update_ticket_status(999, "Resolved", test_db) is False
        try:
            update_ticket_status(first, "Closed forever", test_db)
            raise AssertionError("Should have rejected unknown status")
        except ValueError:
            pass
        assert [t["id"] for t in list_tickets(status="Resolved", db_path=test_db)] == [1]
        assert get_ticket(first, test_db)["updated_at"] >= get_ticket(first, test_db)["created_at"]

        stats = get_statistics(test_db)
        print("Statistics:", stats)
        assert stats["total"] == 2
        assert stats["by_status"] == {"Open": 1, "In Progress": 0, "Resolved": 1}
        assert stats["by_category"]["Wi-Fi"] == 1
        assert stats["by_category"]["LMS"] == 0
        assert len(stats["per_day"]) == 1 and stats["per_day"][0]["count"] == 2

        try:
            create_ticket("A", "x", "Wi-Fi", "desc", conversation_id=999, db_path=test_db)
            raise AssertionError("Foreign key should have rejected conversation 999")
        except sqlite3.IntegrityError:
            pass

    print("Ticket manager checks passed.")