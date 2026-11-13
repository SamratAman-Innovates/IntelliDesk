"""
evaluation/run_e2e_test.py - Headless end-to-end test of the Streamlit pages.

Run from the project root:   python evaluation\run_e2e_test.py
Simulates a user: asks questions in the chatbot, creates a ticket, then opens
the admin dashboard and changes the ticket status. It uses a temporary
database, so real conversations and tickets are never touched.
Needs Ollama running.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

from src import config
from src.ticket_manager import list_tickets

results = []


def check(label, condition):
    results.append(bool(condition))
    print(("PASS  " if condition else "FAIL  ") + label)


def find_button(app, label):
    for button in app.button:
        if button.label == label:
            return button
    raise AssertionError(f"Button not found: {label}")


def main():
    app_path = str(config.PROJECT_ROOT / "app.py")
    dashboard_path = str(config.PROJECT_ROOT / "pages" / "1_Admin_Dashboard.py")

    with tempfile.TemporaryDirectory() as folder:
        config.DATABASE_PATH = Path(folder) / "e2e.db"

        at = AppTest.from_file(app_path, default_timeout=180).run()
        check("chat page starts without exceptions", not at.exception)

        at.chat_input[0].set_value("My Wi-Fi is connected but there is no internet").run()
        messages = at.session_state["messages"]
        check(
            "in-scope question answered",
            len(messages) == 2 and messages[-1]["status"] == "answered",
        )
        check(
            "answer cites the Wi-Fi guide",
            any(s["source"] == "wifi_troubleshooting_sample.txt" for s in messages[-1]["sources"]),
        )

        at.chat_input[0].set_value("What is the capital of France?").run()
        messages = at.session_state["messages"]
        check("off-topic question refused", messages[-1]["status"] == "out_of_scope")
        check(
            "refusal offers the ticket button",
            any(button.label == "Create a support ticket" for button in at.button),
        )

        find_button(at, "Create a support ticket").click().run()
        at.text_input[0].set_value("E2E User")
        at.text_input[1].set_value("e2e@example.com")
        find_button(at, "Submit ticket").click().run()
        messages = at.session_state["messages"]
        check("ticket created from the chatbot", messages[-1]["status"] == "ticket_created")

        tickets = list_tickets()
        check(
            "ticket saved in SQLite with its conversation",
            len(tickets) == 1 and tickets[0]["conversation_id"] is not None,
        )

        dash = AppTest.from_file(dashboard_path, default_timeout=60).run()
        check("dashboard starts without exceptions", not dash.exception)
        check("dashboard shows 1 ticket", str(dash.metric[0].value) == "1")

        status_box = [box for box in dash.selectbox if box.label == "New status"][0]
        status_box.select("Resolved")
        find_button(dash, "Update status").click().run()
        check(
            "status change saved from the dashboard",
            len(list_tickets(status="Resolved")) == 1,
        )

    print(f"\n{sum(results)}/{len(results)} end-to-end checks passed")
    if all(results):
        print("RESULT: ALL CHECKS PASSED")
        return 0
    print("RESULT: SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())