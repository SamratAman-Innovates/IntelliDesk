"""
pages/1_Admin_Dashboard.py - IntelliDesk admin dashboard (Streamlit page).

Shows ticket statistics and charts, lets staff filter tickets, change their
status and read the chatbot conversation linked to a ticket.
All data comes from src/ticket_manager.py and src/database.py.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.database import get_messages, init_db
from src.ticket_manager import (
    get_statistics,
    get_ticket,
    list_tickets,
    update_ticket_status,
)

st.set_page_config(page_title=f"{config.APP_NAME} - Admin", page_icon=":bar_chart:")
init_db()

st.title(f"{config.APP_NAME} - Admin Dashboard")
st.caption("Ticket statistics and ticket management. This demo page has no login.")

flash_message = st.session_state.pop("flash_message", None)
if flash_message:
    st.success(flash_message)

if st.button("Refresh"):
    st.rerun()

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
stats = get_statistics()

metric_columns = st.columns(1 + len(config.TICKET_STATUSES))
metric_columns[0].metric("Total tickets", stats["total"])
for column, status in zip(metric_columns[1:], config.TICKET_STATUSES):
    column.metric(status, stats["by_status"][status])

if stats["total"] == 0:
    st.info("No tickets yet. Tickets created in the chatbot will appear here.")
    st.stop()

st.subheader("Tickets by category (most common IT problems)")
category_df = pd.DataFrame(
    {
        "Category": list(stats["by_category"].keys()),
        "Tickets": list(stats["by_category"].values()),
    }
)
st.plotly_chart(px.bar(category_df, x="Category", y="Tickets"))

left, right = st.columns(2)
with left:
    st.subheader("Tickets by status")
    status_df = pd.DataFrame(
        {
            "Status": list(stats["by_status"].keys()),
            "Tickets": list(stats["by_status"].values()),
        }
    )
    st.plotly_chart(px.pie(status_df, names="Status", values="Tickets", hole=0.4))
with right:
    st.subheader("Tickets per day")
    day_df = pd.DataFrame(stats["per_day"]).rename(
        columns={"day": "Day", "count": "Tickets"}
    )
    st.plotly_chart(px.bar(day_df, x="Day", y="Tickets"))

# ---------------------------------------------------------------------------
# Ticket list
# ---------------------------------------------------------------------------
st.subheader("All tickets")
status_column, category_column = st.columns(2)
status_filter = status_column.selectbox("Status", ["All"] + config.TICKET_STATUSES)
category_filter = category_column.selectbox("Category", ["All"] + config.TICKET_CATEGORIES)

tickets = list_tickets(
    status=None if status_filter == "All" else status_filter,
    category=None if category_filter == "All" else category_filter,
)

if not tickets:
    st.info("No tickets match these filters.")
    st.stop()

table_columns = [
    "id", "created_at", "requester_name", "contact", "category", "status", "description",
]
tickets_df = pd.DataFrame(tickets)[table_columns]
st.dataframe(tickets_df, hide_index=True)
st.download_button(
    "Download tickets (CSV)",
    tickets_df.to_csv(index=False),
    file_name="intellidesk_tickets.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Update one ticket
# ---------------------------------------------------------------------------
st.subheader("Update a ticket")
options = {
    f"#{ticket['id']} - {ticket['requester_name']} - {ticket['category']} ({ticket['status']})": ticket["id"]
    for ticket in tickets
}
selected_label = st.selectbox("Select a ticket", list(options.keys()))
selected = get_ticket(options[selected_label])

current_index = (
    config.TICKET_STATUSES.index(selected["status"])
    if selected["status"] in config.TICKET_STATUSES
    else 0
)
new_status = st.selectbox("New status", config.TICKET_STATUSES, index=current_index)

if st.button("Update status"):
    if update_ticket_status(selected["id"], new_status):
        st.session_state["flash_message"] = f"Ticket #{selected['id']} is now {new_status}."
        st.rerun()
    else:
        st.error("The ticket no longer exists.")

with st.expander("Ticket details and conversation"):
    st.markdown(f"**Name:** {selected['requester_name']}")
    st.markdown(f"**Contact:** {selected['contact']}")
    st.markdown(f"**Category:** {selected['category']}")
    st.markdown(f"**Status:** {selected['status']}")
    st.markdown(f"**Created:** {selected['created_at']}  |  **Updated:** {selected['updated_at']}")
    st.markdown(f"**Description:** {selected['description']}")

    if selected["conversation_id"]:
        st.markdown("**Conversation before the ticket:**")
        for message in get_messages(selected["conversation_id"]):
            speaker = "User" if message["role"] == "user" else config.APP_NAME
            st.markdown(f"- **{speaker}:** {message['content']}")
    else:
        st.caption("No conversation is linked to this ticket.")