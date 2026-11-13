"""
app.py - IntelliDesk chat page (Streamlit).

Run from the project root:   streamlit run app.py
Flow: question -> rag_pipeline.answer_question() -> answer + sources.
When the bot cannot answer reliably, the user can create a support ticket.
"""

import streamlit as st

from src import config
from src.database import (
    add_message,
    create_conversation,
    get_messages,
    init_db,
    list_conversations,
)
from src.index_store import index_exists
from src.llm_client import check_ollama
from src.rag_pipeline import answer_question
from src.retriever import reload_index
from src.ticket_manager import create_ticket

TICKET_CREATED_STATUS = "ticket_created"

st.set_page_config(page_title=config.APP_NAME, page_icon=":speech_balloon:")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_database():
    """Create the database tables once per server start."""
    init_db()
    return True


@st.cache_data(ttl=30)
def ollama_status():
    """Check Ollama at most every 30 seconds."""
    return check_ollama()


def init_state():
    defaults = {"conversation_id": None, "messages": [], "show_ticket_form": False}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------
def start_new_conversation():
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.show_ticket_form = False


def load_conversation(conversation_id):
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = get_messages(conversation_id)
    st.session_state.show_ticket_form = False


def save_message(role, content, status=None, sources=None):
    """Save a message to SQLite and to the on-screen conversation."""
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = create_conversation()
    add_message(st.session_state.conversation_id, role, content, status, sources)
    st.session_state.messages.append(
        {"role": role, "content": content, "status": status, "sources": sources or []}
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            badge = "SAMPLE" if source["label"] == config.LABEL_SAMPLE else "VERIFIED"
            st.markdown(
                f"- **{source['source']}**, page {source['page']} "
                f"`{badge}` (match {source['score']:.2f})"
            )
        if any(source["label"] == config.LABEL_SAMPLE for source in sources):
            st.warning(
                "Some sources are sample demonstration material, "
                "not verified university procedures."
            )


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


def render_sidebar():
    with st.sidebar:
        st.title(config.APP_NAME)
        st.caption(config.APP_TAGLINE)

        ok, message = ollama_status()
        if ok:
            st.success(message)
        else:
            st.error(message)

        if st.button("New conversation"):
            start_new_conversation()
            st.rerun()

        if st.button("Reload knowledge base"):
            try:
                reload_index()
                st.toast("Knowledge base reloaded.")
            except FileNotFoundError as error:
                st.error(str(error))

        st.divider()
        st.subheader("Recent conversations")
        conversations = list_conversations(limit=10)
        if not conversations:
            st.caption("No conversations yet.")
        for conversation in conversations:
            if st.button(conversation["title"], key=f"conversation_{conversation['id']}"):
                load_conversation(conversation["id"])
                st.rerun()

        st.divider()
        st.caption(
            "Answers come only from the approved IT knowledge base. "
            "Documents marked SAMPLE are demonstration content, "
            "not official procedures."
        )


def render_ticket_form():
    last_question = next(
        (
            message["content"]
            for message in reversed(st.session_state.messages)
            if message["role"] == "user"
        ),
        "",
    )

    st.subheader("Create a support ticket")
    with st.form("ticket_form"):
        name = st.text_input("Your name")
        contact = st.text_input("Email or phone number")
        category = st.selectbox("Category", config.TICKET_CATEGORIES)
        description = st.text_area("Describe the problem", value=last_question)
        submitted = st.form_submit_button("Submit ticket")

    if submitted:
        try:
            ticket_id = create_ticket(
                name,
                contact,
                category,
                description,
                conversation_id=st.session_state.conversation_id,
            )
        except ValueError as error:
            st.error(str(error))
            return
        except Exception as error:
            print(f"Ticket error: {error!r}")
            st.error("The ticket could not be saved. Please try again.")
            return

        st.session_state.show_ticket_form = False
        save_message(
            "assistant",
            f"Your support ticket #{ticket_id} has been created. "
            f"The IT team will contact you at {contact.strip()}.",
            status=TICKET_CREATED_STATUS,
        )
        st.rerun()

    if st.button("Cancel"):
        st.session_state.show_ticket_form = False
        st.rerun()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def main():
    init_state()
    setup_database()
    render_sidebar()

    st.title(f"{config.APP_NAME} - IT Helpdesk")

    if not index_exists():
        st.error(
            "The knowledge base index is missing. "
            "Run `python build_index.py` in the terminal, then reload this page."
        )
        st.stop()

    if not st.session_state.messages:
        st.info(
            f"Hi, I'm {config.APP_NAME}. Ask me about campus Wi-Fi, the student portal, "
            "university email, software installation, computer labs or the LMS. "
            "I answer only from approved IT documents."
        )

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Describe your IT problem...")
    if question:
        st.session_state.show_ticket_form = False
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        save_message("user", question)

        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the IT documents..."):
                try:
                    result = answer_question(question, history)
                except Exception as error:
                    print(f"Pipeline error: {error!r}")
                    result = {
                        "status": config.STATUS_ERROR,
                        "answer": (
                            "Something went wrong while answering. "
                            "I can create a support ticket so the IT team can help you."
                        ),
                        "sources": [],
                    }

        save_message("assistant", result["answer"], result["status"], result["sources"])
        st.rerun()

    if st.session_state.messages:
        last = st.session_state.messages[-1]
        if last["role"] == "assistant" and last.get("status") != TICKET_CREATED_STATUS:
            if not st.session_state.show_ticket_form:
                if st.button("Create a support ticket"):
                    st.session_state.show_ticket_form = True
                    st.rerun()

    if st.session_state.show_ticket_form:
        render_ticket_form()


main()
