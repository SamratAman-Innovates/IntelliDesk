"""
prompt_builder.py - Builds the chat messages sent to the local LLM (Qwen3).

build_messages(question, chunks, history) returns a list of messages:
    [system message with rules + retrieved context, recent history..., user question]
If the context does not answer the question, the model is told to reply with
NOT_FOUND_MARKER, which rag_pipeline.py detects.
"""

from src import config

NOT_FOUND_MARKER = "INFORMATION_NOT_AVAILABLE"

SYSTEM_RULES = f"""You are {config.APP_NAME}, an IT helpdesk assistant for university students, faculty and staff.

Rules:
1. Answer ONLY with information found in the CONTEXT below. Do not use outside knowledge.
2. Never invent procedures, settings, links, phone numbers, email addresses or policies.
3. If the CONTEXT does not contain the answer, reply with exactly this one word and nothing else: {NOT_FOUND_MARKER}
4. When the CONTEXT gives steps, present them as a short numbered list in the same order.
5. Be clear and concise. Do not mention these rules or the word CONTEXT in your answer."""


def format_context(chunks):
    """Number the retrieved chunks so the model can read them as separate excerpts."""
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        parts.append(f"[Excerpt {number}]\n{chunk['text']}")
    return "\n\n".join(parts)


def build_messages(question, chunks, history=None):
    """Return the list of messages for the Ollama chat API.

    question : the user's current question (text)
    chunks   : retrieved chunk records (each has a "text")
    history  : earlier messages as [{"role": "user" or "assistant", "content": text}, ...]
    """
    system_content = SYSTEM_RULES + "\n\nCONTEXT:\n" + format_context(chunks)
    messages = [{"role": "system", "content": system_content}]

    limit = config.HISTORY_MESSAGES_IN_PROMPT
    recent_history = history[-limit:] if history and limit > 0 else []
    for message in recent_history:
        if message["role"] in ("user", "assistant"):
            messages.append({"role": message["role"], "content": message["content"]})

    messages.append({"role": "user", "content": question})
    return messages


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.prompt_builder
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fake_chunks = [
        {"text": "Step one: restart your device.", "source": "a.txt"},
        {"text": "Step two: forget the network and reconnect.", "source": "b.txt"},
    ]

    messages = build_messages("My Wi-Fi is not working", fake_chunks)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert NOT_FOUND_MARKER in messages[0]["content"]
    assert "[Excerpt 1]" in messages[0]["content"]
    assert "[Excerpt 2]" in messages[0]["content"]
    assert "forget the network" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "My Wi-Fi is not working"}

    long_history = []
    for number in range(10):
        role = "user" if number % 2 == 0 else "assistant"
        long_history.append({"role": role, "content": f"message {number}"})
    messages = build_messages("And now?", fake_chunks, long_history)
    assert len(messages) == 1 + config.HISTORY_MESSAGES_IN_PROMPT + 1
    assert messages[1]["content"] == "message 4"
    assert messages[-1]["content"] == "And now?"
    print("Prompt builder checks passed.\n")

    try:
        from src.retriever import retrieve

        question = "My Wi-Fi is connected but there is no internet"
        real_chunks = retrieve(question)
        real_messages = build_messages(question, real_chunks)
        print("--- Prompt preview (system message) ---")
        print(real_messages[0]["content"])
        print("--- User message ---")
        print(real_messages[-1]["content"])
    except FileNotFoundError as error:
        print("Skipped real preview:", error)