"""
llm_client.py - The only file that talks to Ollama (local Qwen3 model).

generate_answer(messages) : send chat messages, get the answer text back.
check_ollama()            : (ok, message) - is Ollama running and is the model installed?
Problems are raised as LLMError so callers can show a friendly message.
"""

import re

import requests

from src import config


class LLMError(Exception):
    """Raised when the local LLM cannot produce an answer."""


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text):
    """Remove any <think>...</think> reasoning block Qwen3 may include."""
    return THINK_BLOCK.sub("", text).strip()


def _post_chat(payload):
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    try:
        return requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError:
        raise LLMError(
            f"Could not connect to Ollama at {config.OLLAMA_BASE_URL}. Is Ollama running?"
        ) from None
    except requests.exceptions.Timeout:
        raise LLMError(
            f"Ollama did not answer within {config.OLLAMA_TIMEOUT_SECONDS} seconds."
        ) from None


def generate_answer(messages):
    """Send chat messages to Ollama and return the answer text."""
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": config.OLLAMA_TEMPERATURE},
    }

    response = _post_chat(payload)

    # Some models do not support the "think" switch: retry without it.
    if response.status_code == 400 and "think" in response.text.lower():
        payload.pop("think")
        response = _post_chat(payload)

    if response.status_code != 200:
        raise LLMError(
            f"Ollama returned an error ({response.status_code}): {response.text[:200]}"
        )

    content = response.json().get("message", {}).get("content", "")
    answer = strip_thinking(content)
    if not answer:
        raise LLMError("Ollama returned an empty answer.")
    return answer


def check_ollama():
    """Return (True, message) if Ollama is reachable and the model is installed."""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return False, f"Ollama is not reachable at {config.OLLAMA_BASE_URL}."

    names = [model.get("name", "") for model in response.json().get("models", [])]
    if config.OLLAMA_MODEL not in names:
        return False, (
            f"Model {config.OLLAMA_MODEL} is not installed. "
            f"Run: ollama pull {config.OLLAMA_MODEL}"
        )
    return True, f"Ollama is running with {config.OLLAMA_MODEL}."


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.llm_client
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    assert strip_thinking("<think>abc\ndef</think>\n\nHello") == "Hello"
    assert strip_thinking("No thinking here") == "No thinking here"
    print("strip_thinking checks passed.")

    # A wrong address must give a clear LLMError, not a crash.
    real_url = config.OLLAMA_BASE_URL
    config.OLLAMA_BASE_URL = "http://localhost:9"
    try:
        generate_answer([{"role": "user", "content": "hi"}])
        raise AssertionError("Expected LLMError for a wrong address")
    except LLMError as error:
        print("Wrong address handled:", error)
    finally:
        config.OLLAMA_BASE_URL = real_url

    ok, message = check_ollama()
    print("check_ollama:", ok, "-", message)

    if ok:
        from src.prompt_builder import build_messages
        from src.retriever import retrieve

        question = "My Wi-Fi is connected but there is no internet"
        chunks = retrieve(question)

        for test_question in [question, "Who won the cricket match yesterday?"]:
            messages = build_messages(test_question, chunks)
            started = time.perf_counter()
            answer = generate_answer(messages)
            elapsed = time.perf_counter() - started
            print(f"\nQuestion: {test_question}")
            print(f"Answer ({elapsed:.1f} s):\n{answer}")