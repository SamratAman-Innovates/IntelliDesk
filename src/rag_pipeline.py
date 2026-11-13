
"""
rag_pipeline.py - The full question -> answer flow (the "brain" of IntelliDesk).

answer_question(question, history) returns ONE result dictionary:
    status              "answered", "out_of_scope", "not_found" or "error"
    answer              text to show the user
    sources             [{"source", "page", "label", "score"}, ...]
    best_score          the best retrieval score found
    uses_sample_content True if any source is sample demonstration material
    offer_ticket        True whenever the chatbot could not answer reliably
    error               technical detail when status is "error", else None

Flow: retrieve -> scope check -> evidence check -> prompt ->
      local LLM -> not-found check -> result
"""

import re

from src import config
from src.llm_client import LLMError, generate_answer
from src.prompt_builder import NOT_FOUND_MARKER, build_messages
from src.retriever import relevant_results, retrieve

OUT_OF_SCOPE_MESSAGE = (
    "I couldn't find information about this in the approved IT support documents, "
    "so I can't answer it reliably. I can create a support ticket so the IT team "
    "can help you."
)
NOT_FOUND_MESSAGE = (
    "The IT support documents I have don't contain a reliable answer to this "
    "question. I can create a support ticket so the IT team can help you."
)
ERROR_MESSAGE = (
    "I can't generate an answer right now because the AI model is unavailable. "
    "I can create a support ticket so the IT team can help you."
)


def _collect_sources(chunks):
    """List each (file, page) once, with its label and best score."""
    sources = []
    seen = set()
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "label": chunk["label"],
                "score": round(chunk["score"], 3),
            }
        )
    return sources


def _is_not_found(answer):
    """True if the model replied with the not-found marker."""
    normalized = answer.upper().replace("_", " ")
    return NOT_FOUND_MARKER.replace("_", " ") in normalized



def _has_evidence(question, chunks):
    """Check meaningful overlap with the best-matching chunk only."""
    stop_words = {
        "how", "what", "when", "where", "why", "who", "which",
        "can", "could", "would", "should", "do", "does", "did",
        "is", "are", "am", "the", "a", "an", "to", "for", "of",
        "in", "on", "at", "my", "me", "i", "we", "our", "it",
        "this", "that", "and", "or", "with", "from", "please",
        "tell", "show", "help",
    }

    question_terms = {
        word for word in re.findall(r"[a-z0-9]+", question.lower())
        if len(word) > 2 and word not in stop_words
    }

    if not question_terms or not chunks:
        return False

    # Examine only the highest-ranked chunk.
    best_chunk_text = chunks[0]["text"].lower()
    document_terms = set(re.findall(r"[a-z0-9]+", best_chunk_text))

    matching_terms = question_terms.intersection(document_terms)

    return len(matching_terms) >= 2


def _result(status, answer, sources=None, best_score=0.0, error=None):
    sources = sources or []
    return {
        "status": status,
        "answer": answer,
        "sources": sources,
        "best_score": round(best_score, 3),
        "uses_sample_content": any(
            source["label"] == config.LABEL_SAMPLE for source in sources
        ),
        "offer_ticket": status != config.STATUS_ANSWERED,
        "error": error,
    }


def answer_question(question, history=None):
    """Answer one question using only the approved knowledge base."""
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")

    results = retrieve(question)
    best_score = results[0]["score"] if results else 0.0
    relevant = relevant_results(results)

    # Safety net 1: nothing in the documents matches.
    if not relevant:
        return _result(
            config.STATUS_OUT_OF_SCOPE,
            OUT_OF_SCOPE_MESSAGE,
            best_score=best_score,
        )

    # Safety net 2: retrieved text must contain meaningful evidence.
    if not _has_evidence(question, relevant):
        return _result(
            config.STATUS_NOT_FOUND,
            NOT_FOUND_MESSAGE,
            best_score=best_score,
        )

    messages = build_messages(question, relevant, history)

    try:
        answer = generate_answer(messages)
    except LLMError as error:
        return _result(
            config.STATUS_ERROR,
            ERROR_MESSAGE,
            best_score=best_score,
            error=str(error),
        )

    # Safety net 3: the model says the context does not contain the answer.
    if _is_not_found(answer):
        return _result(
            config.STATUS_NOT_FOUND,
            NOT_FOUND_MESSAGE,
            best_score=best_score,
        )

    return _result(
        config.STATUS_ANSWERED,
        answer,
        sources=_collect_sources(relevant),
        best_score=best_score,
    )


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.rag_pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("My Wi-Fi is connected but there is no internet", config.STATUS_ANSWERED),
        ("I forgot my email password", config.STATUS_ANSWERED),
        ("How do I change my Wi-Fi password?", config.STATUS_NOT_FOUND),
        ("Who won the cricket match yesterday?", config.STATUS_OUT_OF_SCOPE),
        ("How do I install MATLAB on my laptop?", config.STATUS_OUT_OF_SCOPE),
    ]

    print(f"Similarity threshold: {config.SIMILARITY_THRESHOLD}\n")
    for question, expected in tests:
        result = answer_question(question)
        verdict = "OK   " if result["status"] == expected else "CHECK"
        sources = [(s["source"], s["page"], s["label"]) for s in result["sources"]]
        print(f"[{verdict}] {question}")
        print(
            f"        status={result['status']} (expected {expected}) "
            f"best_score={result['best_score']}"
        )
        print(f"        sources={sources}")
        print(
            f"        offer_ticket={result['offer_ticket']} "
            f"uses_sample_content={result['uses_sample_content']}"
        )
        print(f"        answer: {result['answer'][:250]!r}\n")

    # The LLM being unreachable must give status "error", not a crash.
    real_url = config.OLLAMA_BASE_URL
    config.OLLAMA_BASE_URL = "http://localhost:9"
    try:
        result = answer_question(
            "My Wi-Fi is connected but there is no internet"
        )
        print(
            f"LLM down -> status={result['status']}, "
            f"offer_ticket={result['offer_ticket']}"
        )
        assert result["status"] == config.STATUS_ERROR
        assert result["offer_ticket"] is True
    finally:
        config.OLLAMA_BASE_URL = real_url

    print("\nError handling check passed.")
