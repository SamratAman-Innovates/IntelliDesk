"""
retriever.py - Finds the chunks that best match a question.

retrieve(question)  : returns up to TOP_K chunks that share words with the question,
                      best first, each with a "score" between 0 and 1.
relevant_results()  : keeps only results at or above the similarity threshold.

Run  python -m src.retriever            to print the threshold tuning report.
Run  python -m src.retriever "question" to test one question.
"""

import sys

from src import config
from src.index_store import load_index
from src.similarity import cosine_similarity, top_k_indices
from src.text_cleaner import tokenize

_index = None  # (vectorizer, matrix, chunks), loaded on first use


def _get_index():
    global _index
    if _index is None:
        _index = load_index()
    return _index


def reload_index():
    """Read the index from disk again (use after running build_index.py)."""
    global _index
    _index = load_index()


def retrieve(question, top_k=config.TOP_K):
    """Return up to top_k matching chunks, best first.

    Each result is a chunk record plus a "score". Chunks with score 0
    (no shared words with the question) are never returned.
    """
    vectorizer, matrix, chunks = _get_index()

    query_vector = vectorizer.transform_one(tokenize(question))
    scores = cosine_similarity(query_vector, matrix)

    results = []
    for position in top_k_indices(scores, top_k):
        if scores[position] <= 0:
            break
        result = dict(chunks[position])
        result["score"] = float(scores[position])
        results.append(result)
    return results


def relevant_results(results, threshold=config.SIMILARITY_THRESHOLD):
    """Keep only the results whose score reaches the threshold."""
    return [result for result in results if result["score"] >= threshold]


# ---------------------------------------------------------------------------
# Threshold tuning tool
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    ("in", "My Wi-Fi is connected but there is no internet"),
    ("in", "I can't see the campus network in the list of Wi-Fi networks"),
    ("in", "I forgot my email password"),
    ("in", "Cannot log in to my university email"),
    ("in", "wifi not working on my phone"),
    ("in", "My email account is locked after too many attempts"),
    ("out", "Who won the cricket match yesterday?"),
    ("out", "What is the capital of France?"),
    ("out", "How do I install MATLAB on my laptop?"),
    ("out", "What is the hostel mess timing?"),
    ("out", "How do I connect to the lab printer?"),
]


def run_tuning_report():
    in_scores, out_scores = [], []
    print(f"Current threshold: {config.SIMILARITY_THRESHOLD}\n")

    for expected, question in TEST_QUESTIONS:
        results = retrieve(question)
        best = results[0]["score"] if results else 0.0
        where = f"#{results[0]['chunk_id']} {results[0]['source']}" if results else "-"
        print(f"[{expected:<3}] {best:.3f}  {question}")
        print(f"             best match: {where}")
        (in_scores if expected == "in" else out_scores).append(best)

    print()
    print(f"Lowest best-score among in-scope questions : {min(in_scores):.3f}")
    print(f"Highest best-score among out-of-scope ones : {max(out_scores):.3f}")
    if min(in_scores) > max(out_scores):
        print("A threshold between these two numbers separates them cleanly.")
    else:
        print("The two groups overlap: no single threshold separates them perfectly.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        results = retrieve(question)
        print(f"Question: {question}")
        if not results:
            print("No chunk shares any word with this question.")
        for result in results:
            preview = result["text"][:80].replace("\n", " ")
            print(
                f"  {result['score']:.3f}  #{result['chunk_id']} [{result['label']}] "
                f"{result['source']} page {result['page']}"
            )
            print(f"         {preview}...")
    else:
        run_tuning_report()