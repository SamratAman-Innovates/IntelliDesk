"""
evaluation/run_evaluation.py - End-to-end evaluation of IntelliDesk.

Run from the project root:
    python evaluation\run_evaluation.py                    full run (needs Ollama)
    python evaluation\run_evaluation.py --retrieval-only   fast run, no LLM

Questions and expectations live in evaluation/test_questions.json.
Checks: retrieval quality, answer status, source citations, fallback for
off-topic and unanswerable questions, and ticket escalation (temporary database).
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.database import create_conversation, init_db
from src.index_store import index_exists
from src.llm_client import check_ollama
from src.rag_pipeline import answer_question
from src.retriever import relevant_results, retrieve
from src.ticket_manager import create_ticket, get_statistics, list_tickets

QUESTIONS_FILE = config.EVALUATION_DIR / "test_questions.json"


def load_cases():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def short(text, length=58):
    return text if len(text) <= length else text[: length - 3] + "..."


# ---------------------------------------------------------------------------
# 1. Retrieval only (no LLM)
# ---------------------------------------------------------------------------
def evaluate_retrieval(cases):
    failures = []
    in_scope = top1 = top3 = 0
    refusal_total = refusal_ok = 0

    for case in cases:
        results = retrieve(case["question"])
        sources = [result["source"] for result in results]

        if case["type"] == "in_scope":
            in_scope += 1
            expected = case["expected_source"]
            top1 += bool(sources) and sources[0] == expected
            if expected in sources:
                top3 += 1
            else:
                failures.append(
                    f"retrieval {case['id']}: expected {expected} in the top "
                    f"{config.TOP_K}, got {sources}"
                )

        elif case["type"] == "out_of_scope":
            refusal_total += 1
            if not relevant_results(results):
                refusal_ok += 1
            else:
                failures.append(
                    f"retrieval {case['id']}: off-topic question reached the LLM "
                    f"(best score {results[0]['score']:.3f}, threshold "
                    f"{config.SIMILARITY_THRESHOLD})"
                )

    return {
        "in_scope": in_scope,
        "top1": top1,
        "top3": top3,
        "refusal_total": refusal_total,
        "refusal_ok": refusal_ok,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# 2. Full pipeline (retrieval + Ollama)
# ---------------------------------------------------------------------------
def evaluate_pipeline(cases):
    rows = []
    for case in cases:
        started = time.perf_counter()
        problems = []
        try:
            result = answer_question(case["question"])
        except Exception as error:
            result = {
                "status": "exception", "answer": "", "sources": [],
                "best_score": 0.0, "offer_ticket": False,
            }
            problems.append(f"exception: {error!r}")
        elapsed = time.perf_counter() - started

        if result["status"] not in case["expected_statuses"]:
            problems.append(
                f"status '{result['status']}' not in {case['expected_statuses']}"
            )
        if result["offer_ticket"] != (result["status"] != config.STATUS_ANSWERED):
            problems.append("offer_ticket does not match the status")

        if result["status"] == config.STATUS_ANSWERED:
            cited = [source["source"] for source in result["sources"]]
            if not result["answer"].strip():
                problems.append("empty answer")
            if not cited:
                problems.append("answered without source references")
            elif case.get("expected_source") and case["expected_source"] not in cited:
                problems.append(
                    f"expected source {case['expected_source']} missing from {cited}"
                )
        elif result["sources"]:
            problems.append("a refusal should not list sources")

        rows.append(
            {"case": case, "result": result, "seconds": elapsed, "problems": problems}
        )
    return rows


def evaluate_ticket_escalation(rows):
    """Turn every refused question into a ticket, in a temporary database."""
    refused = [row for row in rows if row["result"]["offer_ticket"]]
    problems = []

    if not refused:
        return 0, ["no refused question was available to escalate"]

    with tempfile.TemporaryDirectory() as temp_folder:
        db_path = Path(temp_folder) / "evaluation.db"
        init_db(db_path)
        for row in refused:
            conversation_id = create_conversation(db_path)
            create_ticket(
                "Evaluation User",
                "evaluation@example.com",
                "Other",
                row["case"]["question"],
                conversation_id=conversation_id,
                db_path=db_path,
            )
        stats = get_statistics(db_path)
        if stats["total"] != len(refused):
            problems.append(f"expected {len(refused)} tickets, found {stats['total']}")
        if stats["by_status"][config.DEFAULT_TICKET_STATUS] != len(refused):
            problems.append("new tickets are not all in the default status")
        if len(list_tickets(db_path=db_path)) != len(refused):
            problems.append("ticket list does not match the number created")

    return len(refused), problems


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_case_rows(rows):
    print("\n== Pipeline results ==")
    for row in rows:
        case, result = row["case"], row["result"]
        mark = "FAIL" if row["problems"] else "ok  "
        print(
            f"  {mark} {case['id']:<11} {result['status']:<13} "
            f"score={result['best_score']:<6} {row['seconds']:>5.1f}s  "
            f"{short(case['question'])}"
        )
        for problem in row["problems"]:
            print(f"       -> {problem}")


def main():
    retrieval_only = "--retrieval-only" in sys.argv

    if not index_exists():
        print("No index found. Run 'python build_index.py' first.")
        return 1

    cases = load_cases()
    counts = {}
    for case in cases:
        counts[case["type"]] = counts.get(case["type"], 0) + 1
    print(f"{config.APP_NAME} evaluation: {len(cases)} questions {counts}")
    print(f"Similarity threshold {config.SIMILARITY_THRESHOLD}, top-k {config.TOP_K}")

    failures = []
    summary = []

    retrieval = evaluate_retrieval(cases)
    failures.extend(retrieval["failures"])
    summary.append(
        (f"Retrieval: expected document in the top {config.TOP_K}",
         retrieval["top3"], retrieval["in_scope"])
    )
    summary.append(
        ("Retrieval: off-topic questions find no relevant chunk",
         retrieval["refusal_ok"], retrieval["refusal_total"])
    )
    top1_text = f"top-1 accuracy {retrieval['top1']}/{retrieval['in_scope']}"

    if not retrieval_only:
        ok, message = check_ollama()
        print(f"Ollama: {message}")
        if not ok:
            print("RESULT: cannot run the full evaluation without Ollama "
                  "(use --retrieval-only)")
            return 1

        print("Running the full pipeline (this can take a minute or two)...")
        rows = evaluate_pipeline(cases)
        print_case_rows(rows)

        for row in rows:
            for problem in row["problems"]:
                failures.append(f"{row['case']['id']}: {problem}")

        def score(case_type):
            selected = [row for row in rows if row["case"]["type"] == case_type]
            return sum(1 for row in selected if not row["problems"]), len(selected)

        for case_type, label in [
            ("in_scope", "Answers: in-scope questions answered with correct sources"),
            ("out_of_scope", "Fallback: off-topic questions refused, ticket offered"),
            ("not_in_documents", "Safety: unanswerable IT questions are not invented"),
        ]:
            passed, total = score(case_type)
            summary.append((label, passed, total))

        escalated, ticket_problems = evaluate_ticket_escalation(rows)
        failures.extend(f"ticket escalation: {problem}" for problem in ticket_problems)
        summary.append(
            (f"Ticket escalation: {escalated} refused question(s) became tickets",
             0 if ticket_problems else 1, 1)
        )

        answered_times = [
            row["seconds"] for row in rows
            if row["result"]["status"] == config.STATUS_ANSWERED
        ]
        if answered_times:
            average = sum(answered_times) / len(answered_times)
            print(f"\nAverage time for an answered question: {average:.1f} s")

    print("\n== Summary ==")
    for label, passed, total in summary:
        mark = "PASS" if passed == total else "FAIL"
        print(f"  [{mark}] {label}: {passed}/{total}")
    print(f"  [INFO] Retrieval {top1_text}")

    if failures:
        print("\n== Problems ==")
        for failure in failures:
            print(f"  - {failure}")
        print(f"\nRESULT: {len(failures)} problem(s) found")
        return 1

    print("\nRESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())