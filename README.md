# IntelliDesk
AI-powered IT Helpdesk chatbot using a manually implemented RAG pipeline, Gemini API, Python, Streamlit, ChromaDB, and SQLite.

# IntelliDesk

An AI-powered IT Helpdesk and Technical Support chatbot using a manually implemented RAG pipeline and a local LLM for intelligent, context-aware responses.

IntelliDesk helps students, faculty and staff with common campus IT problems. It answers only from an approved knowledge base, shows which documents it used, refuses to invent answers, and lets the user escalate to a support ticket when it cannot help.

## Features

- Campus Wi-Fi, student portal, university email, software installation, computer lab, LMS and account security support
- Source-based answers with a source list under every answer
- SAMPLE and VERIFIED labels, so demonstration content is never mistaken for an official procedure
- Fallback when the information is not in the documents (no invented answers)
- Support ticket creation, with the chat conversation linked to the ticket
- Conversation history stored in SQLite
- Admin dashboard with ticket statistics, charts, filters and status updates
- Evaluation scripts that test retrieval, answers, fallback and ticket escalation

## Architecture

All stages of the RAG pipeline are written by us in Python. No LangChain, no LlamaIndex, no external LLM API.

```
Knowledge base (PDF / TXT)
        |  build_index.py (run when documents change)
        v
document_loader -> text_cleaner -> chunker -> tfidf_vectorizer -> index_store
   (PyMuPDF)        (regex)        (words)     (TF-IDF, NumPy)     data/index/

User question (Streamlit app.py)
        |
        v
rag_pipeline.answer_question()
   1. retriever      cleans + vectorizes the question, cosine similarity (NumPy), top-k chunks
   2. scope check    best score below the threshold -> "out_of_scope", the LLM is never called
   3. prompt_builder strict rules + numbered excerpts + recent chat history
   4. llm_client     local Qwen3 through Ollama
   5. not-found check the model answers INFORMATION_NOT_AVAILABLE -> "not_found"
        |
        v
{status, answer, sources, offer_ticket, ...} -> chat page -> ticket form -> SQLite
```

The retrieval is ours (TF-IDF and cosine similarity). The language model is pretrained (Qwen3) and is only used to write the final answer.

### Technologies

| Purpose | Technology |
|---|---|
| Language | Python |
| PDF extraction | PyMuPDF |
| Cleaning | Python and regex |
| Chunking | Manual implementation |
| Vectorization | Manual TF-IDF |
| Similarity | Manual cosine similarity with NumPy |
| Local LLM | Qwen3 1.7B through Ollama |
| Frontend | Streamlit |
| Database | SQLite |
| Analytics | Pandas and Plotly |

## Project structure

```
IntelliDesk/
  app.py                     chat page
  build_index.py             builds the search index from the knowledge base
  requirements.txt
  pages/1_Admin_Dashboard.py ticket statistics and management
  src/
    config.py                all settings
    document_loader.py       reads PDF/TXT files
    text_cleaner.py          cleaning and tokenizing
    chunker.py               overlapping word chunks
    tfidf_vectorizer.py      manual TF-IDF
    similarity.py            manual cosine similarity
    index_store.py           saves/loads the index
    retriever.py             question -> best chunks with scores
    prompt_builder.py        builds the LLM messages
    llm_client.py            talks to Ollama
    rag_pipeline.py          the full question -> answer flow
    database.py              SQLite tables, conversations, messages
    ticket_manager.py        ticket rules and statistics
  data/
    knowledge_base/sample/   demonstration documents
    knowledge_base/verified/ verified university documents
    index/                   generated index (not in Git)
    intellidesk.db           generated database (not in Git)
  evaluation/
    test_questions.json      evaluation questions
    run_evaluation.py        retrieval, answer, fallback and ticket evaluation
    run_e2e_test.py          headless test of the chat page and dashboard
```

## Setup (Windows 11, PowerShell)

Prerequisites: Python 3.10 or newer (tested with 3.13), Git, and [Ollama](https://ollama.com).

```powershell
git clone https://github.com/SamratAman-Innovates/IntelliDesk.git
cd IntelliDesk
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen3:1.7b
python build_index.py
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## Run

Make sure Ollama is running (the llama icon in the system tray), then:

```powershell
streamlit run app.py
```

The app opens at http://localhost:8501.

- **Chat page:** ask an IT question. Answers show a Sources section. If the bot cannot answer reliably, use **Create a support ticket**.
- **Admin Dashboard** (sidebar): statistics, charts, ticket table with filters, CSV download, status updates and the conversation linked to each ticket. The demo dashboard has no login.

## Adding or changing knowledge-base documents

1. Put `.pdf` or `.txt` files in `data/knowledge_base/sample/` (demonstration content) or `data/knowledge_base/verified/` (verified university documents). The folder decides the label shown to users.
2. Rebuild the index: `python build_index.py`
3. In the running app, click **Reload knowledge base** in the sidebar (or restart the app).

The documents in `sample/` are generic demonstration material and are not official university procedures. Only put a document in `verified/` if it comes from the university.

## Configuration

All settings are in `src/config.py`: chunk size and overlap, number of retrieved chunks (`TOP_K`), the similarity threshold for the out-of-scope check (`SIMILARITY_THRESHOLD`), the Ollama address and model name, ticket categories and statuses.

To try a larger model: `ollama pull qwen3:4b`, then change `OLLAMA_MODEL` in `src/config.py`.

## Evaluation and testing

```powershell
python evaluation\run_evaluation.py --retrieval-only   # fast, no LLM
python evaluation\run_evaluation.py                    # full run, needs Ollama
python evaluation\run_e2e_test.py                      # chat page + ticket + dashboard, needs Ollama
```

`run_evaluation.py` uses `evaluation/test_questions.json` and reports retrieval quality, answer status with source citations, fallback for off-topic and unanswerable questions, and ticket escalation (in a temporary database). `run_e2e_test.py` simulates a user in the Streamlit pages with a temporary database. Because the local model is small, results for questions the documents cannot answer can vary slightly between runs.

Most modules also have a built-in check, for example `python -m src.text_cleaner`, `python -m src.database`, `python -m src.ticket_manager` and `python -m src.rag_pipeline`.

## Limitations

- Retrieval is keyword based (TF-IDF), so it matches words, not meaning. A question that uses very different words from the document may be missed.
- A small local model (1.7B parameters) can occasionally ignore instructions. The retrieval-score check is the main protection against invented answers.
- Follow-up questions are searched without the earlier conversation, so very short follow-ups such as "that did not work" lead to a ticket offer.
- Scanned PDFs (images without text) cannot be read.
- The admin dashboard has no authentication.

## Future improvements

- Replace TF-IDF with a local embedding model (for example through Ollama) for semantic retrieval. Only `tfidf_vectorizer.py` and the retriever would need to change.
- Add authentication to the admin dashboard.
- Improve handling of follow-up questions.

## Team

Add your team members' names here.