"""
document_loader.py - Reads knowledge-base documents from disk.

Each page of each document becomes one "page record", a dictionary:
    {"text": ..., "source": file name, "page": page number, "label": "sample" or "verified"}
This file only loads text. Cleaning and chunking happen in later files.
"""

import pymupdf

from src import config


def load_pdf(file_path):
    """Read a PDF and return a list of {"page": number, "text": text}, one per page with text."""
    pages = []
    with pymupdf.open(str(file_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({"page": page_number, "text": text})
    return pages


def load_text_file(file_path):
    """Read a .txt file. A text file has no pages, so everything is page 1."""
    text = file_path.read_text(encoding="utf-8-sig")
    if text.strip():
        return [{"page": 1, "text": text}]
    return []


def load_documents():
    """Load every supported file from every knowledge-base folder.

    Returns a list of page records (see the top of this file).
    """
    documents = []

    for label, folder in config.KNOWLEDGE_BASE_FOLDERS.items():
        if not folder.exists():
            print(f"Warning: folder not found: {folder}")
            continue

        for file_path in sorted(folder.iterdir()):
            extension = file_path.suffix.lower()
            if extension not in config.SUPPORTED_EXTENSIONS:
                continue

            try:
                if extension == ".pdf":
                    pages = load_pdf(file_path)
                else:
                    pages = load_text_file(file_path)
            except Exception as error:
                print(f"Warning: could not read {file_path.name}: {error}")
                continue

            if not pages:
                print(f"Warning: no text found in {file_path.name} (skipped)")
                continue

            for page in pages:
                documents.append(
                    {
                        "text": page["text"],
                        "source": file_path.name,
                        "page": page["page"],
                        "label": label,
                    }
                )

    return documents


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.document_loader
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} page(s) from the knowledge base\n")
    for doc in documents:
        preview = doc["text"][:70].replace("\n", " ")
        print(f"[{doc['label']}] {doc['source']} (page {doc['page']}, {len(doc['text'])} characters)")
        print(f"    {preview}...")