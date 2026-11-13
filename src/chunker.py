"""
chunker.py - Splits page records into smaller, overlapping chunks.

Input : page records  {"text", "source", "page", "label"}
Output: chunk records {"chunk_id", "text", "source", "page", "label"}
Chunk size is measured in words (see config.py). Line breaks inside a chunk
are preserved, so numbered steps stay readable.
"""

import re

from src import config

WORD_SPAN = re.compile(r"\S+")


def split_into_chunks(text, chunk_size, overlap):
    """Split one text into overlapping chunks of about chunk_size words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    spans = [match.span() for match in WORD_SPAN.finditer(text)]
    if not spans:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(spans):
        end = min(start + chunk_size, len(spans))
        chunks.append(text[spans[start][0]:spans[end - 1][1]])
        if end == len(spans):
            break
        start += step

    return chunks


def chunk_documents(
    documents,
    chunk_size=config.CHUNK_SIZE_WORDS,
    overlap=config.CHUNK_OVERLAP_WORDS,
):
    """Turn a list of page records into a list of chunk records."""
    chunks = []

    for document in documents:
        for chunk_text in split_into_chunks(document["text"], chunk_size, overlap):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"],
                    "label": document["label"],
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.chunker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.document_loader import load_documents
    from src.text_cleaner import clean_text

    numbers = " ".join(str(n) for n in range(1, 11))
    parts = split_into_chunks(numbers, chunk_size=4, overlap=1)
    print(parts)
    assert parts == ["1 2 3 4", "4 5 6 7", "7 8 9 10"]
    assert split_into_chunks("hello world", 4, 1) == ["hello world"]
    assert split_into_chunks("a b\nc d", 4, 1) == ["a b\nc d"]
    assert split_into_chunks("   ", 4, 1) == []
    print("Window logic checks passed.\n")

    documents = load_documents()
    for document in documents:
        document["text"] = clean_text(document["text"])

    chunks = chunk_documents(documents)
    print(f"{len(documents)} page(s) -> {len(chunks)} chunk(s)\n")
    for chunk in chunks:
        words = len(chunk["text"].split())
        print(f"#{chunk['chunk_id']} [{chunk['label']}] {chunk['source']} "
              f"page {chunk['page']} ({words} words)")
        print("    " + chunk["text"][:90].replace("\n", " ") + "...")