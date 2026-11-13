"""
build_index.py - Builds the search index from the knowledge base.

Run from the project root:   python build_index.py
Re-run it whenever documents in data/knowledge_base/ change.

Flow: load -> clean -> chunk -> tokenize -> TF-IDF -> save
"""

import sys
from collections import Counter

from src import config
from src.chunker import chunk_documents
from src.document_loader import load_documents
from src.index_store import save_index
from src.text_cleaner import clean_text, tokenize
from src.tfidf_vectorizer import TfidfVectorizer


def build_index():
    """Build and save the index. Returns True on success, False otherwise."""
    print(f"{config.APP_NAME}: building the search index")
    print("-" * 50)

    documents = load_documents()
    for document in documents:
        document["text"] = clean_text(document["text"])
    documents = [document for document in documents if document["text"]]

    if not documents:
        print(
            f"No documents found in {config.KNOWLEDGE_BASE_DIR}. "
            "Add .pdf or .txt files and try again."
        )
        return False

    chunks = chunk_documents(documents)
    token_lists = [tokenize(chunk["text"]) for chunk in chunks]

    empty_chunks = sum(1 for tokens in token_lists if not tokens)
    if empty_chunks:
        print(f"Warning: {empty_chunks} chunk(s) contain no searchable words.")

    vectorizer = TfidfVectorizer().fit(token_lists)
    matrix = vectorizer.transform(token_lists)
    save_index(vectorizer, matrix, chunks)

    files_per_label = Counter()
    for source, label in {(chunk["source"], chunk["label"]) for chunk in chunks}:
        files_per_label[label] += 1
    chunks_per_label = Counter(chunk["label"] for chunk in chunks)

    print(f"Files read : {sum(files_per_label.values())}")
    print(f"Pages read : {len(documents)}")
    print(f"Chunks     : {len(chunks)}")
    print(f"Vocabulary : {len(vectorizer.vocabulary)} words")
    for label in (config.LABEL_SAMPLE, config.LABEL_VERIFIED):
        print(
            f"  {label:<9}: {files_per_label[label]} file(s), "
            f"{chunks_per_label[label]} chunk(s)"
        )
    if chunks_per_label[config.LABEL_VERIFIED] == 0:
        print("Note: no verified documents yet, so answers rely on SAMPLE content only.")
    print(f"Index saved to: {config.INDEX_DIR}")
    return True


if __name__ == "__main__":
    if not build_index():
        sys.exit(1)