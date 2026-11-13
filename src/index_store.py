"""
index_store.py - Saves and loads the search index in data/index/.

The index is three files:
    vectorizer.json   the TF-IDF vocabulary and IDF values
    chunk_matrix.npy  the NumPy matrix with one vector per chunk
    chunks.json       the chunk records (text, source, page, label)
build_index.py saves it; retriever.py loads it.
"""

import json
from pathlib import Path

import numpy as np

from src import config
from src.tfidf_vectorizer import TfidfVectorizer

VECTORIZER_FILENAME = "vectorizer.json"
MATRIX_FILENAME = "chunk_matrix.npy"
CHUNKS_FILENAME = "chunks.json"


def _resolve_dir(index_dir):
    """Use the given folder, or the default index folder from config."""
    return Path(index_dir) if index_dir is not None else config.INDEX_DIR


def index_exists(index_dir=None):
    """True if all three index files are present."""
    folder = _resolve_dir(index_dir)
    filenames = (VECTORIZER_FILENAME, MATRIX_FILENAME, CHUNKS_FILENAME)
    return all((folder / name).exists() for name in filenames)


def save_index(vectorizer, matrix, chunks, index_dir=None):
    """Write the vectorizer, the chunk matrix and the chunk records to disk."""
    folder = _resolve_dir(index_dir)
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / VECTORIZER_FILENAME, "w", encoding="utf-8") as file:
        json.dump(vectorizer.get_state(), file)

    np.save(folder / MATRIX_FILENAME, matrix)

    with open(folder / CHUNKS_FILENAME, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def load_index(index_dir=None):
    """Load and return (vectorizer, matrix, chunks)."""
    folder = _resolve_dir(index_dir)
    if not index_exists(folder):
        raise FileNotFoundError(
            f"No index found in {folder}. Run 'python build_index.py' first."
        )

    with open(folder / VECTORIZER_FILENAME, "r", encoding="utf-8") as file:
        vectorizer = TfidfVectorizer.from_state(json.load(file))

    matrix = np.load(folder / MATRIX_FILENAME)

    with open(folder / CHUNKS_FILENAME, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return vectorizer, matrix, chunks


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.index_store
# (uses a temporary folder, so it never touches the real index)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    texts = [["wifi", "connect"], ["email", "login"]]
    vectorizer = TfidfVectorizer().fit(texts)
    matrix = vectorizer.transform(texts)
    chunks = [
        {"chunk_id": 0, "text": "Wi-Fi caf\u00e9 test", "source": "a.txt", "page": 1, "label": "sample"},
        {"chunk_id": 1, "text": "Email help", "source": "b.pdf", "page": 2, "label": "verified"},
    ]

    with tempfile.TemporaryDirectory() as temp_folder:
        target = Path(temp_folder) / "index"

        assert not index_exists(target)
        try:
            load_index(target)
            raise AssertionError("load_index should fail when no index exists")
        except FileNotFoundError as error:
            print("Missing index handled:", error)

        save_index(vectorizer, matrix, chunks, target)
        assert index_exists(target)
        print("Files saved:", sorted(path.name for path in target.iterdir()))

        loaded_vectorizer, loaded_matrix, loaded_chunks = load_index(target)
        assert loaded_vectorizer.vocabulary == vectorizer.vocabulary
        assert np.allclose(loaded_vectorizer.idf, vectorizer.idf)
        assert np.allclose(loaded_matrix, matrix)
        assert loaded_chunks == chunks

    print("Index store checks passed.")
    print(f"Real index present in {config.INDEX_DIR}: {index_exists()}")