"""
tfidf_vectorizer.py - Our own TF-IDF implementation (NumPy only).

fit()       : learn the vocabulary and the IDF value of every word.
transform() : turn token lists into a matrix, one row per text,
              each row scaled to length 1 (so cosine similarity = dot product).

Formulas:
    tf  = how many times the word appears in the text
    idf = log((1 + N) / (1 + df)) + 1     N = number of texts, df = texts containing the word
    weight = tf * idf, then each row is divided by its length (L2 norm)
Words not seen during fit() are ignored.
"""

import math
from collections import Counter

import numpy as np


class TfidfVectorizer:
    def __init__(self):
        self.vocabulary = {}   # word -> column number
        self.idf = None        # NumPy array: one IDF value per word

    def fit(self, token_lists):
        """Learn vocabulary and IDF from a list of token lists."""
        n_texts = len(token_lists)
        if n_texts == 0:
            raise ValueError("Cannot fit on an empty list of texts")

        document_frequency = Counter()
        for tokens in token_lists:
            document_frequency.update(set(tokens))

        words = sorted(document_frequency)
        self.vocabulary = {word: column for column, word in enumerate(words)}
        self.idf = np.array(
            [
                math.log((1 + n_texts) / (1 + document_frequency[word])) + 1
                for word in words
            ]
        )
        return self

    def transform(self, token_lists):
        """Return a matrix of shape (number of texts, vocabulary size)."""
        matrix = np.zeros((len(token_lists), len(self.vocabulary)))

        for row, tokens in enumerate(token_lists):
            for word, count in Counter(tokens).items():
                column = self.vocabulary.get(word)
                if column is not None:
                    matrix[row, column] = count * self.idf[column]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0   # avoid dividing by zero for empty rows
        return matrix / norms

    def transform_one(self, tokens):
        """Vectorize a single token list (for example a user question)."""
        return self.transform([tokens])[0]

    def get_state(self):
        """Return plain Python data that can be saved to disk."""
        words = sorted(self.vocabulary, key=self.vocabulary.get)
        return {"words": words, "idf": self.idf.tolist()}

    @classmethod
    def from_state(cls, state):
        """Rebuild a vectorizer from data produced by get_state()."""
        vectorizer = cls()
        vectorizer.vocabulary = {
            word: column for column, word in enumerate(state["words"])
        }
        vectorizer.idf = np.array(state["idf"])
        return vectorizer


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.tfidf_vectorizer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 1. Tiny example we can verify by hand ---
    texts = [
        ["wifi", "connect", "internet"],
        ["email", "login", "password"],
        ["wifi", "password", "reset"],
    ]
    vectorizer = TfidfVectorizer().fit(texts)
    matrix = vectorizer.transform(texts)

    assert len(vectorizer.vocabulary) == 7
    assert matrix.shape == (3, 7)
    assert vectorizer.idf[vectorizer.vocabulary["wifi"]] < vectorizer.idf[vectorizer.vocabulary["connect"]]
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    scores = matrix @ vectorizer.transform_one(["wifi", "internet"])
    print("Tiny example scores:", np.round(scores, 3))
    assert scores.argmax() == 0
    assert np.isclose(scores[1], 0.0)

    assert not vectorizer.transform_one(["zzz"]).any()

    restored = TfidfVectorizer.from_state(vectorizer.get_state())
    assert np.allclose(restored.transform(texts), matrix)
    print("Tiny example checks passed.")

    # --- 2. The real knowledge base ---
    from src.chunker import chunk_documents
    from src.document_loader import load_documents
    from src.text_cleaner import clean_text, tokenize

    documents = load_documents()
    for document in documents:
        document["text"] = clean_text(document["text"])
    chunks = chunk_documents(documents)

    token_lists = [tokenize(chunk["text"]) for chunk in chunks]
    real_vectorizer = TfidfVectorizer().fit(token_lists)
    real_matrix = real_vectorizer.transform(token_lists)
    print(
        f"\n{len(chunks)} chunks, vocabulary size {len(real_vectorizer.vocabulary)}, "
        f"matrix shape {real_matrix.shape}"
    )

    questions = [
        "My phone connects to the campus Wi-Fi but there is no internet",
        "I forgot my email password",
        "Who won the cricket match yesterday?",
    ]
    for question in questions:
        query_vector = real_vectorizer.transform_one(tokenize(question))
        scores = real_matrix @ query_vector
        print(f"\nQuestion: {question}")
        for chunk, score in zip(chunks, scores):
            print(f"  {score:.3f}  #{chunk['chunk_id']} {chunk['source']}")