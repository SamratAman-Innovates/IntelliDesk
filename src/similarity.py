"""
similarity.py - Manual cosine similarity with NumPy.

cosine(a, b) = (a . b) / (|a| * |b|)
Score 1.0 = same direction (very similar), 0.0 = nothing in common.
"""

import numpy as np


def cosine_similarity(query_vector, matrix):
    """Compare one query vector with every row of a matrix.

    Returns a NumPy array with one score per row.
    """
    dot_products = matrix @ query_vector
    denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)

    scores = np.zeros(len(dot_products), dtype=float)
    nonzero = denominators > 0
    scores[nonzero] = dot_products[nonzero] / denominators[nonzero]
    return scores


def top_k_indices(scores, k):
    """Return the positions of the k highest scores, best first.

    Ties keep their original order (the earlier chunk wins).
    """
    return np.argsort(-scores, kind="stable")[:k]


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.similarity
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = cosine_similarity(np.array([1.0, 0.0]), matrix)
    print("Scores:", np.round(scores, 4))
    assert np.allclose(scores, [1.0, 0.0, 0.7071], atol=1e-4)

    assert np.allclose(cosine_similarity(np.array([0.0, 0.0]), matrix), [0, 0, 0])

    zero_row = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(cosine_similarity(np.array([1.0, 0.0]), zero_row), [0.0, 1.0])

    assert np.allclose(cosine_similarity(np.array([1.0, 0.0]), matrix * 5), scores)

    ranking = top_k_indices(np.array([0.1, 0.9, 0.5, 0.9]), 3)
    print("Top 3 positions:", ranking.tolist())
    assert ranking.tolist() == [1, 3, 2]
    assert len(top_k_indices(np.array([0.2, 0.4]), 10)) == 2
    print("Similarity checks passed.")

    # --- Real knowledge base ---
    from src.chunker import chunk_documents
    from src.document_loader import load_documents
    from src.text_cleaner import clean_text, tokenize
    from src.tfidf_vectorizer import TfidfVectorizer

    documents = load_documents()
    for document in documents:
        document["text"] = clean_text(document["text"])
    chunks = chunk_documents(documents)

    token_lists = [tokenize(chunk["text"]) for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(token_lists)
    chunk_matrix = vectorizer.transform(token_lists)

    for question in [
        "How do I fix wifi with no internet access?",
        "I cannot sign in to my email",
        "Who won the cricket match yesterday?",
    ]:
        query_vector = vectorizer.transform_one(tokenize(question))
        scores = cosine_similarity(query_vector, chunk_matrix)
        assert np.allclose(scores, chunk_matrix @ query_vector)

        print(f"\nQuestion: {question}")
        for position in top_k_indices(scores, 3):
            chunk = chunks[position]
            preview = chunk["text"][:60].replace("\n", " ")
            print(f"  {scores[position]:.3f}  #{chunk['chunk_id']} {chunk['source']}  {preview}...")