
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_semantic_features(tokens, embeddings):
    """
    Extracts semantic features from given embeddings using TF-IDF weighting.

    Args:
        tokens: A list of tokens.
        embeddings: A list of word embeddings.

    Returns:
        A representation of semantic features.
    """
    print("Extracting semantic features...")
    if not embeddings or not tokens:
        # Ensure a consistent return shape
        return np.zeros(300)  # Default to a common embedding size

    # Convert all embeddings to numpy arrays
    embeddings = [np.array(e) for e in embeddings]

    # Ensure all embeddings have the same shape
    if len(set(e.shape for e in embeddings)) > 1:
        # Handle inconsistent shapes, e.g., by padding or averaging
        # For simplicity, we'll average them here
        max_len = max(e.shape[0] for e in embeddings)
        embeddings = [np.pad(e, (0, max_len - e.shape[0]), 'constant') if e.shape[0] < max_len else e for e in embeddings]

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    try:
        tfidf_matrix = vectorizer.fit_transform([" ".join(tokens)])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = {word: tfidf_matrix[0, i] for i, word in enumerate(feature_names)}
    except ValueError:
        return np.mean(embeddings, axis=0)

    weighted_embeddings = []
    for i, token in enumerate(tokens):
        if i < len(embeddings):
            weight = tfidf_scores.get(token, 0)
            weighted_embeddings.append(weight * embeddings[i])

    if not weighted_embeddings:
        return np.mean(embeddings, axis=0)

    return np.sum(weighted_embeddings, axis=0)
