from gensim.models import Word2Vec
import numpy as np

def get_word2vec_embeddings(tokens):
    """
    Generates Word2Vec embeddings for a list of tokens.
    Args:
        tokens: A list of tokens.
    Returns:
        A list of Word2Vec embeddings.
    """
    print("Generating Word2Vec embeddings...")
    model = Word2Vec([tokens], min_count=1)
    return [model.wv[token] for token in tokens]

def get_glove_embeddings(tokens):
    """
    Generates GloVe embeddings for a list of tokens.

    Args:
        tokens: A list of tokens.

    Returns:
        A list of GloVe embeddings.
    """
    # In a real-world scenario, you would load a pre-trained GloVe model.
    # For this example, we'll simulate the process with a dummy dictionary.
    print("Generating GloVe embeddings (dummy implementation)...")
    dummy_glove_model = {
        "the": np.array([0.1, 0.2, 0.3]),
        "quick": np.array([0.4, 0.5, 0.6]),
        "brown": np.array([0.7, 0.8, 0.9]),
        "fox": np.array([0.1, 0.3, 0.5]),
    }
    return [dummy_glove_model.get(token, np.zeros(3)) for token in tokens]