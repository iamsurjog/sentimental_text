
from gensim.models import Word2Vec
import numpy as np
import gensim.downloader as api

def get_word2vec_embeddings(tokens):
    """
    Generates Word2Vec embeddings for a list of tokens using a pre-trained model.
    Args:
        tokens: A list of tokens.
    Returns:
        A list of Word2Vec embeddings.
    """
    print("Generating Word2Vec embeddings...")
    # Load pre-trained Word2Vec model
    try:
        model = api.load("word2vec-google-news-300")
    except ValueError:
        print("Downloading word2vec-google-news-300 model...")
        model = api.load("word2vec-google-news-300")
        
    return [model[token] if token in model else np.zeros(300) for token in tokens]

def load_glove_model(glove_file):
    """
    Loads a GloVe model from a file.

    Args:
        glove_file: The path to the GloVe file.

    Returns:
        A dictionary mapping words to their embeddings.
    """
    print(f"Loading GloVe model from {glove_file}...")
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_glove_embeddings(tokens, embeddings_index, embedding_dim=100):
    """
    Generates GloVe embeddings for a list of tokens.

    Args:
        tokens: A list of tokens.
        embeddings_index: A dictionary mapping words to their embeddings.
        embedding_dim: The dimension of the embeddings.

    Returns:
        A list of GloVe embeddings.
    """
    print("Generating GloVe embeddings...")
    return [embeddings_index.get(token, np.zeros(embedding_dim)) for token in tokens]
