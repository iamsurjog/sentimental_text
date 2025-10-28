
STOPWORDS = set(['a', 'an', 'the', 'in', 'on', 'at'])

def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.

    Args:
        tokens: A list of tokens.

    Returns:
        A list of tokens with stopwords removed.
    """
    return [token for token in tokens if token.lower() not in STOPWORDS]
