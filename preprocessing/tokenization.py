import nltk

def tokenize(text):
    """
    Tokenizes a string of text.

    Args:
        text: The string to tokenize.

    Returns:
        A list of tokens.
    """
    return nltk.word_tokenize(text)