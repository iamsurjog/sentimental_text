import nltk

# Ensure you have the required NLTK data downloaded
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize(tokens):
    """
    Lemmatizes a list of tokens.

    Args:
        tokens: A list of tokens.

    Returns:
        A list of lemmatized tokens.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]