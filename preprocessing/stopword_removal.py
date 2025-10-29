
from nltk.corpus import stopwords
import nltk

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.

    Args:
        tokens: A list of tokens.

    Returns:
        A list of tokens with stopwords removed.
    """
    return [token for token in tokens if token.lower() not in STOPWORDS]
