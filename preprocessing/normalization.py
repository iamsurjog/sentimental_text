
import re

def normalize_text(text):
    """
    Normalizes a string of text.

    Args:
        text: The string to normalize.

    Returns:
        A normalized string.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
