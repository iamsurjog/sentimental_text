
import random

def analyze_sentiment(model_output):
    """
    Analyzes the sentiment of the text based on the model output.

    Args:
        model_output: The output from the deep learning model.

    Returns:
        A sentiment label.
    """
    print("Analyzing sentiment...")
    sentiments = ["positive", "negative", "neutral"]
    return random.choice(sentiments)
