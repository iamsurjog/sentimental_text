
import random

def predict_next_word(model_output):
    """
    Predicts the next word based on the model output.

    Args:
        model_output: The output from the deep learning model.

    Returns:
        The predicted next word.
    """
    print("Predicting next word...")
    words = ["apple", "banana", "cherry", "date"]
    return random.choice(words)
