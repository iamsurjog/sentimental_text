
import random

def classify_text(model_output):
    """
    Classifies the text based on the model output.

    Args:
        model_output: The output from the deep learning model.

    Returns:
        A classification label.
    """
    print("Classifying text...")
    classes = ["sports", "politics", "technology", "entertainment"]
    return random.choice(classes)
