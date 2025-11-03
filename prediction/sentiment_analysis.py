import torch
import torch.nn.functional as F

def analyze_sentiment(model_output):
    """
    Analyzes the sentiment from the model output.

    Args:
        model_output: The output from the deep learning model (logits).

    Returns:
        A sentiment label.
    """
    print("Analyzing sentiment...")
    
    if not isinstance(model_output, torch.Tensor):
        # Handle case where model_output is not a tensor (e.g., "Not available")
        return "Not available"

    # The model output is the logits.
    # Apply softmax to get probabilities.
    probabilities = F.softmax(model_output, dim=1)
    
    # Get the predicted class id
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # The labels are positive and negative.
    labels = ["positive", "negative"]
    
    return labels[predicted_class_id]
