import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def classify_text(text):
    """
    Classifies the text using a pre-trained DistilBERT model.

    Args:
        text: The input text to classify.

    Returns:
        A classification label.
    """
    print("Classifying text...")
    
    # Load pre-trained model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    except OSError:
        print(f"Downloading model {model_name}...")
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # The model is fine-tuned for sentiment analysis, so the labels are positive and negative.
    # We will map them to more general-purpose labels for this example.
    labels = ["negative", "positive"]
    
    return labels[predicted_class_id]