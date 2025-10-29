import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the text using a pre-trained DistilBERT model.

    Args:
        text: The input text to analyze.

    Returns:
        A sentiment label.
    """
    print("Analyzing sentiment...")
    
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
    labels = ["negative", "positive"]
    
    return labels[predicted_class_id]