import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def predict_next_word(text):
    """
    Predicts the next word using a pre-trained GPT-2 model.

    Args:
        text: The input text.

    Returns:
        The predicted next word.
    """
    print("Predicting next word...")
    
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    except OSError:
        print(f"Downloading model {model_name}...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Get the logits for the last token
        last_token_logits = logits[0, -1, :]
        # Get the predicted token id
        predicted_token_id = torch.argmax(last_token_logits).item()
        # Decode the token id to a word
        predicted_word = tokenizer.decode(predicted_token_id)
        
    return predicted_word