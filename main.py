import argparse
import torch
import numpy as np
from preprocessing.normalization import normalize_text
from preprocessing.tokenization import tokenize
from preprocessing.stopword_removal import remove_stopwords
from preprocessing.lemmatization import lemmatize
from feature_extraction.word_embeddings import get_word2vec_embeddings, get_glove_embeddings, load_glove_model
from feature_extraction.contextual_embeddings import get_bert_embeddings
from semantic_analysis.semantic_feature_extraction import extract_semantic_features
from semantic_analysis.context_understanding import understand_context
from semantic_analysis.syntactic_analysis import perform_syntactic_analysis
from deep_learning_models.lstm_rnn import run_lstm, run_rnn
from deep_learning_models.transformer import run_transformer
from deep_learning_models.attention import apply_attention
from prediction.text_classification import classify_text
from prediction.next_word_prediction import predict_next_word
from prediction.sentiment_analysis import analyze_sentiment
from output.results import display_results

def preprocess_text(text):
    """
    Applies the full preprocessing pipeline to a string of text.
    Args:
        text: The string to preprocess.
    Returns:
        A list of preprocessed tokens.
    """
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens

def main():
    """
    The main function of the CLI tool.
    """
    parser = argparse.ArgumentParser(description="A modular CLI tool for sentiment analysis.")
    parser.add_argument("text", type=str, help="The input text to analyze.")
    args = parser.parse_args()

    raw_text = args.text
    processed_tokens = preprocess_text(raw_text)

    # Feature Extraction Layer
    word2vec_embeddings = get_word2vec_embeddings(processed_tokens)
    
    glove_file = 'glove.6B.100d.txt'
    embeddings_index = load_glove_model(glove_file)
    glove_embeddings = get_glove_embeddings(processed_tokens, embeddings_index)
    
    bert_embeddings = get_bert_embeddings(raw_text)

    # Semantic Analysis Module
    word2vec_embeddings_list = [np.array(e) for e in word2vec_embeddings]
    glove_embeddings_list = [np.array(e) for e in glove_embeddings]
    bert_embeddings_list = [np.array(e) for e in bert_embeddings.squeeze(0).tolist()]
    
    # Averaging embeddings to a common dimension (e.g., 300)
    def average_embeddings(embeddings_list, target_dim=300):
        averaged_embeddings = []
        for embeddings in embeddings_list:
            if embeddings.size > 0:
                # Pad or truncate to target_dim
                if embeddings.shape[0] < target_dim:
                    padded = np.pad(embeddings, (0, target_dim - embeddings.shape[0]), 'constant')
                else:
                    padded = embeddings[:target_dim]
                averaged_embeddings.append(padded)
        return np.mean(averaged_embeddings, axis=0) if averaged_embeddings else np.zeros(target_dim)

    combined_embeddings = average_embeddings(word2vec_embeddings_list + glove_embeddings_list + bert_embeddings_list)
    
    semantic_features = extract_semantic_features(processed_tokens, [combined_embeddings])
    context_understanding = understand_context(raw_text)
    syntactic_analysis_result = perform_syntactic_analysis(processed_tokens)

    # Deep Learning Models
    if semantic_features.any():
        attention_features = apply_attention(torch.tensor(semantic_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        # Run models
        lstm_output = run_lstm(attention_features.unsqueeze(0))
        rnn_output = run_rnn(attention_features.unsqueeze(0))
        transformer_output = run_transformer(attention_features.unsqueeze(0))

        # Prediction Module
        text_classification_result = classify_text(raw_text)
        next_word_prediction_result = predict_next_word(raw_text)
        sentiment_analysis_result = analyze_sentiment(raw_text)
    else:
        # Handle case with no semantic features
        text_classification_result = "Not available"
        next_word_prediction_result = "Not available"
        sentiment_analysis_result = "Not available"

    # Output Layer
    predictions = {
        "Text Classification": text_classification_result,
        "Next Word Prediction": next_word_prediction_result,
        "Sentiment Analysis": sentiment_analysis_result,
        "Syntactic Analysis": syntactic_analysis_result,
        "Context Understanding": context_understanding,
    }
    display_results(predictions)

if __name__ == '__main__':
    main()