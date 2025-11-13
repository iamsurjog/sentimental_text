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
    parser.add_argument("text", type=str, help="The path to the input text file to analyze.")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "rnn", "transformer", "attention"], help="The type of model to use.")
    parser.add_argument("--output_type", type=str, default="sentiment_analysis", choices=["text_classification", "next_word_prediction", "sentiment_analysis"], help="The type of output to get.")
    args = parser.parse_args()

    # with open(args.file, 'r') as f:
        # raw_text = f.read()
    raw_text = args.text
    processed_tokens = preprocess_text(raw_text)

    # Feature Extraction Layer
    word2vec_embeddings = get_word2vec_embeddings(processed_tokens)
    
    glove_file = 'glove.6B.50d.txt'
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
    model_output = None
    if semantic_features.any():
        attention_features = apply_attention(torch.tensor(semantic_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        if args.model_type == "lstm":
            model_output = run_lstm(attention_features.unsqueeze(0))
        elif args.model_type == "rnn":
            model_output = run_rnn(attention_features.unsqueeze(0))
        elif args.model_type == "transformer":
            model_output = run_transformer(attention_features.unsqueeze(0))
        elif args.model_type == "attention":
            model_output = attention_features

    # Prediction Module
    prediction_result = "Not available"
    if args.output_type == "next_word_prediction":
        prediction_result = predict_next_word(raw_text)
    elif model_output is not None:
        if args.output_type == "text_classification":
            prediction_result = classify_text(model_output)
        elif args.output_type == "sentiment_analysis":
            prediction_result = analyze_sentiment(model_output)

    # Output Layer
    predictions = {
        args.output_type.replace('_', ' ').title(): prediction_result,
        "Syntactic Analysis": syntactic_analysis_result,
        "Context Understanding": context_understanding,
    }
    display_results(predictions)
    return predictions [args.output_type.replace('_', ' ').title()]

if __name__ == '__main__':
    main()
