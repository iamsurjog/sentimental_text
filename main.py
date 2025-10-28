import argparse
from preprocessing.normalization import normalize_text
from preprocessing.tokenization import tokenize
from preprocessing.stopword_removal import remove_stopwords
from preprocessing.lemmatization import lemmatize
from feature_extraction.word_embeddings import get_word2vec_embeddings, get_glove_embeddings
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
    print(f"Processed Tokens: {processed_tokens}")

    # Feature Extraction Layer
    word2vec_embeddings = get_word2vec_embeddings(processed_tokens)
    print(f"Word2Vec Embeddings (placeholder): {word2vec_embeddings}")

    glove_embeddings = get_glove_embeddings(processed_tokens)
    print(f"GloVe Embeddings (placeholder): {glove_embeddings}")

    bert_embeddings = get_bert_embeddings(raw_text) # BERT typically works on raw text
    print(f"BERT Embeddings (placeholder): {bert_embeddings}")

    # Semantic Analysis Module
    print("\n--- Semantic Analysis Module ---")
    semantic_features = extract_semantic_features(word2vec_embeddings + bert_embeddings) # Combine embeddings for semantic features
    print(f"Semantic Features (placeholder): {semantic_features}")

    context_understanding = understand_context(raw_text, word2vec_embeddings + bert_embeddings)
    print(f"Context Understanding (placeholder): {context_understanding}")

    syntactic_analysis_result = perform_syntactic_analysis(processed_tokens)
    print(f"Syntactic Analysis (placeholder): {syntactic_analysis_result}")

    # Deep Learning Models
    print("\n--- Deep Learning Models ---")
    # Apply attention to semantic features
    attention_features = apply_attention(semantic_features)
    print(f"Attention Features (placeholder): {attention_features}")

    # Run models
    lstm_output = run_lstm(attention_features)
    print(f"LSTM Output (placeholder): {lstm_output}")

    rnn_output = run_rnn(attention_features)
    print(f"RNN Output (placeholder): {rnn_output}")

    transformer_output = run_transformer(attention_features)
    print(f"Transformer Output (placeholder): {transformer_output}")

    # Prediction Module
    print("\n--- Prediction Module ---")
    # Using transformer_output as a placeholder input for prediction tasks
    text_classification_result = classify_text(transformer_output)
    next_word_prediction_result = predict_next_word(transformer_output)
    sentiment_analysis_result = analyze_sentiment(transformer_output)

    # Output Layer
    predictions = {
        "Text Classification": text_classification_result,
        "Next Word Prediction": next_word_prediction_result,
        "Sentiment Analysis": sentiment_analysis_result,
    }
    display_results(predictions)

if __name__ == '__main__':
    main()