
from preprocessing.normalization import normalize_text
from preprocessing.tokenization import tokenize
from preprocessing.stopword_removal import remove_stopwords
from preprocessing.lemmatization import lemmatize
from feature_extraction.word_embeddings import get_word2vec_embeddings, get_glove_embeddings
from feature_extraction.contextual_embeddings import get_bert_embeddings
from semantic_analysis.semantic_feature_extraction import extract_semantic_features
from semantic_analysis.context_understanding import understand_context
from semantic_analysis.syntactic_analysis import perform_syntactic_analysis

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

if __name__ == '__main__':
    raw_text = "This is a sample sentence, showing off the stop words filtration."
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

