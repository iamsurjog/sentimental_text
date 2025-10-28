import nltk

def perform_syntactic_analysis(tokens):
    """
    Performs syntactic analysis on a list of tokens.

    Args:
        tokens: A list of tokens.

    Returns:
        A representation of the syntactic structure.
    """
    print("Performing syntactic analysis...")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("NLTK's 'averaged_perceptron_tagger' not found. Please download it by running: python -m nltk.downloader averaged_perceptron_tagger")
        return []
    return nltk.pos_tag(tokens)