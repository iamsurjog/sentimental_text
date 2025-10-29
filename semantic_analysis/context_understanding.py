
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def understand_context(text, n_topics=1, n_words=5):
    """
    Analyzes text for context understanding using Latent Dirichlet Allocation (LDA).

    Args:
        text: The original or preprocessed text.
        n_topics: The number of topics to identify.
        n_words: The number of words to describe each topic.

    Returns:
        A dictionary representing the text's context.
    """
    print("Understanding context with LDA...")
    vectorizer = CountVectorizer(stop_words='english')
    try:
        doc_term_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {"topics": []}

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_indices = topic.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics.append(top_words)

    return {"topics": topics}
