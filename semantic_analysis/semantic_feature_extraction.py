import numpy as np

def extract_semantic_features(embeddings):
    """
    Extracts semantic features from given embeddings.

    Args:
        embeddings: A list of word or contextual embeddings.

    Returns:
        A representation of semantic features.
    """
    print("Extracting semantic features...")
    if len(embeddings) == 0:
        return np.zeros(300)  # Assuming a 300-dimensional embedding

    resized_embeddings = []
    for emb in embeddings:
        emb = np.array(emb)
        if emb.shape[0] != 300:
            emb = np.resize(emb, (300,))
        resized_embeddings.append(emb)

    return np.mean(resized_embeddings, axis=0)