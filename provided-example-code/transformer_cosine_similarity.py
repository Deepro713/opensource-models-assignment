from transformers import pipeline
import sys
import numpy as np

def get_embedding(text, extractor):
    # Extract embeddings (returns nested lists)
    embeddings = extractor(text)

    # embeddings shape: [1, seq_len, hidden_size]
    # Take mean across seq_len dimension to get sentence embedding
    sentence_embedding = np.mean(embeddings[0], axis=0)
    return sentence_embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <text1> <text2>")
        sys.exit(1)

    text1, text2 = sys.argv[1], sys.argv[2]

    # Load the feature-extraction pipeline once
    extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

    emb1 = get_embedding(text1, extractor)
    emb2 = get_embedding(text2, extractor)

    sim = cosine_similarity(emb1, emb2)
    print(f"Similarity between '{text1}' and '{text2}': {sim:.4f}")