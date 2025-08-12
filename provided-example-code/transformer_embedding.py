from transformers import pipeline
import sys

def get_embedding(text):
    # Load the feature-extraction pipeline with MiniLM v6 model
    extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

    # Extract embeddings (returns nested lists)
    embeddings = extractor(text)

    # embeddings shape: [1, seq_len, hidden_size]
    # To get a sentence embedding, take mean across seq_len dimension
    sentence_embedding = [float(sum(token_emb) / len(token_emb)) for token_emb in zip(*embeddings[0])]

    return sentence_embedding

if __name__ == "__main__":
    text = sys.argv[1]
    embedding = get_embedding(text)
    print(embedding)