from sentence_transformers import SentenceTransformer
from get_corpus import get_corpus
import pandas as pd
import numpy as np
import json

def get_embeddings(corpus_path, model_name= 'sentence-transformers/all-MiniLM-L6-v2'):
    try:
        corpus_df = get_corpus(corpus_path)
        model = SentenceTransformer(model_name)
        embeddings = model.encode(corpus_df['text'].tolist(), show_progress_bar=True)
        print(embeddings.shape)
        print(type(embeddings))
        corpus_df['embeddings'] = list(embeddings)
        corpus_df['embeddings'] = corpus_df['embeddings'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        corpus_dict = corpus_df.set_index('text')['embeddings'].to_dict()
        # Save embeddings to a JSON file
        with open('embeddings.json', 'w') as f:
            json.dump(corpus_dict, f)
        print("Embeddings saved to embeddings.json")
        return corpus_df
    except Exception as e:
        raise ValueError(f"An error occurred while getting embeddings: {e}")

def load_embeddings(file_path='embeddings.json') -> pd.DataFrame:
    try:
        with open(file_path, 'r') as f:
            embeddings_dict = json.load(f)
        embeddings_df = pd.DataFrame.from_dict(embeddings_dict, orient='index')
        return embeddings_df
    except Exception as e:
        raise ValueError(f"An error occurred while loading embeddings: {e}")

if __name__ == "__main__":
    corpus_path = 'corpus.csv'
    try:
        embeddings_df = get_embeddings(corpus_path)
        print("Embeddings generated successfully.")
        print(embeddings_df.head())
    except Exception as e:
        print(f"Error: {e}")