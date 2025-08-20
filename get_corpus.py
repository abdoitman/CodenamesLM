import pandas as pd
import os

def get_corpus(corpus_path):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"The corpus file at {corpus_path} does not exist.")
    try:
        corpus = pd.read_csv(corpus_path, sep='\t', header=None, names=['text'])
        return corpus
    except Exception as e:
        raise ValueError(f"An error occurred while reading the corpus: {e}")

if __name__ == "__main__":
    corpus_path = 'corpus.csv'
    try:
        corpus_df = get_corpus(corpus_path)
        print("Corpus loaded successfully.")
        print(corpus_df.head())
    except Exception as e:
        print(f"Error: {e}")