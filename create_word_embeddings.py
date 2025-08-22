from nltk.corpus import words , wordnet
from wordfreq import word_frequency
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

codenames_corpus = pd.read_csv('corpus.csv').values.tolist()
codenames_corpus = set(codenames_corpus)
raw_vocab = {w.lower() for w in words.words() if w.isalpha()}

wordnet_vocab = set(wordnet.words())
modern_vocab = raw_vocab & wordnet_vocab

filtered_vocab = {w for w in modern_vocab if word_frequency(w, 'en') > 1e-5}

function_words = {"a", "i", "the", "an", "of", "in", "on", "to", "at", "and", "or"}
final_vocab = filtered_vocab | codenames_corpus - function_words

print("Raw vocab:", len(raw_vocab))
print("Modern vocab:", len(modern_vocab))
print("Filtered vocab:", len(filtered_vocab))
print("Final vocab:", len(final_vocab))

corpus = list(final_vocab) 

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(corpus, convert_to_numpy=True).astype("float32")


dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim) #Cosine similarity
index.add(embeddings)

print(f"Stored {index.ntotal} word embeddings.")

id_to_word = {i: w for i, w in enumerate(corpus)}

query = model.encode(["leg", "foot", "lion"], convert_to_numpy=True).astype("float32")
group_query = np.mean(query, axis=0, keepdims=True)
D, I = index.search(group_query, k=10)

for w, s in dict(zip([id_to_word[i] for i in I[0]], D[0])).items():
    print(f"{w} ||| Score: {s:.4f}")

faiss.write_index(index, "word_embeddings.index")
np.save("id_to_word.npy", id_to_word)
