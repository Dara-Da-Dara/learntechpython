# GloVe (Global Vectors for Word Representation)

## What is GloVe?

**GloVe** is a type of **word embedding** technique that converts words into numerical vectors so that machines can understand the meaning of words based on their context. Unlike simple one-hot encoding, GloVe captures **semantic relationships** between words, e.g., “king - man + woman ≈ queen.”

It is based on **matrix factorization of word co-occurrence** in a large corpus. In simpler terms, it counts how often words appear together and uses that to learn their relationships.

---

## How GloVe Works

1. **Build a co-occurrence matrix:** Count how often each word appears with every other word in a corpus.
2. **Transform counts into probabilities:** Compute how likely words appear together.
3. **Factorize the matrix:** Learn vectors that approximate these probabilities.
4. **Get word embeddings:** Each word is now represented as a vector in multi-dimensional space (e.g., 50, 100, 200 dimensions).

---

## Example Use Case

* **Semantic similarity:** Find words with similar meaning.
  Example: “Paris” is close to “France”, “king” is close to “queen”.
* **Text classification:** Convert words into vectors before feeding into a neural network.
* **Clustering:** Group words with similar meanings.

---

## Python Example

```python
# Install Gensim if not installed
# pip install gensim

from gensim.models import KeyedVectors

# Load pre-trained GloVe vectors (example: 100-dimensional)
glove_file = 'glove.6B.100d.txt'  # download from https://nlp.stanford.edu/projects/glove/
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# Get vector for a word
vector_king = model['king']
print("Vector for 'king':", vector_king)

# Find similar words
similar_words = model.most_similar('king', topn=5)
print("Words similar to 'king':", similar_words)
```

---

## Key Points

* GloVe is **unsupervised**.
* Captures **global context** using co-occurrence statistics.
* Produces **dense vectors** (vs sparse one-hot vectors).
* Works well for **semantic similarity** and **text analysis**.
