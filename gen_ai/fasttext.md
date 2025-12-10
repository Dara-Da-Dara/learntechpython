# Word Embeddings: GloVe and FastText

## GloVe (Global Vectors for Word Representation)

**GloVe** is a type of **word embedding** technique that converts words into numerical vectors so that machines can understand the meaning of words based on their context. Unlike simple one-hot encoding, GloVe captures **semantic relationships** between words, e.g., “king - man + woman ≈ queen.”

It is based on **matrix factorization of word co-occurrence** in a large corpus.

### How GloVe Works

1. **Build a co-occurrence matrix:** Count how often each word appears with every other word in a corpus.
2. **Transform counts into probabilities:** Compute how likely words appear together.
3. **Factorize the matrix:** Learn vectors that approximate these probabilities.
4. **Get word embeddings:** Each word is now represented as a vector in multi-dimensional space.

### Python Example

```python
from gensim.models import KeyedVectors

glove_file = 'glove.6B.100d.txt'
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
vector_king = model['king']
similar_words = model.most_similar('king', topn=5)
print(vector_king)
print(similar_words)
```

---

## FastText

**FastText** is another word embedding technique developed by Facebook AI. Unlike GloVe, FastText represents each word as a **bag of character n-grams**, which helps it understand **subword information**. This is particularly useful for:

* Rare words
* Misspellings
* Morphologically rich languages

### How FastText Works

1. Break each word into **character n-grams** (e.g., “where” → <w>wh, whe, her, ere, re</w>).
2. Learn embeddings for each n-gram.
3. The word vector is the **sum of its n-gram vectors**.

### Python Example

```python
# Install fasttext if not installed
# pip install fasttext

import fasttext

# Train a FastText model on your corpus
model = fasttext.train_unsupervised('corpus.txt', model='skipgram')

# Get vector for a word
vector_king = model.get_word_vector('king')
print("Vector for 'king':", vector_king)

# Find similar words
similar_words = model.get_nearest_neighbors('king')
print(similar_words)
```

### Key Points

* FastText can generate vectors for **out-of-vocabulary words**.
* Captures **subword information**, making it robust to misspellings.
* Works well for languages with rich morphology.

---

This file now includes both **GloVe** and **FastText** embeddings with theory and Python examples.
