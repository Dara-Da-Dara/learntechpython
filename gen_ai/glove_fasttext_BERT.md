# Word Embeddings: GloVe, FastText, and BERT

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

**FastText** is another word embedding technique developed by Facebook AI. Unlike GloVe, FastText represents each word as a **bag of character n-grams**, which helps it understand **subword information**.

### How FastText Works

1. Break each word into **character n-grams**.
2. Learn embeddings for each n-gram.
3. The word vector is the **sum of its n-gram vectors**.

### Python Example

```python
import fasttext

model = fasttext.train_unsupervised('corpus.txt', model='skipgram')
vector_king = model.get_word_vector('king')
similar_words = model.get_nearest_neighbors('king')
print(vector_king)
print(similar_words)
```

---

## BERT (Bidirectional Encoder Representations from Transformers)

**BERT** is a **contextual word embedding** model developed by Google. Unlike GloVe and FastText, BERT generates **different embeddings for the same word depending on its context**.

### How BERT Works

1. Uses **Transformer architecture** with attention mechanisms.
2. Trains with **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**.
3. Produces **contextual embeddings** for each word in a sentence.

### Python Example using Hugging Face Transformers

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode a sentence
sentence = "The king is wise"
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)

# Get embeddings for each token
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # (batch_size, sequence_length, hidden_size)
```

### Key Points

* BERT embeddings are **contextual**.
* Captures **word meaning based on surrounding words**.
* Very effective for NLP tasks like **QA, NER, classification, and translation**.

---

This file now includes **GloVe, FastText, and BERT** embeddings with theory and Python examples.
