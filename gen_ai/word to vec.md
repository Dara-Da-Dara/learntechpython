# 🧠 Understanding Embeddings in NLP & Generative AI

Embeddings are numeric vector representations of words, sentences, or documents that capture semantic meaning. They are essential in **Natural Language Processing (NLP)** and **Generative AI**, enabling models to understand and generate text.

---

## 1. What are Embeddings?

**Definition:**  
Embeddings are multi-dimensional vectors that represent words, phrases, or sentences in a continuous space. Similar meanings are close together.

**Example:**  
- `"king"` and `"queen"` → close vectors  
- `"apple"` and `"banana"` → close vectors  
- `"king"` and `"apple"` → far apart

Embeddings allow AI to understand **semantic similarity**.

---

## 2. Word2Vec

**Definition:**  
Word2Vec is an algorithm to create word embeddings. It captures meaning based on **context**.

**Architectures:**  
1. **CBOW (Continuous Bag of Words):** Predicts a word from surrounding words.  
2. **Skip-gram:** Predicts surrounding words given a target word.

**Python Example:**

```python
from gensim.models import Word2Vec

sentences = [
    ["I", "love", "football"],
    ["She", "enjoys", "playing", "soccer"],
    ["He", "loves", "basketball"]
]

model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=4)

vector = model.wv['football']
print(vector)


output :
Vector for 'football': [0.12, -0.03, 0.44, ...]
Most similar words: [('soccer', 0.92), ('basketball', 0.81)]

Use Cases:

Semantic search

Chatbots

Text classification


similar_words = model.wv.most_similar('football', topn=3)
print(similar_words)
