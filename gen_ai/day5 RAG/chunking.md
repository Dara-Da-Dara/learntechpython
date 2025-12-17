# Chunking Types with Python Examples

Chunking is the process of splitting text or data into smaller, manageable pieces. In NLP, chunking often groups tokens into meaningful phrases. In data processing, it refers to splitting large datasets for easier processing.

---

## 1. Fixed-size Chunking

Split text or data into chunks of a fixed size.

```python
def fixed_size_chunking(text, size):
    return [text[i:i+size] for i in range(0, len(text), size)]

text = "This is an example of fixed size chunking."
chunks = fixed_size_chunking(text, 10)
print(chunks)
```

**Output:**

```
['This is an', ' example o', 'f fixed si', 'ze chunkin', 'g.']
```

---

## 2. Sentence-based Chunking

Split text based on sentences.

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = "This is the first sentence. Here is the second. And finally the third."
chunks = sent_tokenize(text)
print(chunks)
```

**Output:**

```
['This is the first sentence.', 'Here is the second.', 'And finally the third.']
```

---

## 3. Word-based Chunking

Split text into chunks of `n` words.

```python
def word_chunking(text, n):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(0, len(words), n)]

text = "This is an example of word-based chunking in Python."
chunks = word_chunking(text, 3)
print(chunks)
```

**Output:**

```
['This is an', 'example of word-based', 'chunking in Python.']
```

---

## 4. POS-based Chunking (Noun Phrases)

Using NLTK for chunking phrases based on Part-of-Speech tagging.

```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)

# Define chunk grammar
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(pos_tags)
tree.draw()  # Opens a tree visualization
```

This identifies **Noun Phrases (NP)** like “The quick brown fox” or “the lazy dog”.

---

## 5. Dataframe/CSV Chunking

For large datasets, you can read CSV in chunks to save memory.

```python
import pandas as pd

chunk_size = 1000
chunks = pd.read_csv('large_dataset.csv', chunksize=chunk_size)

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i}")
    print(chunk.head())
```

---

## Summary Table of Chunking Types

| Type           | Purpose                                 | Example Use Case                    |
| -------------- | --------------------------------------- | ----------------------------------- |
| Fixed-size     | Split text/data into equal-sized chunks | Text processing, memory handling    |
| Sentence-based | Split text into sentences               | NLP preprocessing, summarization    |
| Word-based     | Split text into n-word chunks           | Context window for LLMs             |
| POS-based      | Chunk phrases based on part-of-speech   | Extract noun phrases, entity chunks |
| Dataframe/CSV  | Split large datasets into smaller parts | Big data processing, ETL            |
