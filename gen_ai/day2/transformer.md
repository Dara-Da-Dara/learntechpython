# Elements of Transformer Model

The **Transformer** model is the backbone of many modern NLP models like BERT, GPT, and T5. It uses **attention mechanisms** to process sequential data without relying on recurrence.

---

## 1. Input Embeddings

* Converts tokens into **dense vectors**.
* Includes **positional encoding** to retain word order.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer("Hello world", return_tensors='pt')
print(tokens)
```

---

## 2. Positional Encoding

* Adds information about the **position of each token**.
* Ensures the model knows the order of words.

---

## 3. Multi-Head Attention

* **Attention mechanism** allows the model to focus on relevant parts of the input.
* **Multiple heads** capture information from different representation subspaces.

```python
# Conceptual example using PyTorch
import torch
import torch.nn as nn

attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
query = torch.rand(10, 32, 512)  # seq_len, batch, embedding_dim
key = value = query
output, weights = attention(query, key, value)
```

---

## 4. Feed-Forward Network (FFN)

* Fully connected layers applied to each token.
* Provides non-linear transformations.

```python
ffn = nn.Sequential(
    nn.Linear(512, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512)
)
```

---

## 5. Layer Normalization

* Normalizes outputs of sub-layers.
* Helps **stabilize and speed up training**.

---

## 6. Residual Connections

* Shortcut connections that **add input to output** of sub-layers.
* Helps prevent **vanishing gradients**.

---

## 7. Encoder and Decoder Stacks

* **Encoder:** Stacks of attention + feed-forward layers.
* **Decoder:** Similar stack, but includes **masked attention** to prevent future token leakage.

---

## 8. Output Layer

* Final linear layer + softmax to predict token probabilities.

```python
output = nn.Linear(512, vocab_size)
```

---

## Summary

* Transformers use **self-attention** instead of recurrence.
* Can process sequences in parallel.
* Core elements: **Input Embeddings, Positional Encoding, Multi-Head Attention, FFN, Layer Norm, Residual Connections, Encoder-Decoder Stacks, Output Layer ** 

---

This document provides a clear overview of the **key elements of the Transformer model** with code snippets and explanations.
