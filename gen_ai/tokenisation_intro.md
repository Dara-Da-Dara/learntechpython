# Tokenization in Generative AI

Tokenization is a core concept in generative AI and large language models (LLMs). This document explains what tokenization is, types of tokenization, token limits and context windows, max/min tokens, how tokens relate to model evaluation, and how pricing is calculated based on tokens.

---

## 1. What is Tokenization?

Tokenization is the process of breaking raw text into smaller units called **tokens** that a model can process and understand.  
Tokens can be:
- Whole words (e.g., `Hello`)
- Subwords (e.g., `genera`, `##tion`)
- Characters (e.g., `A`, `I`)
- Punctuation or special symbols (e.g., `!`, `?`)

Models operate on these tokens by mapping each token to an integer ID and then to a vector (embedding). This is how human language is converted into a numerical form that neural networks can work with.

---

## 2. Types of Tokenization

### 2.1 Word Tokenization
- Splits text into words, usually on whitespace and simple punctuation.
- Example:  
  `"Hello, world!"` → `["Hello", ",", "world", "!"]`
- Simple but struggles with:
  - Out-of-vocabulary words
  - Morphologically rich languages
  - Misspellings

### 2.2 Character Tokenization
- Splits text into individual characters.
- Example:  
  `"AI"` → `["A", "I"]`
- Advantages:
  - No out-of-vocabulary problem.
- Disadvantages:
  - Sequences are very long.
  - Makes training more computationally expensive.

### 2.3 Subword Tokenization (used in modern LLMs)
Subword methods balance vocabulary size and flexibility by splitting words into smaller units when needed.

Common algorithms:

1. **Byte Pair Encoding (BPE)**  
   - Starts with characters as tokens.  
   - Iteratively merges the most frequent adjacent pairs into new tokens.  
   - Common in GPT-style models.

2. **WordPiece**  
   - Used in models like BERT.  
   - Builds a vocabulary by selecting subwords that maximize the likelihood of the training corpus.  
   - Often uses markers like `##` to indicate tokens that are continuations of a word.

3. **Unigram**  
   - Starts with a large candidate vocabulary.  
   - Iteratively removes tokens that contribute the least to modeling the data.  
   - Probabilistic and more flexible than strict merge-based methods.

### 2.4 Sentence Tokenization
- Splits text into sentences.  
- Example:  
  `"I love AI. It is fascinating!"` →  
  `["I love AI.", "It is fascinating!"]`
- Useful for sentence-level tasks (e.g., summarization, translation, evaluation).

---

## 3. Token Limits and Context Windows

### 3.1 Context Window

The **context window** (or context length) is the maximum number of tokens an LLM can handle in a single request.  
It includes:
- Input tokens (prompt)
- Output tokens (model’s response)

For example:
- If a model has a 8,000-token context window:
  - You could send 6,000 tokens of input and allow up to 2,000 tokens of output.
  - If you exceed 8,000 total tokens, some input must be truncated or omitted.

Different models have different context windows (e.g., 8K, 32K, 128K, etc.).

### 3.2 Why Context Window Matters

- Limits the amount of information the model can "see" at once.
- Affects:
  - How much context you can provide (documents, chat history, instructions).
  - How long an answer you can request.
- Forces design decisions:
  - Chunking documents for RAG systems.
  - Summarizing long histories.
  - Choosing what context is truly necessary.

---

## 4. Max Tokens and Min Tokens

Most APIs expose parameters controlling generation length:

### 4.1 `max_tokens`

- Maximum number of **output** tokens the model is allowed to generate.
- Important for:
  - **Cost control**: Fewer max tokens → lower worst-case cost for a call.
  - **Latency control**: Fewer tokens → response returns faster.
  - **Quota management**: Some systems reserve quota based on `max_tokens`, so setting it too high can block other requests.

Example:
- Context window = 8,000 tokens
- Input = 3,000 tokens
- `max_tokens = 2,000`  
  → You are safe (3,000 + 2,000 = 5,000 < 8,000).

### 4.2 `min_tokens`

- Some systems expose a minimum number of tokens to generate (or a similar concept like “minimum length”).
- Ensures the model does not stop too early, useful when:
  - You always want a sufficiently detailed answer.
- Less widely used than `max_tokens` and sometimes not exposed as an explicit parameter.

---

## 5. Model Evaluation Using Tokens

Tokens are central to evaluation and monitoring of LLMs in production.

### 5.1 Token Count Metrics

You often track:

- **Input tokens**  
  - Total input tokens processed over time.
  - Average, min, max per request.
- **Output tokens**  
  - Total output tokens generated.
  - Average, min, max per response.

These metrics help measure:

- Workload volume.
- User behavior (short vs long prompts).
- Cost drivers (which features or users are expensive).

### 5.2 Performance: Tokens per Second (TPS)

Throughput is often measured in tokens per second:

- **Prompt TPS**: How fast the model ingests tokens.
- **Generation TPS**: How fast the model produces new tokens.

Higher TPS → lower latency and better scalability.

You might track:

- Median / p95 latency for generating N tokens.
- Tokens per second per model and per hardware type (CPU vs GPU).

### 5.3 Quality Metrics (Indirectly Related)

While not token-specific, typical evaluation metrics often involve token-level behavior:

- Perplexity (on token sequences).
- Token-level accuracy for tasks like language modeling or masked token prediction.
- Error analysis at token or span level (e.g., hallucinations, omissions).

---

## 6. Pricing Based on Tokens

Most commercial LLM APIs bill **per token**, often with different prices for input and output.

### 6.1 Input vs Output Token Pricing

Common patterns:

- **Input tokens**: Cheaper (model just reads them).
- **Output tokens**: More expensive (model performs full generation).

Prices are typically quoted as:
- Cost per 1,000 tokens, or
- Cost per 1,000,000 tokens.

Example structure (illustrative only):

- Model A:
  - Input: \$0.50 per 1M tokens
  - Output: \$1.50 per 1M tokens

### 6.2 Cost Calculation Example

Suppose:

- Price:
  - \$0.02 per 1,000 input tokens
  - \$0.04 per 1,000 output tokens
- Request:
  - 1,500 input tokens
  - 500 output tokens

Cost:
- Input:  
  1,500 / 1,000 × \$0.02 = \$0.03  
- Output:  
  500 / 1,000 × \$0.04 = \$0.02  
- Total: \$0.05 for this call.

### 6.3 Other Pricing/Quota Concepts

Vendors may also define:

- **Tokens per minute (TPM)** or per day:
  - Rate limits on how many tokens you can process in a time window.
- **Cached tokens**:
  - Discounted pricing for prompts that are reused and served from cache.
- **Tiered pricing**:
  - Different prices by model, latency, or context size.

---

## 7. How These Concepts Connect

Putting it all together:

- **Tokenization** converts human text into tokens that models can handle.
- **Types of tokenization** (word, subword, character) affect how robustly models handle rare words, morphology, and different languages.
- The **context window** defines how many tokens (input + output) fit in a single interaction.
- **Max/min token parameters** control how long outputs can be, directly influencing latency and cost.
- **Model evaluation** uses token counts and token throughput to understand performance and usage.
- **Pricing** is almost always token-based, so understanding tokens is essential for cost planning and optimization.

This is why tokens are often called the “language and currency” of generative AI: they define what the model sees, how it works, how it is measured, and how it is billed.

Tokenization breaks text into tokens.

Token IDs convert tokens into numbers.

Encode adds model-specific special tokens.

Decode converts numbers back into text.
---
# Install transformers (if not installed)
# !pip install transformers

from transformers import AutoTokenizer

# Load a pretrained tokenizer (WordPiece tokenizer of BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Generative AI is amazing!"

# --- STEP 1: Tokenization (split text into small pieces) ---
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# --- STEP 2: Convert tokens → token IDs (numbers) ---
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)

# --- STEP 3: Encode (tokenization + add special tokens) ---
encoded = tokenizer.encode(text)
print("Encoded with special tokens:", encoded)

# --- STEP 4: Decode (convert IDs → text) ---
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
---


## ✅ What Are Token IDs?
Token IDs are **numbers** that represent each token.

Neural networks cannot understand words directly—they only understand **numbers**—so every token is mapped to a unique ID from the model’s vocabulary.

Example:
Generative AI is amazing!"
→ ["generative", "ai", "is", "amazing", "!"]
"generative" → 48219
"ai" → 993
"is" → 2003
"amazing" → 6429
