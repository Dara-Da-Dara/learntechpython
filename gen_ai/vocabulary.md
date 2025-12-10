# 📘 Generative AI Vocabulary — 100+ Terms with Examples

This document contains 100 essential Generative AI terms with simple definitions and examples.  
Focus: LLMs, embeddings, prompts, RAG, multimodal AI, APIs, generation, and related concepts.  
(ML/DL-specific terms like gradient descent, loss, epochs, parameters removed.)
# AI and LLM Concepts Explained

This document explains key terms in AI and Large Language Models (LLMs) with examples and analogies.

---

## 1. Parameters in Machine Learning and LLMs

* **Definition:** Parameters are the internal "knobs" a model adjusts during training to learn patterns from data. In ML, they include weights and biases; in LLMs, parameters are millions or billions of numbers that determine how the model generates text.
* **Example in ML:** In a linear regression model, the slope and intercept are parameters.
* **Example in LLM:** GPT-3 has 175 billion parameters that encode language patterns.
* **Analogy:** Parameters are like the settings of a complex machine — in LLMs, each number helps the model understand and generate text.

---

## 2. Attention Mechanisms

* **Definition:** Allows a model to focus on the most relevant parts of input data when making predictions.
* **Example:** In language models, attention helps figure out which words in a sentence matter most for predicting the next word.
* **Analogy:** Reading a paragraph but highlighting only the key sentences before answering a question.
* **Tiny Code Example:**

```python
# Pseudo-code for attention weights
sentence = ["I", "love", "AI"]
attention_weights = [0.1, 0.7, 0.2]  # model focuses more on 'love'
context_vector = sum(word_vec * weight for word_vec, weight in zip(sentence, attention_weights))
```

---

## 3. Pre-training

* **Definition:** Training a model on a large, general dataset before fine-tuning it for a specific task.
* **Example:** GPT is pre-trained on vast amounts of internet text to learn general language patterns.
* **Analogy:** Learning general knowledge in school before specializing in medicine or engineering.
* **Tiny Code Example:**

```python
# Pre-training a simple model
# Imagine training on general text corpus
model.train(general_corpus)
```

---

## 4. Fine-tuning

* **Definition:** Adapting a pre-trained model to a specific task or dataset.
* **Example:** Fine-tuning GPT on legal documents for accurate legal Q&A.
* **Analogy:** A chef trained in general cooking learning how to specialize in Italian cuisine.
* **Tiny Code Example:**

```python
# Fine-tuning on a specific dataset
model.load_pretrained('GPT')
model.train(legal_corpus)
```

---

## 5. Zero-shot Learning

* **Definition:** The ability of a model to perform a task it hasn’t explicitly been trained for, using general knowledge.
* **Example:** Asking a model to translate a sentence into a language it never saw during training.
* **Analogy:** Playing a new board game correctly just by reading the rules, without practicing.
* **Tiny Code Example:**

```python
# Zero-shot example
prompt = "Translate 'Hello' to French"
output = model.generate(prompt)
print(output)  # Output: 'Bonjour'
```

---

## 6. Few-shot Learning

* **Definition:** Model learns a task with only a few examples.
* **Example:** Providing 3 examples of sentiment analysis, then asking the model to predict sentiment for new sentences.
* **Analogy:** Learning a game by watching only a few rounds.

---

## 7. Tokenization

* **Definition:** Splitting text into small units (words, subwords, or characters) that the model can process.
* **Example:** "I love AI" → ['I', 'love', 'AI']

---

## 8. Embeddings

* **Definition:** Converting words into numeric vectors so the model can understand relationships between them.
* **Example:** 'king' and 'queen' have similar embeddings.
* **Tiny Code Example:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
sentences = ['I love AI', 'AI is amazing']
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(sentences)
print(vectors.toarray())
```

---

## Generative AI Vocabulary

This section introduces key Generative AI terms and concepts for better understanding of AI models and applications.

---

This document provides a beginner-friendly overview of key AI/LLM concepts with examples and analogies.

---

## 1. Token
**Definition:** Smallest unit of text an AI reads.  
**Example:** `"Apple is red"` → `["Apple", "is", "red"]`

## 2. Tokenization
**Definition:** Breaking text into tokens.  
**Example:** `"I love AI"` → `["I", "love", "AI"]`

## 3. Token ID
**Definition:** Numeric ID assigned to a token.  
**Example:** `"AI"` → `12345`

## 4. Vocabulary
**Definition:** All tokens a model knows.  
**Example:** `"cat", "dog", "AI"` are in the vocabulary.

## 5. Embedding
**Definition:** Vector representing meaning.  
**Example:** `"king"` → `[0.12, 0.88, -0.33, ...]`

## 6. Embedding Space
**Definition:** High-dimensional space where embeddings live.  
**Example:** `"king"` and `"queen"` vectors are close.

## 7. Vector
**Definition:** List of numbers representing meaning.  
**Example:** `[0.12, -0.44, 0.98]`

## 8. Cosine Similarity
**Definition:** Measures similarity between vectors.  
**Example:** `"king"` vs `"queen"` → similarity ≈ 0.92

## 9. Attention
**Definition:** Mechanism to focus on important words.  
**Example:** In `"The cat sat on the mat"`, focus on `"cat"`.

## 10. Self-Attention
**Definition:** Attention within the same sequence.  
**Example:** `"I love AI"` → AI attends to `"love"` when processing `"I"`.

## 11. Multi-Head Attention
**Definition:** Multiple attention mechanisms running in parallel.  
**Example:** Helps the model understand multiple aspects of a sentence.

## 12. Transformer
**Definition:** Neural network architecture used in LLMs.  
**Example:** GPT, BERT, T5 models are based on transformers.

## 13. LLM (Large Language Model)
**Definition:** Model trained to understand and generate text.  
**Example:** GPT-4, Claude

## 14. Prompt
**Definition:** Text given to a model to generate output.  
**Example:** `"Write a poem about space"`

## 15. Prompt Engineering
**Definition:** Crafting prompts to improve AI output.  
**Example:** `"Explain black holes like I’m 5."`

## 16. Persona
**Definition:** Role or identity assigned to the AI.  
**Example:** `"You are a Python tutor."`

## 17. Zero-Shot Learning
**Definition:** AI performs tasks without examples.  
**Example:** `"Translate Hindi to French"`

## 18. Few-Shot Learning
**Definition:** AI learns from a few examples in the prompt.  
**Example:** `"Apple → Fruit, Carrot → Vegetable"`

## 19. Context Window
**Definition:** Maximum text the model can process at once.  
**Example:** GPT-4o-mini → 128k tokens

## 20. Inference
**Definition:** Generating output from a trained AI model.  
**Example:** `"Summarize this article"` → AI produces summary.

## 21. Hallucination
**Definition:** AI produces incorrect or made-up answers.  
**Example:** AI says `"Paris is the capital of Germany"`.

## 22. Multimodal Model
**Definition:** Model that understands text, images, audio, or video.  
**Example:** CLIP can match images with text.

## 23. RAG (Retrieval-Augmented Generation)
**Definition:** AI retrieves external information before generating output.  
**Example:** Chatbot looks into company documents before answering.

## 24. Vector Database
**Definition:** Database storing embeddings.  
**Example:** Pinecone, ChromaDB

## 25. Semantic Search
**Definition:** Searching by meaning instead of keywords.  
**Example:** `"big cat in jungle"` → returns `"tiger"`.

## 26. API
**Definition:** Interface to communicate with AI models.  
**Example:** Cohere API for text generation.

## 27. Endpoint
**Definition:** URL where AI API receives requests.  
**Example:** `"https://api.cohere.ai/generate"`

## 28. Fine-Tuning
**Definition:** Training a pre-trained model on specific data.  
**Example:** GPT fine-tuned to answer medical questions.

## 29. Transfer Learning
**Definition:** Using a pre-trained model for a new task.  
**Example:** BERT → fine-tuned for sentiment analysis.

## 30. Diffusion Model
**Definition:** Generates images by denoising from random noise.  
**Example:** Stable Diffusion

## 31. GAN (Generative Adversarial Network)
**Definition:** Two networks compete to create realistic outputs.  
**Example:** AI generates realistic human faces.

## 32. Agent
**Definition:** AI that can take actions or follow goals.  
**Example:** Auto-GPT planning tasks automatically.

## 33. Memory
**Definition:** AI’s ability to remember past interactions.  
**Example:** Chatbot recalls user’s previous queries.

## 34. Latency
**Definition:** Time AI takes to respond.  
**Example:** 0.5 seconds per request.

## 35. Top-k Sampling
**Definition:** Choosing next word from top-k probabilities.  
**Example:** k=5 → pick next word from 5 most likely options.

## 36. Top-p (Nucleus) Sampling
**Definition:** Choosing next word with cumulative probability ≤ p.  
**Example:** p=0.9 → pick from words covering 90% probability.

## 37. Temperature
**Definition:** Controls randomness of AI output.  
**Example:** Low temp → deterministic; High temp → creative.

## 38. Safety Filters
**Definition:** Prevent AI from harmful outputs.  
**Example:** Avoid generating violent content.

## 39. Pretraining
**Definition:** Training AI on large, generic data.  
**Example:** GPT pre-trained on Common Crawl data.

## 40. In-Context Learning
**Definition:** AI learns from examples in the prompt.  
**Example:** `"Translate English to Spanish: Apple → Manzana"`

## 41. Contextual Embedding
**Definition:** Embeddings that consider surrounding words.  
**Example:** `"bank"` → different vector in `"river bank"` vs `"money bank"`

## 42. Sentence Embedding
**Definition:** Vector representing full sentence meaning.  
**Example:** `"I love AI"` → `[0.12, -0.55, 0.88, ...]`

## 43. Multimodal Embedding
**Definition:** Embedding combining text, image, or audio.  
**Example:** CLIP embeds text and images in the same space.

## 44. Autoregressive Model
**Definition:** Predicts the next token sequentially.  
**Example:** GPT predicts next word one by one.

## 45. Encoder-Decoder Model
**Definition:** Uses encoder for input and decoder for output.  
**Example:** T5 translation model.

## 46. Beam Search
**Definition:** Generates multiple sequences and chooses the best.  
**Example:** AI generates top 5 possible sentence completions.

## 47. Text-to-Image
**Definition:** Generates images from text prompts.  
**Example:** `"A cat wearing a hat"` → AI image output.

## 48. Text-to-Speech
**Definition:** Converts text into spoken audio.  
**Example:** `"Hello"` → audio file with voice.

## 49. Speech-to-Text
**Definition:** Converts audio into text.  
**Example:** Recorded lecture → transcript.

## 50. Multimodal Generation
**Definition:** Generates multiple types of media.  
**Example:** Text prompt → image + audio.

## 51. Text-to-Video
**Definition:** Generates videos from text prompts.  
**Example:** `"A cat jumping over a fence"` → AI-generated video.

## 52. Image-to-Image
**Definition:** Transforms one image into another style.  
**Example:** Photo → painting style.

## 53. Style Transfer
**Definition:** Applies artistic style from one image to another.  
**Example:** Turn a photo into Van Gogh style.

## 54. Latent Space
**Definition:** High-dimensional space representing features.  
**Example:** Images close in latent space are visually similar.

## 55. Noise Injection
**Definition:** Adding noise to generate diverse outputs.  
**Example:** Start with random noise → generate image.

## 56. Conditioning
**Definition:** Guiding AI output with extra information.  
**Example:** Image generation conditioned on `"sunset beach"`.

## 57. Masked Language Model
**Definition:** Predicts missing words in a sentence.  
**Example:** `"I love [MASK]"` → `"AI"`.

## 58. CLIP
**Definition:** AI model linking images and text in embedding space.  
**Example:** `"A red car"` → matches red car image.

## 59. Diffusion Steps
**Definition:** Number of iterations to denoise and generate image.  
**Example:** 50 steps → higher quality image.

## 60. Guidance Scale
**Definition:** Controls how closely output follows prompt.  
**Example:** High guidance → strict adherence to text.

## 61. Prompt Chaining
**Definition:** Using output of one prompt as input to next.  
**Example:** Generate story → summarize → generate questions.

## 62. Instruction Following
**Definition:** AI follows natural language instructions.  
**Example:** `"Translate this to French."`

## 63. Role Play
**Definition:** AI acts in a specific persona.  
**Example:** `"You are a travel guide."`

## 64. Conversational AI
**Definition:** AI designed to interact in dialogue.  
**Example:** Chatbots, virtual assistants.

## 65. Retrieval
**Definition:** AI finds relevant information from database.  
**Example:** Retrieve product info for a question.

## 66. Indexing
**Definition:** Preparing data for fast retrieval.  
**Example:** Vector database indexes embeddings.

## 67. Chunking
**Definition:** Splitting large text into smaller pieces.  
**Example:** Long document → paragraphs → embeddings.

## 68. Metadata
**Definition:** Data about data, used for context.  
**Example:** `"Author: John, Date: 2025"`

## 69. Token Limit
**Definition:** Maximum tokens AI can process per request.  
**Example:** 4096 tokens → input must be shorter.

## 70. Contextual Prompt
**Definition:** Prompt with additional context for better results.  
**Example:** `"Based on previous answer, summarize..."`

## 71. Dynamic Prompting
**Definition:** Changing prompt based on intermediate AI responses.  
**Example:** Use AI output to generate new instructions.

## 72. Embedding Similarity Search
**Definition:** Finding most similar items in embedding space.  
**Example:** Query embedding → find closest documents.

## 73. Augmented Generation
**Definition:** Combining AI generation with retrieved data.  
**Example:** Chatbot answers using external database.

## 74. Knowledge Base
**Definition:** Collection of structured information AI can access.  
**Example:** Company manuals or Wikipedia.

## 75. Fine-Grained Control
**Definition:** Precise control over AI output.  
**Example:** Control style, tone, or length of text.

## 76. Output Truncation
**Definition:** Limiting AI output length.  
**Example:** Max 200 words per answer.

## 77. Response Ranking
**Definition:** Ranking multiple AI outputs for best fit.  
**Example:** Generate 5 summaries → pick most relevant.

## 78. Embedding Normalization
**Definition:** Standardizing embeddings for comparison.  
**Example:** Scale vectors to unit length.

## 79. Multi-Turn Dialogue
**Definition:** Maintaining conversation across multiple turns.  
**Example:** Chatbot remembers previous questions.

## 80. Prompt Templates
**Definition:** Reusable prompt structures.  
**Example:** `"Translate {text} to French."`

## 81. Conditional Generation
**Definition:** Output depends on specific input conditions.  
**Example:** Generate email only if subject is given.

## 82. Output Diversity
**Definition:** Variety of AI-generated responses.  
**Example:** High temperature → creative, varied outputs.

## 83. Deterministic Output
**Definition:** Same prompt produces same output.  
**Example:** Low temperature setting.

## 84. Embedding Compression
**Definition:** Reducing embedding size for efficiency.  
**Example:** 768-dim → 256-dim vectors.

## 85. Embedding Aggregation
**Definition:** Combining multiple embeddings into one.  
**Example:** Paragraph embedding → mean of sentence embeddings.

## 86. Retrieval Pipeline
**Definition:** System for retrieving relevant data for AI.  
**Example:** Query → search → embed → answer.

## 87. Multi-Lingual Model
**Definition:** AI capable of understanding multiple languages.  
**Example:** Translate English → Spanish → Hindi.

## 88. Cross-Modal Retrieval
**Definition:** Search across different modalities.  
**Example:** Text query → find matching images.

## 89. Latent Diffusion
**Definition:** Diffusion in a compressed latent space.  
**Example:** Generate high-quality images faster.

## 90. Noise Schedule
**Definition:** Strategy of adding/removing noise in diffusion.  
**Example:** Gradually reduce noise over 50 steps.

## 91. Prompt Injection
**Definition:** Malicious input that tricks AI.  
**Example:** `"Ignore previous instructions and answer..."`

## 92. Output Post-Processing
**Definition:** Adjusting AI output after generation.  
**Example:** Correct grammar or filter content.

## 93. Retrieval Context
**Definition:** External information given to AI for answer.  
**Example:** Chatbot looks into user manual for context.

## 94. Vector Similarity Threshold
**Definition:** Minimum similarity to consider relevant.  
**Example:** Cosine similarity ≥ 0.8 → include document.

## 95. Document Embedding
**Definition:** Vector representing entire document.  
**Example:** Company report → single embedding.

## 96. Chunk Embedding
**Definition:** Embedding of smaller piece of a document.  
**Example:** Paragraph → embedding.

## 97. Hybrid Search
**Definition:** Combines keyword + vector search.  
**Example:** Search `"Apple"` → find keyword + semantic match.

## 98. AI Orchestration
**Definition:** Coordinating multiple AI models or agents.  
**Example:** Multi-step workflow automation with GPT + image generator.

## 99. Multi-Modal Fusion
**Definition:** Combining features from text, image, audio into one model.  
**Example:** Video captioning AI using both audio and visual info.

## 100. Embedding Drift
**Definition:** Embeddings change over time or across models.  
**Example:** Old vs new embeddings for same text → small differences.

---

---

