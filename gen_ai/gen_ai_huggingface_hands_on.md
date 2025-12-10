# Generative AI Hands-on with Hugging Face Transformers

## 1. Setup

Install necessary packages:

```bash
pip install transformers torch
```

> **Note:** `transformers` is required for pre-trained models and pipelines. `torch` is the deep learning framework to run the models. Some models can also use TensorFlow, but PyTorch (`torch`) is more common.

---

## 2. Text Generation (GPT-style)

```python
from transformers import pipeline

# Load the text-generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Generate text
prompt = "Once upon a time in a futuristic city"
output = generator(prompt, max_length=50, num_return_sequences=1)

print(output[0]['generated_text'])
```

**Explanation:**
- `pipeline('text-generation')` loads a pre-trained GPT-2 model.
- `max_length` defines how long the generated text will be.
- `num_return_sequences` is the number of variations generated.

---

## 3. Text Summarization

```python
from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

# Example text
text = """
Artificial Intelligence is transforming industries worldwide. It has applications in healthcare, finance, education,
and entertainment. The ability to analyze large datasets quickly and make predictions is changing the way businesses operate.
"""

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])
```

**Explanation:**
- `facebook/bart-large-cnn` is optimized for summarization.
- `max_length` and `min_length` control summary size.

---

## 4. Question Answering

```python
from transformers import pipeline

# Load the question-answering pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

context = """
Hugging Face is a company that provides open-source libraries for Natural Language Processing.
They also maintain the Transformers library, which contains state-of-the-art pre-trained models.
"""

question = "What does Hugging Face provide?"

answer = qa_pipeline(question=question, context=context)
print(answer['answer'])
```

**Explanation:**
- `distilbert-base-cased-distilled-squad` is a smaller, fast model trained for question answering.
- The model uses the context to extract the answer.

---

## 5. Text-to-Text Generation (T5)

```python
from transformers import pipeline

# Load text-to-text generation
t5_generator = pipeline('text2text-generation', model='t5-small')

# Example: Translation
text = "translate English to French: I love learning AI."
translated = t5_generator(text)
print(translated[0]['generated_text'])
```

**Explanation:**
- T5 model treats every NLP problem as text-to-text.
- Can be used for summarization, translation, and more.

---

## Notes
- You **must install `transformers` and `torch`** to use these pipelines.
- You can swap models (`gpt2`, `t5-small`, `facebook/bart-large-cnn`) with any Hugging Face model.
- Use GPU for faster inference if available.
- Hugging Face `pipeline` is beginner-friendly for rapid experimentation.