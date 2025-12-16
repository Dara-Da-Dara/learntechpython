---
# Cohere Multilingual Model API Guide

## Overview
Cohere provides multilingual capabilities in both **text generation** and **embeddings**. You can use their API to build chatbots, semantic search systems, translation helpers, and more.

---

## 1. Generation (Text-Generation / Conversational)
- **Models:** `command-a`, `command-r+`
- **Languages Supported:** English, French, Spanish, Italian, German, Japanese, Korean, Chinese, Arabic, Hindi, Russian, Polish, Turkish, and more.
- **Use Cases:** Chatbots, translation, summarization, Q&A.
### !pip install cohere
### Python Example
```python
import cohere

api_key = "YOUR_COHERE_API_KEY"
co = cohere.Client(api_key)

response = co.generate(
    model="command-a-03-2025",
    prompt="Translate the following sentence to Hindi:\nHello, how are you?",
    max_tokens=60
)

print(response.text)
```

### Node.js Example
```javascript
import Cohere from "cohere-ai";
Cohere.init("YOUR_COHERE_API_KEY");

const response = await Cohere.generate({
  model: "command-a-03-2025",
  prompt: "Summarize the following text in Spanish:\nThis year we expanded into new regions.",
  max_tokens: 60,
});

console.log(response.body.text);
```

---

## 2. Embeddings (Semantic / Search / Vectors)
- **Models:** `cohere/embed-multilingual-v3.0`, `cohere/embed-multilingual-light-v3.0`
- **Use Cases:** Semantic search, clustering, classification, cross-language similarity.

### Python Example
```python
import cohere

api_key = "YOUR_COHERE_API_KEY"
client = cohere.Client(api_key)

texts = ["Hello world!", "नमस्ते दुनिया!"]
response = client.embed(
    model="cohere/embed-multilingual-v3.0",
    texts=texts
)

embeddings = response.embeddings
print(embeddings)
```

---

## 3. Best Practices
- Test model outputs in target languages for quality.
- Use multilingual embeddings in vector databases for semantic search.
- Be aware of API rate limits with trial keys.

---

## 4. Requirements
Create a `requirements.txt` file with the following:
```
cohere>=5.0.0
python-dotenv>=1.0.0
```

---

## References
- [Cohere Docs](https://docs.cohere.com/)
- [Cohere Multilingual Embeddings](https://promptlayer.com/models/cohere-embed-multilingual-v30)
- [Cohere API FAQ](https://docs.cohere.com/v1/docs/cohere-faqs)
