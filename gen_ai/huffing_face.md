# 🌟 Introduction to Hugging Face

## 🔹 What is Hugging Face?
Hugging Face is an open-source AI platform providing tools, models, and libraries for:
- Natural Language Processing (NLP)
- Computer Vision (CV)
- Audio Processing
- Generative AI (Text, Image, Video)

It is widely used in research and industry because of its **Transformers**, **Diffusers**, **Datasets**, and **Model Hub**.

---

# 🔹 Key Features of Hugging Face

## 1️⃣ Transformers Library
A powerful library for:
- Pretrained models (BERT, GPT, T5, RoBERTa, ViT, Whisper, CLIP)
- Tokenization
- Training & fine-tuning
- Multimodal models

Example tasks supported:
- Text Classification  
- Sentiment Analysis  
- Question Answering  
- Summarization  
- Translation  
- Image Classification  
- Audio Transcription  

---

## 2️⃣ Model Hub
A centralized repository with **100,000+ pretrained models**:
- NLP models  
- Vision models  
- Speech models  
- Diffusion models  

You can search popular models:
- `bert-base-uncased`
- `gpt2`
- `facebook/bart-large-cnn`
- `stabilityai/stable-diffusion-2-1`

---

## 3️⃣ Pipelines API (Beginner-Friendly)
Hugging Face provides a high-level API called **pipeline**.

### ✔ Example: Sentiment Analysis
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love Hugging Face!")
