# Using LLaMA Without an API

LLaMA models are **open-source** (or have released weights), which allows you to run them **locally without relying on an API**. This section explains how, why, and provides an example.

---

## 1. Overview

### Open-Weight Access
- **LLaMA 1 and LLaMA 2** provide downloadable model weights.  
- You can **host the model locally** or on your own cloud servers.  
- Full customization options:
  - Fine-tuning on your own datasets  
  - Modifying architecture (research purposes)  
  - Running inference locally

### Advantages of Local Deployment
1. **Offline usage** → No internet/API required.  
2. **Data privacy** → Sensitive data stays on your system.  
3. **Custom fine-tuning** → Adapt model to domain-specific tasks.  
4. **Cost control** → Only pay for your compute resources.

---

## 2. Software Requirements

- **Python 3.8+**  
- **PyTorch** or **Hugging Face Transformers**  
- GPU recommended for large models (≥7B parameters)  

---

## 3. Example: Local Inference Using Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model (LLaMA 2 7B)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Encode input and generate output
input_text = "Explain AI in simple terms"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
