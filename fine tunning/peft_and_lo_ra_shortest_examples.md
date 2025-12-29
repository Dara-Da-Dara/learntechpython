# PEFT and LoRA – Shortest Possible Examples

This document provides **separate, minimal examples** of **PEFT** and **LoRA**, suitable for **quick reference, exams, assignments, and slides**.

---

## 1️⃣ PEFT (Parameter-Efficient Fine-Tuning) – Minimal Example

**Objective:** Demonstrate PEFT using *Prompt Tuning* (without LoRA).

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, PromptTuningConfig

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# PEFT configuration (Prompt Tuning)
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10
)

# Apply PEFT
model = get_peft_model(model, peft_config)

# Check trainable parameters
model.print_trainable_parameters()
```

### Explanation (Short)
- PEFT is a **framework** for efficient fine-tuning
- Only **virtual prompt tokens** are trained
- Base model weights remain frozen

---

## 2️⃣ LoRA (Low-Rank Adaptation) – Minimal Example

**Objective:** Apply LoRA adapters for efficient fine-tuning.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type="CAUSAL_LM"
)

# Apply LoRA using PEFT
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
```

### Explanation (Short)
- LoRA is a **PEFT technique**
- Injects **low-rank trainable matrices**
- Reduces trainable parameters drastically

---

## 🔑 Key Difference (Exam-Friendly)

> **PEFT** is a general framework for parameter-efficient fine-tuning, while **LoRA** is a specific PEFT method that fine-tunes low-rank adapter matrices.

---

## 📌 Ultra-Short Comparison Table

| Aspect | PEFT | LoRA |
|------|------|------|
| Category | Framework | Technique |
| Purpose | Efficient fine-tuning | Low-rank adaptation |
| Trainable Params | Very few | Very few |
| Base Model | Frozen | Frozen |
| Common Use | LLM fine-tuning | LLM fine-tuning |

---

## One-Line Summary

> PEFT enables efficient fine-tuning of large models, and LoRA is one of its most popular methods for reducing training cost and memory usage.

