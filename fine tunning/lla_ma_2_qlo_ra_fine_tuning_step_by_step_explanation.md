# LLaMA 2 Fine-Tuning with QLoRA – Step-by-Step Explanation

This document explains the provided **Jupyter Notebook** (`LLama2_FineTuning (1) QLORA.ipynb`) step by step. The notebook demonstrates how to fine-tune **LLaMA 2 (7B Chat)** using **QLoRA (Quantized Low-Rank Adaptation)** for memory‑efficient training.

---

## Step 1: Install Required Libraries

```python
%%capture
# Installs libraries required for model fine-tuning
```

### Explanation
- `%%capture` hides verbose installation logs.
- Installs key libraries such as:
  - **transformers** – Hugging Face models & tokenizers
  - **datasets** – dataset loading and preprocessing
  - **peft** – parameter‑efficient fine‑tuning (LoRA)
  - **bitsandbytes** – 4‑bit & 8‑bit quantization
  - **trl** – training utilities for LLM fine‑tuning (SFTTrainer)

👉 This setup enables **low‑memory fine‑tuning** of large models.

---

## Step 2: Import Required Modules

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
```

### Explanation
- **AutoModelForCausalLM** – Loads LLaMA for text generation
- **AutoTokenizer** – Tokenizes text inputs
- **BitsAndBytesConfig** – Enables 4‑bit quantization (QLoRA)
- **TrainingArguments** – Controls training behavior
- **LoraConfig** – Defines LoRA parameters
- **SFTTrainer** – Supervised fine‑tuning trainer for LLMs

---

## Step 3: Load the Dataset

```python
dataset = load_dataset("mlabonne/guanaco-llama2-1k")
```

### Explanation
- Loads a **1,000‑sample instruction‑tuning dataset**
- Designed specifically for **LLaMA 2 chat models**
- Dataset includes instruction–response pairs

---

## Step 4: Explore the Dataset

```python
print("Dataset Structure:", dataset)
print("First few examples from the dataset:", dataset[:5])
```

### Explanation
- Displays dataset splits and schema
- Helps verify the **text field** used for training

---

## Step 5: Convert Dataset to Pandas & Perform EDA

```python
import pandas as pd
```

### Explanation
- Converts dataset to Pandas DataFrame
- Performs **basic exploratory data analysis**:
  - Text length distribution
  - Data sampling
- Ensures prompts are within model context limits

---

## Step 6: Sample Dataset Rows

```python
df.sample(2)
```

### Explanation
- Randomly inspects training samples
- Helps validate **instruction–response formatting**

---

## Step 7: Define Base Model

```python
base_model = "NousResearch/Llama-2-7b-chat-hf"
```

### Explanation
- Uses **LLaMA 2 – 7B Chat** model
- Chat‑optimized version of LLaMA
- Requires Hugging Face access permission

---

## Step 8: Define Output Model Name

```python
new_model = "abhi/Llama-2-7b-chat-finetuned"
```

### Explanation
- Name for saving the fine‑tuned model
- Can later be uploaded to Hugging Face Hub

---

## Step 9: Configure QLoRA (4‑bit Quantization)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
```

### Explanation
- Enables **4‑bit quantization** using QLoRA
- `nf4` → Normal Float 4 (best for LLMs)
- Reduces GPU memory usage drastically
- Allows fine‑tuning on **single consumer GPUs**

---

## Step 10: Load Quantized Model

```python
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1
```

### Explanation
- Loads LLaMA 2 in **4‑bit precision**
- `use_cache=False` avoids training conflicts
- `pretraining_tp=1` avoids tensor parallel issues

---

## Step 11: Load Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

### Explanation
- Uses the same tokenizer as base model
- Padding set to EOS token for causal LM
- Right‑padding improves training stability

---

## Step 12: Configure LoRA (PEFT)

```python
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Explanation
- **LoRA** fine‑tunes only small adapter matrices
- `r` → rank of LoRA matrices
- Reduces trainable parameters by **>99%**
- Ideal for large LLMs

---

## Step 13: Training Configuration

```python
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    logging_steps=25,
    save_steps=0,
    report_to="tensorboard",
)
```

### Explanation
- Defines training behavior
- Gradient accumulation simulates larger batches
- Uses **paged AdamW optimizer** for memory efficiency
- TensorBoard logging enabled

---

## Step 14: Initialize SFTTrainer

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_params,
)
```

### Explanation
- Combines model, dataset, tokenizer, and LoRA
- Handles formatting and tokenization internally
- Optimized for **instruction fine‑tuning**

---

## Step 15: Train the Model

```python
trainer.train()
```

### Explanation
- Fine‑tunes LoRA adapters only
- Base LLaMA weights remain frozen
- Training is fast and memory‑efficient

---

## Step 16: Save the Fine‑Tuned Model

```python
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
```

### Explanation
- Saves LoRA‑adapted model
- Tokenizer saved for inference compatibility

---

## Step 17: Inference with Fine‑Tuned Model

```python
prompt = "Who is Ratan Tata?"
result = trainer.model.generate(
    **tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt")
)
print(result[0]['generated_text'])
```

### Explanation
- Tests the fine‑tuned model
- Uses **LLaMA instruction format**
- Generates a response based on learned behavior

---

## Summary

- QLoRA enables **LLM fine‑tuning on limited hardware**
- LoRA reduces trainable parameters drastically
- Ideal for domain‑specific chatbots and assistants
- Production‑ready approach for LLaMA 2 fine‑tuning

---

✅ This markdown file can be directly used for **teaching, assignments, or documentation**.

