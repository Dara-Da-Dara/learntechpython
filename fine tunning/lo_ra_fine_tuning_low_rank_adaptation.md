# LoRA Fine-Tuning (Low-Rank Adaptation)

## 1. What is LoRA?
**LoRA (Low-Rank Adaptation)** is a **Parameter-Efficient Fine-Tuning (PEFT)** technique used to adapt large pre-trained models (LLMs, Vision Transformers, Diffusion models) by training only a **small number of additional parameters**, instead of updating all model weights.

The original model weights remain **frozen**, and LoRA learns small trainable matrices that modify the behavior of the model.
#### example 1-100 numbers  as llm  , opting 9, 19, 29 , 39--99 as lora 
---

## 2. Core Idea
Instead of updating a large weight matrix **W**, LoRA decomposes the update into two low-rank matrices:

\[
W' = W + \Delta W
\]

\[
\Delta W = B \cdot A
\]

Where:
- **W** → frozen pre-trained weight matrix
- **A** ∈ ℝ^(r × d)
- **B** ∈ ℝ^(d × r)
- **r** → low rank (small value like 4, 8, 16)

This drastically reduces the number of trainable parameters.

---

## 3. Why Use LoRA?
- ✅ 90–99% fewer trainable parameters
- ✅ Faster fine-tuning
- ✅ Low GPU / VRAM usage
- ✅ Original model remains unchanged
- ✅ Easy to save, load, and swap adapters

---

## 4. Where LoRA is Applied
LoRA is commonly injected into:
- Transformer **Attention Layers**
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Sometimes **Feed Forward (MLP) layers**

Used across:
- Large Language Models (LLaMA, Mistral, GPT-style)
- Vision Transformers (ViT)
- Diffusion Models (Stable Diffusion)

---

## 5. LoRA vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA Fine-Tuning |
|------|----------------|----------------|
| Trainable Parameters | All model weights | Only LoRA layers |
| GPU Memory | Very High | Low |
| Training Speed | Slow | Fast |
| Risk of Overfitting | High | Lower |
| Model Storage | Very Large | Small adapters |

---

## 6. Minimal LoRA Fine-Tuning Example (Hugging Face + PEFT)

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Only LoRA parameters are trainable.

---

## 7. Training Workflow
1. Load a pre-trained model
2. Freeze base model weights
3. Inject LoRA adapters
4. Train only LoRA parameters
5. Save adapter weights (MBs)
6. Load adapters during inference

---

## 8. Inference Options
- **Adapter-based inference** (Base model + LoRA)
- **Merged inference** (LoRA merged into base model)

```python
model.merge_and_unload()
```

---

## 9. Important Hyperparameters
- **Rank (r)**: 4, 8, 16
- **Alpha (lora_alpha)**: usually 2 × r
- **Dropout**: 0.05 – 0.1
- **Target Modules**: attention projection layers

---

## 10. When to Use LoRA
- ✔ Domain-specific adaptation
- ✔ Instruction fine-tuning
- ✔ RAG + fine-tuning pipelines
- ✔ Limited compute environments
- ✔ Multi-task adapter deployment

---

## 11. Summary
LoRA enables efficient fine-tuning of large models by learning low-rank updates while keeping the base model frozen. It is widely used in modern LLM pipelines due to its speed, low cost, and flexibility.

---

**End of Document**

