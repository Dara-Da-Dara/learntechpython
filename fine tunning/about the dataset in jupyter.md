# Dataset Used for LLaMA2 Fine-Tuning with QLoRA

## Dataset Name
**mlabonne/guanaco-llama2-1k**

This dataset is used in the notebook **LLama2_FineTuning (1) QLORA.ipynb** and is loaded using the Hugging Face `datasets` library.

```python
dataset = load_dataset("mlabonne/guanaco-llama2-1k")


## About the Dataset

**Guanaco-LLaMA2-1K** is a compact **instruction fine-tuning dataset** designed for experimenting with **LLaMA / LLaMA2 models**. It is a curated subset derived from larger Guanaco instruction datasets and contains approximately **1,000 high-quality instruction–response samples**. This makes it ideal for **low-resource fine-tuning approaches** such as **LoRA** and **QLoRA**.

---

## Dataset Type

- Instruction Fine-Tuning (**Supervised Fine-Tuning – SFT**)
- Text-only dataset
- Optimized for instruction-following and alignment tasks

---

## Dataset Structure

Each record follows an **instruction–response format**, typically merged into a single text field for ease of training.

### Typical Schema

| Field | Description |
|------|-------------|
| `text` | Combined instruction and response |

---

## Example Data Format

```text
### Instruction:
Explain fine-tuning in machine learning.

### Response:
Fine-tuning is the process of adapting a pre-trained model to a specific task by training it on domain-specific data.
