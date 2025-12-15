# LLM Evaluation Python Code Explained

This Markdown file explains line by line the `llm_evaluation.py` code for evaluating LLM outputs using BLEU, ROUGE, and METEOR metrics.

---

```python
llm_evaluation.py
from datasets import load_metric
from typing import List
```

**Explanation:**
- `from datasets import load_metric`: Imports the `load_metric` function from Hugging Face datasets library to use pre-defined NLP metrics.
- `from typing import List`: Imports `List` type hint for specifying that function parameters are lists of strings.

---

```python
def llm_evaluation(references: List[str], predictions: List[str]):
    """
    Evaluate LLM outputs using common NLP metrics:
    - BLEU
    - ROUGE (simplified explanation)
    - METEOR
    """
```

**Explanation:**
- Defines the function `llm_evaluation` that takes two parameters:
  - `references`: List of reference texts.
  - `predictions`: List of generated texts from LLM.
- The docstring explains that this function computes BLEU, ROUGE, and METEOR metrics.

---

```python
    # Load metrics
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")
```

**Explanation:**
- Loads the evaluation metrics:
  - `bleu`: Measures sequence overlap between prediction and reference.
  - `rouge`: Measures coverage of reference content in generated text.
  - `meteor`: Measures semantic similarity, accounting for synonyms and paraphrasing.

---

```python
    # Prepare data for BLEU (tokenized)
    refs_tokens = [[ref.split()] for ref in references]
    preds_tokens = [pred.split() for pred in predictions]
```

**Explanation:**
- BLEU requires **tokenized text** (words separated into lists).
- `refs_tokens`: A list of tokenized reference texts, each wrapped in an extra list for BLEU format.
- `preds_tokens`: A list of tokenized predicted texts.

---

```python
    # Compute metrics
    bleu_score = bleu.compute(predictions=preds_tokens, references=refs_tokens)["bleu"]
    rouge_score = rouge.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
```

**Explanation:**
- Computes each metric:
  - `bleu_score`: BLEU value between 0 and 1, higher means closer match to reference.
  - `rouge_score`: ROUGE metrics (like ROUGE-L) for content coverage.
  - `meteor_score`: METEOR value measuring semantic similarity.

---

```python
    # Combine results
    metrics = {
        "BLEU": bleu_score,
        "ROUGE": rouge_score["rougeL"].mid.fmeasure,
        "METEOR": meteor_score
    }
    return metrics
```

**Explanation:**
- Stores all computed scores in a dictionary.
- For ROUGE, `rougeL.mid.fmeasure` is used to get the main F1-score.
- Returns the dictionary of metrics.

---

```python
# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    references = [
        "The cat sat on the mat",
        "Hello world"
    ]
    predictions = [
        "The cat is sitting on the mat",
        "Hello there world"
    ]
    results = llm_evaluation(references, predictions)
    print("LLM Evaluation Metrics:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
```

**Explanation:**
- Provides a simple example to test the function.
- Defines sample `references` and `predictions`.
- Calls `llm_evaluation` function and prints each metric with 4 decimal places.
- This section allows users to **quickly see metric results** for sample LLM outputs.

---

**Summary:**
- The code loads standard NLP metrics, tokenizes input for BLEU, computes BLEU, ROUGE, and METEOR, and returns a dictionary with the results.
- Scores closer to 1 indicate higher similarity or better content coverage.
- This setup provides a simple, ready-to-use evaluation for LLM-generated text.

