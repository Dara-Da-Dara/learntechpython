# Large Language Model (LLM) Evaluation Metrics

This document provides a detailed overview of **20+ evaluation metrics** used to assess the performance of Large Language Models (LLMs), including their **scale**, **meaning**, **theory**, and **Python code examples** where applicable.

---

## 1. Perplexity (PPL)

* **Theory:** Measures how well a probabilistic language model predicts a sample. It represents the exponentiated average negative log-likelihood of a sequence.
* **Scale:** [0, ∞), lower values indicate better prediction.
* **Code:**

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
loss = model(**inputs, labels=inputs["input_ids"]).loss
perplexity = torch.exp(loss)
print("Perplexity:", perplexity.item())
```

## 2. BLEU

* **Theory:** Evaluates n-gram overlap between generated and reference text; commonly used in translation.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['The', 'cat', 'sat', 'on', 'the', 'mat']]
candidate = ['The', 'cat', 'is', 'on', 'the', 'mat']
score = sentence_bleu(reference, candidate)
print("BLEU Score:", score)
```

## 3. ROUGE

* **Theory:** Measures recall-focused n-gram overlap for summarization tasks.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score("The cat is on the mat", "The cat sat on the mat")
print(scores)
```

## 4. METEOR

* **Theory:** Considers synonymy, stemming, and word order; good for semantic evaluation.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from nltk.translate.meteor_score import meteor_score
reference = "The cat sat on the mat"
candidate = "The cat is on the mat"
score = meteor_score([reference], candidate)
print("METEOR Score:", score)
```

## 5. Accuracy

* **Theory:** Fraction of correct predictions out of total predictions.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from sklearn.metrics import accuracy_score
y_true = [1,0,1,1]
y_pred = [1,0,0,1]
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
```

## 6. F1 Score

* **Theory:** Harmonic mean of precision and recall; balances both.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)
```

## 7. Precision

* **Theory:** True positives over predicted positives.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
print("Precision:", precision)
```

## 8. Recall

* **Theory:** True positives over actual positives.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
print("Recall:", recall)
```

## 9. Exact Match (EM)

* **Theory:** Measures if predicted output exactly matches reference.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
y_pred_text = "The cat sat on the mat"
y_true_text = "The cat sat on the mat"
em = int(y_pred_text == y_true_text)
print("Exact Match:", em)
```

## 10. Mean Reciprocal Rank (MRR)

* **Theory:** Evaluates ranking quality by averaging reciprocal of the rank of first correct answer.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
def mrr_score(ranks):
    return sum(1/rank for rank in ranks)/len(ranks)
ranks = [1,2,3]
print("MRR:", mrr_score(ranks))
```

## 11. nDCG

* **Theory:** Discounted cumulative gain, evaluates top-k ranking relevance.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from sklearn.metrics import ndcg_score
import numpy as np
y_true = np.asarray([[3,2,3,0,1,2]])
y_score = np.asarray([[2,1,2,0,0,1]])
print("nDCG:", ndcg_score(y_true, y_score))
```

## 12. Mean Average Precision (MAP)

* **Theory:** Average precision over multiple queries.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from sklearn.metrics import average_precision_score
y_true = [0,1,1,0]
y_score = [0.1,0.8,0.9,0.2]
print("MAP:", average_precision_score(y_true, y_score))
```

## 13. BERTScore

* **Theory:** Uses contextual embeddings to measure semantic similarity.
* **Scale:** 0-1, higher is better.
* **Code:**

```python
from bert_score import score
cands = ["The cat is on the mat"]
refs = ["The cat sat on the mat"]
P, R, F1 = score(cands, refs, lang="en")
print("BERTScore F1:", F1)
```

## 14. Per-Response Likelihood

* **Theory:** Likelihood of generated tokens; higher indicates better model confidence.
* **Scale:** [0,1]

## 15. Token-Level Accuracy

* **Theory:** Fraction of correctly predicted tokens.
* **Scale:** 0-1

## 16. Diversity Metrics (Distinct-1, Distinct-2)

* **Theory:** Measures lexical diversity in generated text using unique n-grams.
* **Scale:** 0-1
* **Code:**

```python
def distinct_n_grams(sentences, n):
    ngrams = set()
    total = 0
    for sentence in sentences:
        tokens = sentence.split()
        total += len(tokens)-n+1
        for i in range(len(tokens)-n+1):
            ngrams.add(tuple(tokens[i:i+n]))
    return len(ngrams)/total
sentences = ["The cat is on the mat", "The dog is on the log"]
print("Distinct-2:", distinct_n_grams(sentences,2))
```

## 17. Human Evaluation

* **Theory:** Subjective scoring on fluency, relevance, coherence.
* **Scale:** 1-5 or 1-10

## 18. Win Rate / Preference Rate

* **Theory:** Fraction of times LLM output preferred over a baseline.
* **Scale:** 0-1

## 19. Length / Brevity Metrics

* **Theory:** Average token length; ensures conciseness.
* **Scale:** Contextual

## 20. Hallucination Rate

* **Theory:** Fraction of content that is factually incorrect.
* **Scale:** 0-1, lower is better

## 21. Consistency / Contradiction Metrics

* **Theory:** Evaluates internal consistency in generated content.
* **Scale:** 0-1, higher is better

---

## Summary Table

| Metric                 | Scale    | Theory / Meaning                     | Goal       |
| ---------------------- | -------- | ------------------------------------ | ---------- |
| Perplexity             | [0,∞)    | Prediction uncertainty               | Minimize   |
| BLEU                   | 0-1      | N-gram overlap                       | Maximize   |
| ROUGE                  | 0-1      | Summarization n-gram overlap         | Maximize   |
| METEOR                 | 0-1      | Semantic match with synonyms         | Maximize   |
| Accuracy               | 0-1      | Correct predictions                  | Maximize   |
| F1 Score               | 0-1      | Harmonic mean of precision & recall  | Maximize   |
| Precision              | 0-1      | True positives / predicted positives | Maximize   |
| Recall                 | 0-1      | True positives / actual positives    | Maximize   |
| Exact Match            | 0-1      | Perfect match                        | Maximize   |
| MRR                    | 0-1      | Reciprocal rank                      | Maximize   |
| nDCG                   | 0-1      | Top-k ranking relevance              | Maximize   |
| MAP                    | 0-1      | Average precision                    | Maximize   |
| BERTScore              | 0-1      | Semantic similarity using embeddings | Maximize   |
| Diversity (Distinct-N) | 0-1      | Text lexical diversity               | Maximize   |
| Human Eval             | 1-5/1-10 | Fluency, relevance, coherence        | Maximize   |
| Win Rate               | 0-1      | Preference over baseline             | Maximize   |
| Length / Brevity       | -        | Average length                       | Contextual |
| Hallucination Rate     | 0-1      | Factual correctness                  | Minimize   |
| Consistency            | 0-1      | Internal consistency                 | Maximize   |

**Note:** Metrics should be selected according to the LLM task, e.g., summarization, translation, generation, or QA.
