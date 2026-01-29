
# LLM Evaluation Metrics: BLEU, ROUGE, and Others

## Score Ranges (Important)

### BLEU
- Range: **0 to 1**
- Interpretation:
  - 0.0 – 0.2 → Poor overlap
  - 0.2 – 0.4 → Fair
  - 0.4 – 0.6 → Good
  - 0.6 – 1.0 → Excellent

---

### ROUGE (ROUGE-1 / ROUGE-2 / ROUGE-L)
- Range: **0 to 1**
- Interpretation:
  - < 0.2 → Weak summary
  - 0.2 – 0.4 → Acceptable
  - 0.4 – 0.6 → Good
  - > 0.6 → Very strong

---

### METEOR
- Range: **0 to 1**
- Interpretation:
  - < 0.3 → Poor
  - 0.3 – 0.5 → Average
  - 0.5 – 0.7 → Good
  - > 0.7 → Excellent

---

### BERTScore
- Range: **0 to 1**
- Interpretation:
  - < 0.85 → Low semantic similarity
  - 0.85 – 0.90 → Acceptable
  - 0.90 – 0.95 → Strong
  - > 0.95 → Excellent

---

### Perplexity
- Range: **0 to ∞ (lower is better)**
- Interpretation:
  - < 20 → Very good
  - 20 – 50 → Good
  - 50 – 100 → Average
  - > 100 → Poor

---

## BLEU Code Example
```python
from nltk.translate.bleu_score import sentence_bleu

reference = [["the", "cat", "is", "on", "the", "mat"]]
candidate = ["the", "cat", "sat", "on", "the", "mat"]

score = sentence_bleu(reference, candidate)
print(score)
```

---

## ROUGE Code Example
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
scores = scorer.score("The cat is on the mat", "The cat sat on the mat")
print(scores)
```

---

## METEOR Code Example
```python
from nltk.translate.meteor_score import meteor_score

reference = ["The cat is on the mat"]
prediction = "The cat sat on the mat"

print(meteor_score(reference, prediction))
```

---

## BERTScore Code Example
```python
from bert_score import score

P, R, F1 = score(
    ["The cat sat on the mat"],
    ["The cat is on the mat"],
    lang="en"
)

print(F1.mean().item())
```

---

## Summary Table

| Metric | Range | Best Value |
|------|-------|------------|
| BLEU | 0–1 | 1 |
| ROUGE | 0–1 | 1 |
| METEOR | 0–1 | 1 |
| BERTScore | 0–1 | 1 |
| Perplexity | 0–∞ | Lower |

---

## Best Practice
Always combine **automatic metrics + human evaluation** when evaluating LLMs.
