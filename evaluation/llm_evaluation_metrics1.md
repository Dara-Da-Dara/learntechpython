# LLM Evaluation Metrics

This document provides a structured overview of **Large Language Model (LLM) evaluation metrics**, covering automatic, human-centric, task-specific, and system-level evaluations.

---

## 1. Why LLM Evaluation Matters

Evaluating LLMs is essential to:
- Measure output quality and correctness
- Compare models and prompts
- Detect hallucinations, bias, and drift
- Ensure safety, reliability, and alignment

No single metric is sufficient—**multi-metric evaluation** is best practice.

---

## 2. Lexical & N-gram Based Metrics

These compare generated text to reference text using surface-level similarity.

### 2.1 BLEU (Bilingual Evaluation Understudy)
- Measures n-gram overlap
- Precision-focused
- Common in translation

**Range:** 0 – 1 (higher is better)

**Limitations:**
- Poor for open-ended generation
- Penalizes valid paraphrases

---

### 2.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- Measures recall of n-grams
- Popular for summarization

**Variants:**
- ROUGE-1 (unigrams)
- ROUGE-2 (bigrams)
- ROUGE-L (longest common subsequence)

---

### 2.3 METEOR
- Uses synonym and stem matching
- Better correlation with human judgment than BLEU

---

## 3. Semantic Similarity Metrics

These capture **meaning**, not just word overlap.

### 3.1 BERTScore
- Uses contextual embeddings (BERT)
- Measures semantic similarity

**Outputs:** Precision, Recall, F1

---

### 3.2 Sentence Embedding Cosine Similarity
- Compares vector embeddings
- Model-agnostic

**Range:** -1 to 1

---

## 4. Model-Based Evaluation (LLM-as-a-Judge)

Using an LLM to evaluate another LLM.

### 4.1 Pairwise Comparison
- Judge selects better response

### 4.2 Scoring Rubrics
- Scores across dimensions:
  - Accuracy
  - Relevance
  - Completeness
  - Safety

**Pros:** Scalable, flexible

**Cons:** Bias, judge inconsistency

---

## 5. Task-Specific Metrics

### 5.1 Question Answering (QA)
- Exact Match (EM)
- F1 Score

---

### 5.2 Classification Tasks
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

### 5.3 Code Generation
- Pass@k
- Unit Test Pass Rate
- Cyclomatic Complexity

---

## 6. Retrieval-Augmented Generation (RAG) Metrics

### 6.1 Retrieval Quality
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

---

### 6.2 Generation Quality
- Faithfulness
- Answer Relevance
- Context Utilization

**Popular frameworks:**
- RAGAS
- TruLens
- LangSmith

---

## 7. Hallucination & Faithfulness Metrics

### 7.1 Faithfulness Score
- Measures grounding in provided context

### 7.2 Attribution Metrics
- Checks whether claims are supported by sources

---

## 8. Human Evaluation Metrics

Human judgment remains the gold standard.

### Common Dimensions:
- Fluency
- Coherence
- Helpfulness
- Harmlessness
- Truthfulness

### Methods:
- Likert scales
- A/B testing
- Expert review

---

## 9. Safety, Bias & Alignment Metrics

### 9.1 Toxicity
- Perspective API scores

### 9.2 Bias & Fairness
- Demographic parity
- Stereotype detection

### 9.3 Refusal & Policy Compliance
- Over-refusal rate
- Under-refusal rate

---

## 10. System-Level Metrics

### 10.1 Performance
- Latency
- Throughput
- Cost per request

### 10.2 Reliability
- Error rate
- Drift detection

---

## 11. Metric Selection Guide

| Use Case | Recommended Metrics |
|--------|---------------------|
| Translation | BLEU, METEOR, BERTScore |
| Summarization | ROUGE, BERTScore |
| QA | EM, F1, Faithfulness |
| RAG | Recall@k, RAGAS |
| Chatbots | Human Eval, LLM Judge |
| Safety | Toxicity, Bias Scores |

---

## 12. Key Takeaways

- No single metric is sufficient
- Combine automatic + human evaluation
- Align metrics with business goals
- Continuously monitor in production

---

**End of File**
