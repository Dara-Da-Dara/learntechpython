# RAG-Aware Evaluation Metrics for Retrieval-Augmented Generation

RAG models combine **retrieval + generation**, so evaluation metrics must account for both components. Here is a structured approach to **RAG metrics**.

---

## 1. Retrieval Metrics

Evaluate how well the retriever finds **relevant documents**.

### 1.1 Precision@k

Fraction of top-k retrieved documents that are relevant.

```python
relevant_docs = {"doc1", "doc3"}
retrieved_docs = ["doc1", "doc2", "doc3"]
k = 3
precision_at_k = len(set(retrieved_docs[:k]) & relevant_docs) / k
print(f"Precision@{k}: {precision_at_k}")
```

### 1.2 Recall@k

Fraction of relevant documents retrieved in top-k.

```python
recall_at_k = len(set(retrieved_docs[:k]) & relevant_docs) / len(relevant_docs)
print(f"Recall@{k}: {recall_at_k}")
```

### 1.3 Mean Reciprocal Rank (MRR)

Average reciprocal rank of first relevant document.

```python
ranks = [1, 3, 5]
mrr = sum([1/rank for rank in ranks]) / len(ranks)
print(f"MRR: {mrr}")
```

### 1.4 Normalized Discounted Cumulative Gain (nDCG)

Evaluates ranking quality by relevance and position.

```python
import numpy as np
relevance_scores = [3, 2, 1, 0]

def dcg(scores):
    return sum([rel/np.log2(idx+2) for idx, rel in enumerate(scores)])

def ndcg(scores):
    ideal = sorted(scores, reverse=True)
    return dcg(scores)/dcg(ideal)

print(f"nDCG: {ndcg(relevance_scores)}")
```

---

## 2. Generation Metrics

Evaluate **accuracy, fluency, and semantic quality** of generated responses.

### 2.1 BLEU Score

Measures n-gram overlap with reference answers.

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['RAG', 'combines', 'retrieval', 'and', 'generation']]
candidate = ['RAG', 'uses', 'retrieval', 'to', 'generate']
bleu_score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {bleu_score}")
```

### 2.2 ROUGE Score

Measures recall-oriented n-gram overlap and longest common subsequence.

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
reference = "RAG combines retrieval and generation."
candidate = "RAG uses retrieval to generate responses."
scores = scorer.score(reference, candidate)
print(scores)
```

### 2.3 METEOR Score

Balances precision, recall, and semantic similarity.

```python
from nltk.translate.meteor_score import single_meteor_score
reference = 'RAG combines retrieval and generation'
candidate = 'RAG uses retrieval for generation'
meteor = single_meteor_score(reference, candidate)
print(f"METEOR Score: {meteor}")
```

### 2.4 BERTScore

Contextual embedding similarity for semantic evaluation.

```python
from bert_score import score
candidates = ['RAG uses retrieval to answer queries']
references = ['RAG combines retrieval and generation']
p, r, f1 = score(candidates, references, lang='en', rescale_with_baseline=True)
print(f"BERTScore F1: {f1.mean().item()}")
```

---

## 3. RAG-Specific Metrics

Metrics that combine **retriever + generator evaluation**.

### 3.1 Retrieval-Augmented Answer Accuracy (RAAA)

Fraction of answers correct **based on retrieved context**.

```python
correct_answers = 3
total_queries = 5
raaa = correct_answers / total_queries
print(f"RAG Answer Accuracy: {raaa}")
```

### 3.2 Context Utilization Rate

Percentage of retrieved documents actually used in generation.

```python
retrieved_docs = 5
used_docs = 3
context_utilization = used_docs / retrieved_docs
print(f"Context Utilization: {context_utilization}")
```

### 3.3 Hallucination Rate

Fraction of generated facts **not present in retrieved context**.

```python
total_facts = 10
hallucinated_facts = 2
hallucination_rate = hallucinated_facts / total_facts
print(f"Hallucination Rate: {hallucination_rate}")
```

### 3.4 Combined RAG Score (Example)

Weighted metric combining retrieval + generation quality.

```python
combined_rag_score = 0.4*precision_at_k + 0.4*raaa + 0.2*(1-hallucination_rate)
print(f"Combined RAG Score: {combined_rag_score}")
```

---

## 4. Human Evaluation

* Fluency: 1-5 scale
* Relevance: 1-5 scale
* Factual Correctness: 1-5 scale
* Completeness: 1-5 scale

### Example Form

```text
Query: What is RAG?
Generated Answer: ...
Ratings:
- Fluency: 5/5
- Relevance: 4/5
- Factual Correctness: 5/5
- Completeness: 4/5
```

---

## Key Takeaways

* RAG evaluation must combine **retrieval metrics + generation metrics**
* Use automatic metrics (BLEU, ROUGE, BERTScore) and human evaluation
* Consider **RAG-specific metrics** like Context Utilization and Hallucination Rate
* Weighted combined scores help assess **end-to-end RAG performance**
