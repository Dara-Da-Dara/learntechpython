# Metrics of LangChain Memory

This document focuses **only on how LangChain memory is evaluated**. LangChain does not provide a single numeric memory score; instead, memory quality is measured using **behavioral, retrieval, and system-level metrics** depending on the memory type.

---

## 1. LangChain Memory Types (Context)

| Memory Class                   | Purpose                          |
| ------------------------------ | -------------------------------- |
| ConversationBufferMemory       | Stores full conversation history |
| ConversationBufferWindowMemory | Stores last *k* messages         |
| ConversationSummaryMemory      | Stores summarized conversation   |
| VectorStoreRetrieverMemory     | Long-term semantic memory (RAG)  |
| CombinedMemory                 | Combination of multiple memories |

---

## 2. Core Metrics for LangChain Memory

### 2.1 Context Retention Accuracy

**Definition:** Ability of the model to correctly recall facts from memory.

**Measurement Method:**

* Ask questions about earlier conversation turns

| Accuracy | Interpretation  |
| -------- | --------------- |
| < 60%    | Poor memory     |
| 60–80%   | Moderate memory |
| > 80%    | Strong memory   |

---

### 2.2 Memory Compression Ratio

**Used with:** `ConversationSummaryMemory`

```text
Compression Ratio = Original Tokens / Summary Tokens
```

| Ratio | Meaning                               |
| ----- | ------------------------------------- |
| < 2×  | Weak compression                      |
| 2×–5× | Balanced                              |
| > 5×  | Aggressive (risk of information loss) |

---

### 2.3 Information Loss Rate

**Definition:** Percentage of important facts lost during summarization.

```text
Information Loss (%) = Lost Facts / Total Facts × 100
```

| Loss Rate | Quality    |
| --------- | ---------- |
| < 10%     | Excellent  |
| 10–25%    | Acceptable |
| > 25%     | Poor       |

---

### 2.4 Retrieval Quality Metrics (Vector Memory)

**Used with:** `VectorStoreRetrieverMemory`

| Metric      | Description                     |
| ----------- | ------------------------------- |
| Precision@K | Relevance of retrieved memories |
| Recall@K    | Coverage of stored information  |
| MRR         | Mean Reciprocal Rank            |

| Score   | Memory Quality |
| ------- | -------------- |
| < 0.6   | Weak           |
| 0.6–0.8 | Moderate       |
| > 0.85  | Strong         |

---

### 2.5 Memory Retrieval Latency

**Definition:** Time overhead added due to memory lookup.

| Latency   | Interpretation            |
| --------- | ------------------------- |
| < 50 ms   | Excellent                 |
| 50–150 ms | Acceptable                |
| > 300 ms  | Poor (needs optimization) |

---

### 2.6 Memory Growth Rate

**Used with:** Buffer-based memories

```text
Growth Rate = Tokens added per conversation turn
```

| Growth Rate | Risk                  |
| ----------- | --------------------- |
| Low         | Safe                  |
| Medium      | Needs windowing       |
| High        | Context overflow risk |

---

### 2.7 Memory Consistency Score

**Definition:** Stability of recalled information across multiple turns.

| Consistency | Interpretation      |
| ----------- | ------------------- |
| High        | Reliable memory     |
| Medium      | Occasional drift    |
| Low         | Hallucination-prone |

---

## 3. Agent-Level Memory Metrics (LangChain + LangGraph)

| Metric                  | Description                      |
| ----------------------- | -------------------------------- |
| Episodic Recall Depth   | Number of past episodes recalled |
| Temporal Accuracy       | Correct ordering of events       |
| Tool Recall Accuracy    | Correct reuse of tools           |
| Reflection Success Rate | Learning from past mistakes      |

---

## 4. Mapping Metrics to LangChain Memory Classes

| Memory Type       | Key Metrics                         |
| ----------------- | ----------------------------------- |
| BufferMemory      | Growth rate, recall accuracy        |
| WindowMemory      | Recall accuracy, consistency        |
| SummaryMemory     | Compression ratio, information loss |
| VectorStoreMemory | Precision@K, Recall@K, latency      |
| CombinedMemory    | Overall task success rate           |

---

## 5. Practical Evaluation Checklist

* Can the system recall earlier user preferences?
* Is summarization preserving critical facts?
* Is retrieval fast and relevant?
* Does memory improve task success over time?

---

## 6. Exam / Interview-Ready Statement

> **LangChain memory is evaluated using recall accuracy, compression quality, retrieval precision, latency, and consistency rather than a single numeric metric.**

---

**End of Document**
