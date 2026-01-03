# 📘 Terminology for Understanding Memory in LLMs and AI Agents

This document lists **key terminology** required to clearly understand **memory concepts in Large Language Models (LLMs)** and **Agentic AI systems**. These terms are commonly used in **LangChain, LangGraph, AutoGPT, CrewAI, n8n-based agents**, and exam/interview contexts.

---

## 1. Memory (Core Term)
**Memory** is the ability of an LLM or AI agent to **store, retrieve, and use past information** to influence current or future responses and actions.

---

## 2. Parametric Memory
**Parametric memory** is memory stored **inside the model parameters** (weights) learned during training.

### Key Points
- Fixed after training
- Cannot be updated without retraining
- Stores generalized knowledge

### Example
- Knowledge inside GPT model weights

---

## 3. Non-Parametric Memory
**Non-parametric memory** is memory stored **outside the model**, usually in external systems.

### Key Points
- Can be updated dynamically
- Stores raw data or embeddings
- Used heavily in agents

### Example
- Vector databases (FAISS, Pinecone)

---

## 4. Context Window
The **context window** is the maximum amount of text (tokens) an LLM can consider at one time.

### Key Points
- Acts as short-term memory
- Older information is forgotten when window overflows

---

## 5. Short-Term Memory (STM)
**Short-term memory** refers to information available **only within the current conversation or context window**.

### Example
- Current chat history
- Recent user instructions

---

## 6. Long-Term Memory (LTM)
**Long-term memory** stores information **across sessions and conversations**.

### Characteristics
- Persistent
- External to the model
- Retrieved when needed

### Example
- User preferences stored in vector DB

---

## 7. Episodic Memory
**Episodic memory** stores **past interactions or events** with time or session context.

### Example
- Previous conversations with a user
- Past agent executions

---

## 8. Semantic Memory
**Semantic memory** stores **facts, concepts, and general knowledge**, independent of time.

### Example
- “Python is a programming language”
- Company policies stored as embeddings

---

## 9. Working Memory
**Working memory** is the temporary memory used by an agent **while reasoning or executing tasks**.

### Example
- Intermediate thoughts
- Tool outputs during execution

---

## 10. Retrieval-Augmented Generation (RAG)
**RAG** is a technique where an LLM **retrieves relevant information from external memory** before generating a response.

### Flow
Retrieve → Augment Prompt → Generate Response

---

## 11. Vector Database
A **vector database** stores information as **embeddings** for efficient semantic search.

### Common Tools
- FAISS
- Pinecone
- Chroma
- Weaviate

---

## 12. Embeddings
**Embeddings** are numerical vector representations of text, images, or data that capture semantic meaning.

### Purpose
- Similarity search
- Memory retrieval

---

## 13. Agent Memory
**Agent memory** is the structured storage of information that an agent uses to:
- Remember goals
- Track tasks
- Learn from actions

---

## 14. Tool Memory
**Tool memory** stores outputs of previously used tools.

### Example
- API response
- Database query result

---

## 15. Observation
An **observation** is information an agent receives from:
- Environment
- Tools
- User input

Observations may be stored in memory.

---

## 16. State
The **state** represents the agent’s current situation, including:
- Goals
- Memory
- Context
- Tool availability

---

## 17. Planner
A **planner** decides **what steps to take next**, often using memory to inform decisions.

---

## 18. Reflection
**Reflection** is when an agent evaluates past actions and stores lessons learned into memory.

---

## 19. Forgetting / Memory Pruning
**Forgetting** is the deliberate removal or compression of memory to:
- Reduce noise
- Save storage
- Improve relevance

---

## 20. Memory Indexing
**Memory indexing** is organizing stored memory for fast retrieval.

### Example
- Time-based
- Similarity-based
- Metadata-based

---

## 21. Memory Retrieval Strategy
The method used to decide **which memory to fetch**.

### Types
- Similarity search
- Recency-based
- Importance-based

---

## 22. Agentic Loop
The **agentic loop** is:
Observe → Think → Act → Store Memory → Repeat

Memory plays a central role in closing this loop.

---

## 23. Stateless vs Stateful Agents

| Agent Type | Memory Usage |
|-----------|--------------|
| Stateless Agent | No memory across calls |
| Stateful Agent | Maintains short-term and long-term memory |

---

## 24. Human Analogy (Quick Mapping)

| Human Brain | LLM / Agent |
|------------|------------|
| Short-term memory | Context window |
| Long-term memory | Vector DB |
| Experience | Episodic memory |
| Knowledge | Semantic memory |

---

## 25. One-Line Summary
> **Memory in LLMs and agents combines parametric knowledge inside model weights with non-parametric external memory systems to enable context awareness, learning, and autonomous behavior.**

---

## 26. Must-Know Keywords (Quick Revision)
- Parametric Memory  
- Non-Parametric Memory  
- Context Window  
- RAG  
- Embeddings  
- Vector Database  
- Episodic Memory  
- Semantic Memory  
- Agentic Loop  
- Reflection  

