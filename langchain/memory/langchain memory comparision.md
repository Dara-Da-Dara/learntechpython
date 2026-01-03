# 🧠 Comparative Table of LangChain Memory Types

## Overview

| **Memory Type**                 | **Description**                                                 | **Best Use Case**                              | **Performance**      | **Cost** | **Storage**                      | **Accessibility** |
| ------------------------------- | --------------------------------------------------------------- | ---------------------------------------------- | -------------------- | -------- | -------------------------------- | ----------------- |
| **Buffer Memory**               | Stores entire past exchanges directly.                          | Short‑term chat context.                       | Fast, simple.        | Low      | In‑memory                        | Local only        |
| **Buffer Window Memory**        | Stores only last *k* exchanges (window).                        | Keep recent context relevant, avoid overload.  | Fast, simple.        | Low      | In‑memory                        | Local             |
| **Conversation Summary Memory** | LLM‑generated summarized conversation history.                  | Longer chats, multi‑topic sessions.            | Moderate (LLM calls) | Moderate | In‑memory or external            | Local/Remote      |
| **Entity Memory**               | Stores structured facts about entities (names/attributes).      | Personal assistants, QA with entity facts.     | Moderate             | Moderate | In‑memory or external            | Local/Remote      |
| **DynamoDB‑Backed Chat Memory** | Chat history written to DynamoDB with filters.                  | Scalable, long‑term persistence.               | Slower, reliable     | High     | DynamoDB                         | Remote            |
| **Momento‑Backed Chat Memory**  | Uses Momento DB for persistent store.                           | Scalable persistent session memory.            | Slower               | High     | Momento DB                       | Remote            |
| **Redis‑Backed Chat Memory**    | Redis‑stored chat history.                                      | High throughput, persistent storage.           | Slower, robust       | High     | Redis                            | Remote            |
| **Upstash Redis‑Backed Memory** | Redis on Upstash for persistence.                               | Cloud‑managed Redis memory.                    | Slower               | High     | Upstash Redis                    | Remote            |
| **Motörhead**                   | Memory server featuring *incremental summarization*.            | Stateless applications needing summary memory. | Moderate complexity  | Moderate | External server                  | Remote            |
| **Zep Memory**                  | Memory server with storage, summary, embedding, indexing.       | Advanced analysis + enrichment.                | Slow, powerful       | High     | Zep Server                       | Remote            |
| **VectorStore‑Backed Memory**   | Stores memories in a Vector DB with top‑K similarity retrieval. | Semantic retrieval, RAG apps.                  | Moderate             | Moderate | Vector DB (e.g., Pinecone/FAISS) | Remote            |

---

## How to Choose

* **Fast, simple chat context:** *Buffer* or *Buffer Window Memory*
* **Longer, multi‑topic chats:** *Conversation Summary Memory*
* **Entity‑specific / personalized detail:** *Entity Memory*
* **Persistent, scalable storage:** *DynamoDB/Redis/Momento/Upstash*
* **Semantic, retrieval‑based context:** *VectorStore‑Backed Memory*
* **Advanced memory summarization/enrichment:** *Motörhead or Zep*

---

## Notes

* Remote/backed memory types require external services (databases or memory servers).
* Summarization and entity memories typically rely on additional LLM calls, increasing cost.
* Vector DB memory excels when s
