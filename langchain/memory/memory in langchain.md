# 🧠 Memory in LangChain

## 1. What is Memory in LangChain?

In **LangChain**, memory allows an agent or LLM to **store, retrieve, and reference information from previous interactions**. This enables:

* **Context-aware conversations**
* **Multi-step reasoning**
* **Dynamic agent behavior**
* **Personalization of user experience**

LangChain abstracts memory to provide **short-term, long-term, and custom memory stores**.

---

## 2. Why Memory Is Important in LangChain

* Maintains conversation state across multiple turns
* Stores **user preferences and session data**
* Enhances **retrieval-augmented generation (RAG)**
* Supports **agents that can plan, reflect, and act over time**

---

## 3. Types of Memory in LangChain

### 3.1 ConversationBufferMemory (Short-Term Memory)

* Stores messages from a conversation **in order**
* Limited to **current session**
* Useful for chatbots and agents needing immediate context

**Example:**

```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("Hello!")
memory.chat_memory.add_ai_message("Hi! How can I help you?")
```

---

### 3.2 ConversationSummaryMemory

* Summarizes past conversations to **fit within context limits**
* Useful when **context window is limited**
* Maintains **long-term coherence without token overflow**

**Example:**

```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "Hello"}, {"output": "Hi!"})
```

---

### 3.3 CombinedMemory / Multi-Memory

* Combines **short-term + long-term memory**
* Can store structured data alongside conversation context
* Enables **multi-modal memory** for agents

---

### 3.4 VectorStoreMemory (Long-Term / Semantic Memory)

* Stores memory in **vector databases** like **FAISS, Pinecone, Chroma**
* Memory can be **retrieved semantically** using embeddings
* Supports **retrieval-augmented generation (RAG)**

**Example:**

```python
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory

vectorstore = FAISS.load_local("my_faiss_index")
memory = VectorStoreRetrieverMemory(vectorstore=vectorstore.as_retriever())
```

---

### 3.5 Custom Memory

* LangChain allows **developers to create custom memory stores**
* Can include **databases, files, or APIs**
* Useful for domain-specific requirements

---

## 4. Memory Workflow in LangChain

```
User Input → Memory Retrieval → LLM Reasoning → Action → Memory Update
```

**Steps:**

1. **Capture input**
2. **Retrieve relevant past context**
3. **Generate output using LLM**
4. **Store output in memory**

---

## 5. Memory Management Tips

* Use **ConversationSummaryMemory** for long interactions
* Use **VectorStoreMemory** for knowledge bases or RAG tasks
* Limit token size to avoid context overflow
* Regularly **prune or summarize memory** to maintain efficiency

---

## 6. Popular Memory Classes in LangChain

| Memory Class               | Purpose                     | Example Use Case        |
| -------------------------- | --------------------------- | ----------------------- |
| ConversationBufferMemory   | Short-term chat memory      | Chatbots                |
| ConversationSummaryMemory  | Long-term summarized memory | Multi-turn sessions     |
| VectorStoreRetrieverMemory | Semantic memory             | RAG agents              |
| CombinedMemory             | Mixed memory strategies     | Complex agent workflows |
| CustomMemory               | Custom storage logic        | Domain-specific memory  |

---

## 7. Summary

LangChain memory **enhances the capabilities of LLMs** by:

* Providing **short-term context**
* Enabling **long-term knowledge retention**
* Supporting **semantic retrieval with vector stores**
* Allowing **custom memory for domain-specific tasks**

> Memory in LangChain transforms LLMs from **stateless text generators to stateful, context-aware agents**.
