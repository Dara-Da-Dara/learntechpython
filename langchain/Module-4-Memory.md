# Module 4: Memory Management
**Duration:** 4 Hours | **Level:** Intermediate

---

## 1. Why Memory Matters

### The Problem Without Memory

```
Turn 1:
User: "My name is Alice and I'm a software engineer"
Assistant: "Nice to meet you, Alice!"

Turn 2:
User: "What's my profession?"
Assistant: "I don't know your profession"
```

With memory, the LLM remembers previous interactions and can reference them.

---

## 2. Types of Memory in LangChain

### Memory Hierarchy

- **Buffer Memory** - Stores all messages
- **Summarization Memory** - Summarizes conversations
- **Token Buffer Memory** - Controls token usage
- **Vector/Semantic Memory** - Semantic search for memories
- **Entity Memory** - Tracks important entities

---

## 3. Buffer Memory (Most Common)

### ConversationBufferMemory

Stores all messages in a buffer.

```python
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory()

# Add messages
memory.save_context(
    {"input": "Hi, my name is Alice"},
    {"output": "Nice to meet you, Alice!"}
)

# Retrieve memory
print(memory.buffer)
```

**Pros:**
- Simple to implement
- No information loss
- Easy to debug

**Cons:**
- Memory grows indefinitely
- Can exceed token limits

---

## 4. Summarization Memory

### ConversationSummaryMemory

Summarizes conversation to save tokens.

```python
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm)

# Automatically summarizes messages
memory.save_context(
    {"input": "I'm learning Python"},
    {"output": "Great language!"}
)
```

---

## 5. Token-Based Memory

### ConversationTokenBufferMemory

Keeps conversations within a token limit.

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()

# Keep only 1000 tokens
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=1000
)
```

---

## 6. Vector/Semantic Memory

### VectorStoreMemory

Stores memories as vectors for semantic search.

```python
from langchain.memory import VectorStoreMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

memory = VectorStoreMemory(
    vectorstore=vectorstore,
    embedding_function=embeddings
)
```

---

## 7. Entity Memory

### ConversationEntityMemory

Tracks entities mentioned in conversations.

```python
from langchain.memory import ConversationEntityMemory
from langchain.llms import OpenAI

llm = OpenAI()
memory = ConversationEntityMemory(llm=llm)

# Automatically tracks entities
memory.save_context(
    {"input": "John is a doctor from NYC"},
    {"output": "Got it"}
)
```

---

## 8. Using Memory with Chains

### ConversationChain with Memory

```python
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI()
memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=llm,
    memory=memory
)

# Multi-turn conversation
response1 = chain.run(input="My name is Bob")
response2 = chain.run(input="What's my name?")
# Bob is remembered!
```

---

## 9. Memory with RAG

### Combining Memory and Retrieval

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI()

# Create vector store
vectorstore = Chroma(...)

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create RAG chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)
```

---

## 10. Memory Best Practices

### Do's

- Choose right type for your use case
- Monitor token usage
- Clear when needed
- Test memory functionality
- Document important entities

### Don'ts

- Don't use unbounded buffers
- Don't ignore context pollution
- Don't over-summarize
- Don't forget to validate

---

## 11. Review Questions

1. Why is memory important?
2. Name 3 types of memory in LangChain
3. When would you use SummaryMemory?
4. How does VectorStoreMemory work?
5. How do you clear memory?

---

## 12. Next Steps

**Next Module:** Module 5 - Retrieval-Augmented Generation

---

**Module Summary**
- Memory enables LLMs to reference previous interactions
- Buffer memory is simplest, Summary memory is efficient
- Token buffer controls token usage
- Vector memory uses semantic similarity
- Entity memory tracks important information

**Time spent:** 4 hours
