# Module 5: Retrieval-Augmented Generation (RAG)
**Duration:** 6 Hours | **Level:** Intermediate

---

## 1. What is RAG (Retrieval-Augmented Generation)?

### Definition
**RAG** is a technique that combines **information retrieval** with **text generation** to provide LLMs with access to external knowledge without fine-tuning.

### The Problem RAG Solves

LLMs have limitations:
- Knowledge cutoff date (older than training data)
- Cannot access company-specific documents
- Cannot access real-time information
- Hallucinate when uncertain

### The RAG Solution

```
User Query
    ↓
Retrieve relevant documents from database
    ↓
Combine documents with query
    ↓
Send to LLM
    ↓
LLM generates answer based on retrieved documents
```

---

## 2. How RAG Works

### Three Main Steps

#### Step 1: Indexing
Break documents into chunks and create embeddings.

#### Step 2: Retrieval
When user asks question, find relevant documents using similarity search.

#### Step 3: Generation
Pass retrieved documents + question to LLM for answering.

### RAG Flow Diagram

```
Documents
    ↓
Chunk documents
    ↓
Create embeddings
    ↓
Store in vector database
    ↓
[When user asks question]
    ↓
Convert question to embedding
    ↓
Search vector database for similar documents
    ↓
Retrieve top-k documents
    ↓
Add to prompt with question
    ↓
Send to LLM
    ↓
Return answer
```

---

## 3. Document Loading and Processing

### Loading Documents

```python
from langchain.document_loaders import TextLoader, PDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Load document
loader = TextLoader("document.txt")
documents = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
```

### Supported Document Types

- PDF files
- Text files
- Word documents
- Web pages
- CSV files
- And more...

---

## 4. Embeddings

### What are Embeddings?

Embeddings are numerical representations of text that capture meaning.

```
"The cat sat on the mat"
    ↓
[0.21, -0.34, 0.56, ..., 0.12]  # 1536 dimensions for OpenAI

"A dog sat on the floor"
    ↓
[0.25, -0.31, 0.58, ..., 0.15]  # Similar but different
```

### Creating Embeddings

```python
from langchain.embeddings.openai import OpenAIEmbeddings

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a text
embedded_text = embeddings.embed_query("Hello world")
print(len(embedded_text))  # 1536 dimensions
```

---

## 5. Vector Databases

### What is a Vector Database?

A database that stores embeddings and enables semantic similarity search.

### Popular Vector Databases

| Database | Features | Best for |
|----------|----------|----------|
| **Pinecone** | Managed, fast, scalable | Production applications |
| **Weaviate** | Open-source, flexible | Custom deployments |
| **Chroma** | Lightweight, easy setup | Development, prototyping |
| **Milvus** | High performance | Large-scale applications |
| **FAISS** | Simple, efficient | Offline search |

### Using Chroma (Easiest to Start)

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Split documents
splitter = CharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Search
results = vectorstore.similarity_search("What is AI?", k=3)
print(results)
```

---

## 6. Building a Simple RAG System

### Complete RAG Implementation

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Load and index documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Step 2: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 3: Create QA chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Step 4: Ask questions
answer = qa_chain.run("What is the main topic?")
print(answer)
```

---

## 7. Advanced RAG Techniques

### Technique 1: Hybrid Search

Combine keyword search with semantic search.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma

# Semantic retriever
vector_retriever = vectorstore.as_retriever()

# Keyword retriever
keyword_retriever = BM25Retriever.from_documents(chunks)

# Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)
```

### Technique 2: Re-ranking Retrieved Documents

Re-rank retrieved documents for better relevance.

```python
# Get more documents then re-rank
top_k_documents = retriever.get_relevant_documents(query)[:10]

# Re-rank using different method or LLM
# Keep only top 3 most relevant
```

### Technique 3: Metadata Filtering

Filter documents by metadata before retrieval.

```python
# Add metadata to documents
document.metadata = {"source": "company_docs", "year": 2024}

# Filter during retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source": "company_docs"}
    }
)
```

---

## 8. RAG with Memory

### Combining RAG with Conversation Memory

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# Create conversational RAG chain
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Now it remembers previous questions
qa({"question": "What is AI?"})
qa({"question": "Tell me more about that"})
# Second question references first
```

---

## 9. RAG Best Practices

### Chunking Strategy

- **Too small chunks:** Loss of context
- **Too large chunks:** Exceeds token limits
- **Sweet spot:** 500-1500 tokens with 10-20% overlap

### Retrieval Quality

- Use appropriate similarity threshold
- Retrieve k=3 to 5 documents
- Consider document relevance
- Monitor retrieval accuracy

### Generation Quality

- Include source citations
- Add temperature control
- Use specific prompts
- Validate answers against source

---

## 10. Common Issues and Solutions

### Issue 1: Poor Retrieval Quality

**Problem:** Wrong documents retrieved

**Solutions:**
- Improve chunking strategy
- Use hybrid search
- Add metadata filtering
- Use re-ranking

### Issue 2: Context Length Exceeded

**Problem:** Too many documents exceed token limit

**Solutions:**
- Reduce k (number of documents)
- Use smaller chunks
- Summarize documents first
- Use token buffer memory

### Issue 3: Outdated Information

**Problem:** Documents not updated

**Solutions:**
- Refresh documents regularly
- Add update timestamps
- Use metadata filtering by date
- Implement version control

---

## 11. Practical RAG Examples

### Example 1: Customer Support Bot

```python
# Load FAQ documents
faqs = load_documents("faqs.txt")
chunks = split_documents(faqs)

# Create RAG system
vectorstore = create_vectorstore(chunks)
qa = create_rag_chain(vectorstore)

# Answer customer questions
answer = qa("How do I reset my password?")
# Retrieves from FAQ documents
```

### Example 2: Research Assistant

```python
# Load research papers
papers = load_documents("research_papers/")

# Create RAG system with metadata
vectorstore = create_vectorstore(papers)

# Answer research questions
answer = qa("What are latest findings on deep learning?")
# Retrieves from papers with citations
```

---

## 12. Review Questions

1. What does RAG stand for?
2. What are the three main steps in RAG?
3. What is an embedding?
4. Name 3 vector databases
5. How does RAG prevent hallucinations?

---

## 13. Next Steps

**Next Module:** Module 6 - Agents and Autonomous Systems

---

**Module Summary**
- RAG combines retrieval with generation
- Documents are chunked and embedded
- Vector databases enable semantic search
- Retrieved documents augment LLM prompts
- Memory can be added for conversational RAG

**Time spent:** 6 hours
