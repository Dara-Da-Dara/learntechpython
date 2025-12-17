# Introduction to Retrieval-Augmented Generation (RAG) with Code Examples

## 1. What is Retrieval-Augmented Generation (RAG)?

RAG is a **Generative AI architecture** combining:

* **Information Retrieval Systems**
* **Large Language Models (LLMs)**

Instead of relying solely on an LLM’s static knowledge, RAG **retrieves external knowledge at query time** to generate accurate, grounded responses.

```python
# Simple RAG conceptual flow using LangChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings()
vectordb = FAISS.load_local("docs_index", embeddings)
llm = OpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
response = qa.run("What is transfer learning?")
print(response)
```

---

## 2. Why RAG is Needed

### Limitations of Pure LLMs

* Knowledge is **frozen at training time**
* High risk of **hallucinations**
* Cannot access **private data**

### How RAG Solves These Problems

* Injects **external knowledge dynamically**
* Reduces hallucinations
* Enables **enterprise or live data** usage

---

## 3. High-Level RAG Workflow

```text
User Query
    ↓
Query Embedding
    ↓
Retriever (Vector DB / Search)
    ↓
Relevant Context
    ↓
Prompt Construction
    ↓
LLM Generation
    ↓
Final Answer
```

---

## 4. Core Elements of a RAG System

### 4.1 Data Source (Knowledge Base)

```python
# Example: Load documents into vector store
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('docs_folder', glob='**/*.txt')
documents = loader.load()
```

### 4.2 Data Ingestion & Preprocessing

```python
# Clean and normalize text
cleaned_docs = [doc.page_content.lower().strip() for doc in documents]
```

### 4.3 Chunking (Text Splitting)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

### 4.4 Embedding Model

```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
chunk_vectors = [embeddings.embed_text(chunk.page_content) for chunk in chunks]
```

### 4.5 Vector Database (Vector Store)

```python
from langchain.vectorstores import FAISS
vectordb = FAISS.from_documents(chunks, embeddings)
vectordb.save_local("docs_index")
```

### 4.6 Retriever

```python
retriever = vectordb.as_retriever(search_kwargs={"k":5})
response = qa.run("Explain overfitting")
print(response)
```

### 4.7 Reranker

```python
# Using Cohere reranker (conceptual)
from langchain.retrievers.document_compressors import CohereRerank
reranker = CohereRerank()
compressed_retriever = retriever.with_compression(reranker)
response = qa.run("Explain RAG in AI")
```

### 4.8 Prompt Construction (Context Injection)

```python
context = " ".join([doc.page_content for doc in retriever.get_relevant_documents("What is RAG?")])
prompt = f"Use the following context to answer the question.\nContext:\n{context}\nQuestion:\nWhat is RAG?"
answer = llm(prompt)
print(answer)
```

### 4.9 Large Language Model (LLM)

```python
# Already shown in QA examples
llm = OpenAI(temperature=0)
```

### 4.10 Post-Processing & Validation

```python
# Simple check for empty response
if not answer:
    answer = "Sorry, no relevant answer found."
print(answer)
```

---

## 5. RAG vs Fine-Tuning (Quick Comparison)

| Aspect                | RAG    | Fine-Tuning |
| --------------------- | ------ | ----------- |
| Knowledge updates     | Easy   | Difficult   |
| Cost                  | Lower  | Higher      |
| Hallucination control | Better | Limited     |
| Private data          | Yes    | Yes         |
| Real-time data        | Yes    | No          |

---

## 6. Where RAG is Used

```python
# Example: Real-time retrieval + LLM generation
import requests
data = requests.get('https://api.example.com/data').json()
response = llm(f"Analyze the following data: {data}")
print(response)
```

* Enterprise chatbots
* Research assistants
* Legal & medical AI
* Customer support systems
* Agentic AI applications
* Decision support systems

---

## 7. Key Takeaways

* RAG = **retrieval + generation**
* Grounds LLMs with **external knowledge**
* Modular and flexible
* Foundation for **Agentic AI and MCP-based systems**
