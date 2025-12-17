# Types of RAG (Retrieval-Augmented Generation) Models – Code Examples

This document adds **Python-based, minimal code examples** for major RAG types. Examples use **LangChain-style abstractions** and are **conceptual + runnable with minor setup**.

---

## 1. Vanilla (Classical) RAG

### Theory

Vanilla RAG is the **foundational architecture** of Retrieval-Augmented Generation. The model answers a user query by first retrieving relevant documents from an external knowledge base and then passing those documents as context to a Large Language Model (LLM).

**Key characteristics**:

* Single retrieval step
* No query refinement or feedback loop
* Static knowledge base
* Simple and fast

**Strengths**:

* Easy to implement
* Good baseline for RAG systems

**Limitations**:

* May include irrelevant context
* Higher hallucination risk

### Code Example

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings()
vectordb = FAISS.load_local("docs_index", embeddings)

llm = OpenAI()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

response = qa.run("What is transfer learning?")
print(response)
```

---

## 2. Advanced RAG (with Reranking)

### Theory

Advanced RAG improves upon Vanilla RAG by **enhancing the retrieval quality**. It typically retrieves more documents than needed and then applies reranking or filtering to select the most relevant chunks.

**Key enhancements**:

* Query rewriting
* Context compression
* Cross-encoder or reranker models

**Why it matters**:
LLMs are highly sensitive to input context. Better context → better answers.

### Code Example

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank()
retriever = ContextualCompressionRetriever(
    base_retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
    base_compressor=compressor
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print(qa.run("Explain multimodal transfer learning"))
```

---

## 3. Multi-Stage RAG

### Theory

Multi-Stage RAG performs retrieval in **multiple passes**. The output of one retrieval step is used to refine the query or guide the next retrieval.

**Used when**:

* Questions require deep reasoning
* Information is scattered across documents

**Example**:
Research assistants, legal reasoning systems.

### Code Example

```python
def refine_query(query):
    return f"Detailed academic explanation of: {query}"

initial_docs = vectordb.similarity_search("Explain RAG")
refined_query = refine_query("Explain RAG")
final_docs = vectordb.similarity_search(refined_query)

context = " ".join([d.page_content for d in final_docs])
answer = llm(context)
print(answer)
```

---

## 4. Hybrid RAG (Dense + Sparse)

### Theory

Hybrid RAG combines **dense vector search** (semantic similarity) with **sparse retrieval** (keyword-based search like BM25).

**Why hybrid works**:

* Dense search captures meaning
* Sparse search captures exact terms

**Result**: Higher recall and precision.

### Code Example

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25 = BM25Retriever.from_documents(docs)
dense = vectordb.as_retriever()

ensemble = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.4, 0.6]
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=ensemble)
print(qa.run("Define overfitting"))
```

---

## 5. Agentic RAG

### Theory

Agentic RAG treats the LLM as an **autonomous agent** that can decide:

* When to retrieve
* Which tools to use
* Whether to retrieve again

This aligns with **Agentic AI and MCP-based systems**.

**Frameworks**:
AutoGen, CrewAI, LangGraph, Semantic Kernel

### Code Example

```python
from langchain.agents import initialize_agent, Tool

def retrieve_tool(query):
    return vectordb.similarity_search(query)

retrieval_tool = Tool(
    name="Document Retriever",
    func=retrieve_tool,
    description="Fetch relevant documents"
)

agent = initialize_agent(
    tools=[retrieval_tool],
    llm=llm,
    agent="zero-shot-react-description"
)

agent.run("Find and summarize RAG challenges")
```

---

## 6. Knowledge Graph RAG (KG-RAG)

### Theory

KG-RAG retrieves knowledge from **graph databases** instead of flat documents. Data is stored as entities and relationships.

**Advantages**:

* Strong factual grounding
* Explainable reasoning paths

**Use cases**:
Healthcare, finance, compliance.

### Code Example

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687")

with driver.session() as session:
    result = session.run(
        "MATCH (n:Concept)-[:RELATES_TO]->(m) RETURN n.name, m.name"
    )
    facts = list(result)

context = str(facts)
print(llm(context + "Explain relationships"))
```

---

## 7. Multimodal RAG

### Theory

Multimodal RAG extends retrieval beyond text to include **images, audio, video, and tables**. Each modality has its own embedding space.

**Example**:
Medical image + clinical notes → diagnosis support.

### Code Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('clip-ViT-B-32')

image_embedding = model.encode("image: crop disease leaf")
text_embedding = model.encode("What disease affects wheat leaves?")

similarity = image_embedding @ text_embedding.T
print(similarity)
```

---

## 8. Domain-Specific RAG

### Theory

Domain-Specific RAG uses **custom embeddings, chunking strategies, and terminology** tailored to a specific industry or field.

**Why needed**:
General embeddings often fail on specialized vocabulary.

### Code Example

```python
medical_embeddings = HuggingFaceEmbeddings(
    model_name="emilyalsentzer/Bio_ClinicalBERT"
)

medical_db = FAISS.load_local("medical_index", medical_embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=medical_db.as_retriever())

print(qa.run("Explain symptoms of anemia"))
```

---

## 9. Self-RAG

### Theory

Self-RAG allows the model to **evaluate its own responses**. If the answer is incomplete or low quality, the system re-triggers retrieval.

**Benefit**:
Reduces hallucinations and overconfidence.

### Code Example

```python
answer = qa.run("Explain backpropagation")

review_prompt = f"Is this answer correct and complete? {answer}"
review = llm(review_prompt)

if "incomplete" in review.lower():
    answer = qa.run("Explain backpropagation with math")

print(answer)
```

---

## 10. Real-Time / Live RAG

### Theory

Live RAG connects the LLM to **real-time data sources** such as APIs, databases, or streaming systems.

**Typical applications**:
Stock analysis, weather systems, monitoring dashboards.

### Code Example

```python
import requests

def live_data(query):
    return requests.get("https://api.example.com/data").json()

context = str(live_data("stock price"))
print(llm(context + "Analyze trend"))
```

---

## Final Notes

✔ These examples are **training-friendly**
✔ Can be expanded into **labs or assignments**
✔ Suitable for **GenAI, Agentic AI, and RAG courses**

---

If you want next:

* Convert this into **hands-on lab exercises**
* Add **architecture diagrams**
* Align with **AWS Bedrock / PartyRock / SageMaker**
* Create **assessment questions + solutions**
