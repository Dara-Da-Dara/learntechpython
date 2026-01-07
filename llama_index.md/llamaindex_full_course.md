# LlamaIndex Full Course for Students (20-Hour Curriculum)

## Course Overview
This 20-hour course covers LlamaIndex from basics to advanced agentic systems, aligning with RAG, multi-agent patterns, and vector databases. Students will spend approximately 40% of the time on theory, 50% on hands-on coding, and 10% on exercises.

**Prerequisites:** Python basics, familiarity with LLMs (OpenAI/Groq APIs)

**Setup:**
```
pip install llama-index llama-index-embeddings-openai llama-index-llms-openai chromadb streamlit
```

**Time Breakdown:**
- Modules 1-4 (Basics, 6h)
- Modules 5-8 (Intermediate RAG, 6h)
- Modules 9-12 (Advanced Agents, 8h)

---

## Module 1: Introduction to LlamaIndex (1.5h)
**Concepts and Theory:**
- LlamaIndex is a framework designed to connect Large Language Models (LLMs) to structured and unstructured data efficiently.
- Core components: Documents → Nodes → Index → Query Engine.
- **Why LlamaIndex is important:** It simplifies the process of grounding LLM outputs with real-world data.
- LLM Features:
  - Pre-trained on massive text corpora.
  - Understands context and generates coherent, human-like text.
  - Capable of reasoning over documents.
- Versions of LLMs commonly used:
  - GPT-3, GPT-3.5, GPT-4, GroqLLM, LLaMA, Mistral, Falcon.
- **Integration with RAG:**
  - LlamaIndex allows RAG pipelines where the LLM retrieves relevant documents (chunks) to produce accurate, grounded answers.
  - Supports embedding-based retrieval, semantic search, and filtering to ensure responses are contextually valid.

**Code Example:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key="your-key")

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(response)
```
**Exercise:** Load a sample PDF and query it (30min)

---

## Module 2: Data Loading & Parsing (2h)
**Theory:**
- LlamaIndex provides over 100+ loaders via LlamaHub, including for PDFs, web pages, APIs.
- Supports multiple formats and languages, enhancing LLM grounding.
- Nodes: fundamental unit for LLM understanding; each document is split into nodes (sentence, paragraph, or token-level).
- Parsing is crucial because high-quality node creation impacts retrieval accuracy.

**Code Example:**
```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".txt"],
    file_extractor={".pdf": PDFReader()}
)
documents = reader.load_data()
print(f"Loaded {len(documents)} documents")
```
**Exercise:** Load web data and parse Hindi/English mixed documents (45min)

---

## Module 3: Indexing Fundamentals (1.5h)
**Theory:**
- Indexing organizes documents to make retrieval efficient.
- Types of indexes:
  - VectorStoreIndex: semantic search based on embeddings.
  - SummaryIndex: stores condensed document summaries.
  - TreeIndex: hierarchical representation for recursive queries.
- Embeddings translate text into numerical vectors understood by LLMs.
- Effective indexing ensures faster and more accurate RAG responses.

**Code Example:**
```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
```

---

## Module 4: Persistence & Storage (1h)
**Theory:**
- Index persistence avoids re-embedding costs and enables large-scale deployment.
- LlamaIndex supports storage in local files, ChromaDB, Pinecone, Weaviate.
- Persisted indexes allow LLMs to perform grounded queries over time without reprocessing raw data.
- **Key concept:** Storage context separates storage logic from index logic, enabling flexible retrieval and multi-agent usage.

**Code Example:**
```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("rag_course")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
index.storage_context.persist()

# Load later
loaded_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
```
**Exercise:** Persist and reload index (30min)

---

## Module 5: Basic Query Engines (1.5h)
**Theory:**
- Query engines transform user input into retrieval calls for LLMs.
- Streaming responses allow real-time token generation.
- Query engine abstractions separate retrieval logic from LLM generation.
- LLMs combined with query engines produce grounded, dynamic answers.

**Code Example:**
```python
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("Summarize key points")
for token in response.response_gen:
    print(token, end="", flush=True)
```

---

## Module 6: Advanced Retrieval (2h)
**Theory:**
- Top-K retrieval selects the most relevant nodes to improve LLM accuracy.
- Node post-processors like Rerankers filter or score results.
- Query transformations (HyDE) allow LLMs to hypothesize document answers to improve retrieval.
- Advanced retrieval ensures the LLM only sees contextually important data.

**Code Example:**
```python
from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
)
```
**Exercise:** Optimize for multi-document RAG (45min)

---

## Module 7: RAG Optimization (2h)
**Theory:**
- Filters: restrict retrieval to certain metadata, improving relevance.
- Recursive retrieval: dynamically expands queries to capture missing context.
- Routers: direct queries to specialized indexes.
- Integration with vector DBs enables scalable RAG systems.
- LLMs respond more accurately when retrieval is structured and precise.

**Code Example:**
```python
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

filters = MetadataFilters(filters=[ExactMatchFilter(key="category", value="tech")])
retriever = index.as_retriever(filters=filters)
```

---

## Module 8: Evaluation & Fine-Tuning (1.5h)
**Theory:**
- Evaluators check response faithfulness, relevance, and completeness.
- LLM-as-judge enables self-assessment of generated content.
- Fine-tuning improves domain-specific grounding.
- Critical for ensuring RAG outputs remain accurate in high-stakes applications.

**Code Example:**
```python
from llama_index.core.evaluation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator()
eval_result = evaluator.evaluate_response(query_engine, "Your query here")
```

---

## Module 9: Agents & Tools (2h)
**Theory:**
- Agents are LLM-driven systems performing tasks using tools.
- Tools can be query engines, calculators, or custom APIs.
- Multi-agent systems enhance LLM capabilities by specializing agents for subtasks.
- LlamaIndex integrates agents with retrieval, ensuring actions are contextually grounded.

**Code Example:**
```python
from llama_index.agents import OpenAIAgent
from llama_index.core.tools import QueryEngineTool

tool = QueryEngineTool.from_defaults(query_engine)
agent = OpenAIAgent.from_tools([tool], verbose=True)
response = agent.chat("Analyze sales data trends")
```

---

## Module 10: Multi-Agent Systems (2h)
**Theory:**
- Workflows orchestrate agent collaboration.
- Message queues allow distributed agent coordination.
- Multi-agent RAG systems produce robust, grounded outputs by combining knowledge sources.

**Code Example:**
```python
from llama_index.core.workflow import Workflow

class RAGWorkflow(Workflow):
    def __init__(self):
        pass  # Define agent steps
```
**Exercise:** Build researcher + writer agent pair (1h)

---

## Module 11: Integrations & Deployment (2h)
**Theory:**
- Integrating with LangChain enhances agent orchestration.
- Streamlit/Gradio provide interactive UIs for users.
- FastAPI enables scalable API deployment.
- Integration allows LLMs with RAG to serve real-time applications.

**Code Example:**
```python
import streamlit as st

st.title("LlamaIndex RAG Chat")
query = st.text_input("Ask a question:")
if query:
    response = query_engine.query(query)
    st.write(response)
```

---

## Module 12: Advanced Topics & Projects (3h)
**Theory:**
- Multi-modal RAG: incorporate images, PDFs, audio to enhance LLM grounding.
- Custom LlamaParse allows domain-specific preprocessing.
- Local models with Ollama provide AI sovereignty and privacy.
- Capstone project consolidates retrieval, agent, and multi-modal concepts.

**Capstone Project:** Build multi-agent RAG system for educational content analysis (4h self-paced)

---

## Resources & Next Steps
- **Official Docs:** [LlamaIndex Docs](https://docs.llamaindex.ai)
- **GitHub Playbook:** [Playbook](https://github.com/leemark/llamaindex_playbook)
- **DeepLearning.AI Course:** Agentic RAG

Total: ~20 hours with exercises.

