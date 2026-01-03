# Module 6: Retrieval-Augmented Generation (RAG) Systems

**Duration:** 3-4 hours  
**Target Audience:** Advanced developers with Modules 1-5 knowledge  
**Learning Outcomes:** Build production-grade RAG systems with intelligent retrieval

---

## Module Overview

RAG enables agents to ground responses in authoritative documents. This module teaches you to build systems that intelligently retrieve relevant information and synthesize it into accurate, contextual responses. You'll progress from basic retrieval to agentic systems that decide when and what to retrieve.

---

## Lesson 6.1: RAG Fundamentals (45 minutes)

### Learning Objectives
- Understand retrieval pipeline architecture
- Implement vector search
- Augment LLM prompts with documents

### Key Concepts

**RAG Pipeline:**
```
[Query] → [Embed] → [Vector Search] → [Retrieve] → [Augment] → [Generate] → [Response]
```

**Why RAG?**
- Grounds responses in facts
- Reduces hallucination
- Enables knowledge updates without retraining
- Provides source attribution

### Hands-On Activity

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing_extensions import TypedDict

# Setup vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings, collection_name="docs")

# Load documents
documents = [
    "Python is a high-level programming language",
    "Machine learning is a subset of artificial intelligence",
    "LangGraph enables building stateful agents",
    "Vector databases enable semantic search"
]

for i, doc in enumerate(documents):
    vector_store.add_texts([doc], metadatas=[{"source": f"doc_{i}"}])

# Define state
class RAGState(TypedDict):
    query: str
    retrieved_docs: list
    context: str
    response: str

def retrieve_documents(state: RAGState):
    """Retrieve relevant documents"""
    docs = vector_store.similarity_search(state["query"], k=3)
    state["retrieved_docs"] = docs
    return state

def augment_prompt(state: RAGState):
    """Create context from retrieved documents"""
    doc_texts = "\n".join([f"- {doc.page_content}" for doc in state["retrieved_docs"]])
    state["context"] = f"Based on these documents:\n{doc_texts}"
    return state

def generate_response(state: RAGState):
    """Generate response using augmented context"""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        {"role": "system", "content": state["context"]},
        {"role": "user", "content": state["query"]}
    ]
    response = llm.invoke(messages)
    state["response"] = response.content
    return state

# Build graph
from langgraph.graph import StateGraph, START, END

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_documents)
graph.add_node("augment", augment_prompt)
graph.add_node("generate", generate_response)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# Test
result = app.invoke({"query": "What is machine learning?", "retrieved_docs": [], "context": "", "response": ""})
print(result["response"])
```

---

## Lesson 6.2: Agentic RAG Architecture (60 minutes)

### Learning Objectives
- Implement agent that decides when to retrieve
- Query rewriting strategies
- Iterative retrieval refinement

### Key Concepts

**Agentic RAG:** Let the agent decide when retrieval is needed

```
[Query] → [Agent thinks] → [Needs retrieval?]
                                   ├─→ Yes: [Retrieve] → [Refine answer]
                                   └─→ No: [Answer directly]
```

### Hands-On Activity

```python
from langchain_openai import ChatOpenAI

class AgenticRAGState(TypedDict):
    query: str
    should_retrieve: bool
    retrieved_docs: list
    agent_reasoning: str
    final_response: str

def agent_decides(state: AgenticRAGState):
    """Agent decides if retrieval is needed"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    decision_prompt = f"""Given this query: "{state['query']}"
    
Do we need to search documents? Answer YES or NO with reasoning.

Query: {state['query']}"""
    
    response = llm.invoke([{"role": "user", "content": decision_prompt}])
    state["agent_reasoning"] = response.content
    state["should_retrieve"] = "YES" in response.content.upper()
    return state

def retrieve_if_needed(state: AgenticRAGState):
    """Only retrieve if agent decided it's needed"""
    if state["should_retrieve"]:
        docs = vector_store.similarity_search(state["query"], k=3)
        state["retrieved_docs"] = docs
    return state

def generate_with_context(state: AgenticRAGState):
    """Generate answer, using context if available"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    messages = [{"role": "user", "content": state["query"]}]
    
    if state["retrieved_docs"]:
        context = "\n".join([d.page_content for d in state["retrieved_docs"]])
        messages.insert(0, {
            "role": "system",
            "content": f"Reference material:\n{context}"
        })
    
    response = llm.invoke(messages)
    state["final_response"] = response.content
    return state

def should_retrieve_logic(state: AgenticRAGState):
    return "retrieve" if state["should_retrieve"] else "generate"

# Build graph
graph = StateGraph(AgenticRAGState)
graph.add_node("decide", agent_decides)
graph.add_node("retrieve", retrieve_if_needed)
graph.add_node("generate", generate_with_context)

graph.add_edge(START, "decide")
graph.add_conditional_edges("decide", should_retrieve_logic, {
    "retrieve": "retrieve",
    "generate": "generate"
})
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()
```

---

## Lesson 6.3: Advanced Retrieval Patterns (60 minutes)

### Learning Objectives
- Implement query rewriting
- Grade document relevance
- Implement retry logic

### Key Patterns

**Pattern 1: Query Rewriting**
```python
def rewrite_query(original_query: str) -> str:
    """Rewrite query to improve retrieval"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""Rewrite this query to make it better for vector search:
Original: {original_query}

Provide just the rewritten query, nothing else."""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content
```

**Pattern 2: Document Grading**
```python
def grade_document(doc: str, query: str) -> float:
    """Score relevance of document to query (0-1)"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""Rate relevance of this document to query (1-5):
Query: {query}
Document: {doc[:200]}...

Respond with just the number."""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    score = int(response.content.strip()) / 5  # Normalize to 0-1
    return score
```

**Pattern 3: Retry with Reformulation**

```python
class RobustRAGState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_docs: list
    grades: list[float]
    avg_relevance: float
    final_response: str
    iterations: int

def rewrite_node(state: RobustRAGState):
    """Improve query formulation"""
    if state["iterations"] == 0:
        state["rewritten_query"] = state["query"]
    else:
        state["rewritten_query"] = rewrite_query(state["query"])
    return state

def retrieve_and_grade(state: RobustRAGState):
    """Retrieve and score documents"""
    docs = vector_store.similarity_search(state["rewritten_query"], k=5)
    state["retrieved_docs"] = docs
    
    # Grade each document
    grades = [grade_document(d.page_content, state["rewritten_query"]) for d in docs]
    state["grades"] = grades
    state["avg_relevance"] = sum(grades) / len(grades) if grades else 0
    return state

def should_retry(state: RobustRAGState):
    """Check if results are good enough"""
    if state["avg_relevance"] >= 0.7 or state["iterations"] >= 2:
        return "generate"
    return "rewrite"

def generate_final(state: RobustRAGState):
    """Generate response"""
    # Use top-graded documents
    sorted_docs = sorted(
        zip(state["retrieved_docs"], state["grades"]),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    context = "\n".join([d[0].page_content for d in sorted_docs])
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    response = llm.invoke([
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user", "content": state["query"]}
    ])
    
    state["final_response"] = response.content
    return state

# Build robust RAG
graph = StateGraph(RobustRAGState)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve_grade", retrieve_and_grade)
graph.add_node("generate", generate_final)

graph.add_edge(START, "rewrite")
graph.add_edge("rewrite", "retrieve_grade")

def route_after_grade(state: RobustRAGState):
    state["iterations"] += 1
    return should_retry(state)

graph.add_conditional_edges("retrieve_grade", route_after_grade, {
    "rewrite": "rewrite",
    "generate": "generate"
})
graph.add_edge("generate", END)

app = graph.compile()
```

---

## Lesson 6.4: Retrieval State Management (45 minutes)

### Learning Objectives
- Design state for complex retrieval pipelines
- Track retrieval metadata
- Manage document flow

### Hands-On Activity

```python
class ComprehensiveRAGState(TypedDict):
    # Input
    original_query: str
    user_id: str
    
    # Retrieval pipeline
    search_queries: list[str]
    raw_retrieved: list  # Before filtering
    scored_documents: list[dict]  # {doc, score, source}
    final_documents: list  # After filtering
    
    # Response generation
    llm_response: str
    sources_cited: list[str]
    
    # Metadata
    retrieval_metadata: dict  # {num_queries, total_docs, avg_score}
    total_tokens: int

def track_retrieval_metrics(state: ComprehensiveRAGState):
    """Update metadata about retrieval"""
    state["retrieval_metadata"] = {
        "num_queries_issued": len(state["search_queries"]),
        "total_docs_retrieved": len(state["raw_retrieved"]),
        "docs_after_filtering": len(state["final_documents"]),
        "avg_relevance_score": sum(d["score"] for d in state["scored_documents"]) / len(state["scored_documents"]) if state["scored_documents"] else 0
    }
    return state

def extract_citations(state: ComprehensiveRAGState):
    """Extract source citations from response"""
    sources = set()
    for doc in state["final_documents"]:
        if "source" in doc:
            sources.add(doc["source"])
    state["sources_cited"] = list(sources)
    return state
```

---

## Lesson 6.5: RAG Agent Tools (60 minutes)

### Learning Objectives
- Create reusable retrieval tools
- Build multi-source RAG agents
- Implement tool calling patterns

### Hands-On Activity

```python
from langchain_core.tools import tool

@tool
def search_internal_docs(query: str, top_k: int = 3) -> str:
    """Search internal document database"""
    results = vector_store.similarity_search(query, k=top_k)
    return "\n".join([f"- {d.page_content}" for d in results])

@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    # Would call actual web search API
    return f"Web results for: {query}"

@tool
def get_recent_news(topic: str) -> str:
    """Get recent news on topic"""
    return f"Recent news about {topic}"

class RAGAgentState(TypedDict):
    query: str
    tool_choice: str
    search_results: str
    reasoning: str
    final_answer: str

def select_tools(state: RAGAgentState):
    """Agent selects which tools to use"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    tool_options = """
Available tools:
1. search_internal_docs - Search internal knowledge base
2. search_web - Search the web
3. get_recent_news - Get recent news

Which tool(s) should we use?"""
    
    response = llm.invoke([
        {"role": "system", "content": tool_options},
        {"role": "user", "content": state["query"]}
    ])
    
    state["tool_choice"] = response.content
    state["reasoning"] = response.content
    return state

def execute_tools(state: RAGAgentState):
    """Execute selected tools"""
    if "internal" in state["tool_choice"].lower():
        state["search_results"] = search_internal_docs(state["query"])
    elif "web" in state["tool_choice"].lower():
        state["search_results"] = search_web(state["query"])
    elif "news" in state["tool_choice"].lower():
        state["search_results"] = get_recent_news(state["query"])
    return state

def synthesize_answer(state: RAGAgentState):
    """Synthesize final answer from tool results"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    response = llm.invoke([
        {"role": "system", "content": f"Use these results:\n{state['search_results']}"},
        {"role": "user", "content": state["query"]}
    ])
    
    state["final_answer"] = response.content
    return state

# Build multi-tool RAG agent
graph = StateGraph(RAGAgentState)
graph.add_node("select", select_tools)
graph.add_node("execute", execute_tools)
graph.add_node("synthesize", synthesize_answer)

graph.add_edge(START, "select")
graph.add_edge("select", "execute")
graph.add_edge("execute", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()
```

---

## Module 6 Assessment

### Knowledge Check
1. What are the benefits of RAG over training new models?
2. How would you implement query rewriting?
3. Design a state for a complex retrieval pipeline
4. When would agentic retrieval be better than always retrieving?

### Project: Production RAG System

**Requirements:**
- Vector store integration
- Agentic retrieval decisions
- Query rewriting capability
- Document grading/ranking
- Multi-source synthesis
- Proper source attribution

**Rubric:**
- Retrieval Quality (25%): Relevant documents retrieved
- Agent Logic (25%): Smart decision-making on when/what to retrieve
- Architecture (25%): Clean state design, modular code
- Production Readiness (25%): Error handling, logging, metrics

### Success Criteria
- ✓ Correctly retrieves relevant documents
- ✓ Agent makes intelligent retrieval decisions
- ✓ Responses are accurate and properly cited
- ✓ System handles edge cases (no results, poor quality)
- ✓ Performance is acceptable (< 2 seconds per query)

---

## Key Takeaways

1. **Retrieval grounds responses:** Authoritative documents prevent hallucination
2. **Agentic retrieval:** Agents decide when retrieval adds value
3. **Quality matters:** Document grading filters noise
4. **Iterate on retrieval:** Query rewriting improves results
5. **Attribution is crucial:** Always cite sources

---

## Next Steps

Module 7 builds on these systems by teaching how to move RAG agents to production with proper deployment, monitoring, and scaling strategies.

---

## Resources

- [LangChain Vector Store Docs](https://python.langchain.com/docs/integrations/vectorstores/)
- [Semantic Search Guide](https://python.langchain.com/docs/use_cases/rag//)
- [Tool Use in LLMs](https://python.langchain.com/docs/concepts/tool_calling/)
- [RAG Best Practices](https://blog.langchain.com/)