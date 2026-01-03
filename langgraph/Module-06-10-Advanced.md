# Module 6: Retrieval-Augmented Generation (RAG) Systems

**Duration:** 3-4 hours  
**Learning Outcomes:** Build production-grade RAG systems with intelligent retrieval

---

## Module Overview

RAG enables agents to ground responses in authoritative documents. This module teaches you to build systems that intelligently retrieve relevant information and synthesize it into accurate, contextual responses. You'll progress from basic retrieval to agentic systems that decide when and what to retrieve.

---

## Lesson 6.1: RAG Fundamentals (45 minutes)

### Key Concepts

**RAG Pipeline:**
```
[Query] → [Embed] → [Vector Search] → [Retrieve] → [Augment] → [Generate]
```

### Hands-On Activity

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings)

# Load documents
vector_store.add_texts([
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "Agents are autonomous systems"
])

# Retrieve
results = vector_store.similarity_search("What is Python?", k=3)

class RAGState(TypedDict):
    query: str
    documents: list
    response: str

def retrieve(state: RAGState):
    state["documents"] = vector_store.similarity_search(state["query"])
    return state

def generate(state: RAGState):
    doc_context = "\n".join([d.page_content for d in state["documents"]])
    state["response"] = f"Based on: {doc_context}"
    return state
```

---

## Lesson 6.2: Agentic RAG Architecture (60 minutes)

### Learning Objectives
- Implement agent that decides when to retrieve
- Query rewriting strategies
- Iterative retrieval

### Hands-On Activity

```python
class AgenticRAGState(TypedDict):
    query: str
    retrieval_needed: bool
    documents: list
    response: str

def should_retrieve(state: AgenticRAGState):
    # Agent decides if retrieval needed
    state["retrieval_needed"] = "?" in state["query"] or "what" in state["query"]
    return state

def retrieve_if_needed(state: AgenticRAGState):
    if state["retrieval_needed"]:
        state["documents"] = vector_store.similarity_search(state["query"])
    return state

def generate_response(state: AgenticRAGState):
    if state["documents"]:
        context = "\n".join([d.page_content for d in state["documents"]])
        state["response"] = f"From documents: {context}"
    else:
        state["response"] = "From knowledge: ..."
    return state

graph = StateGraph(AgenticRAGState)
graph.add_node("decide", should_retrieve)
graph.add_node("retrieve", retrieve_if_needed)
graph.add_node("generate", generate_response)

graph.add_edge(START, "decide")
graph.add_edge("decide", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
```

---

## Lesson 6.3: Advanced Retrieval Patterns (60 minutes)

### Key Patterns

**Query Rewriting:** Improve retrieval by reformulating queries
**Document Grading:** Score relevance of retrieved documents
**Retry Logic:** Re-retrieve if results poor

### Hands-On Activity

```python
class AdvancedRAGState(TypedDict):
    original_query: str
    reformulated_query: str
    documents: list
    grades: list[dict]
    response: str

def rewrite_query(state: AdvancedRAGState):
    # Improve query for better retrieval
    state["reformulated_query"] = f"Expanded: {state['original_query']}"
    return state

def retrieve_documents(state: AdvancedRAGState):
    state["documents"] = vector_store.similarity_search(state["reformulated_query"])
    return state

def grade_documents(state: AdvancedRAGState):
    # Score relevance
    state["grades"] = [
        {"doc": d.page_content[:50], "relevance": 0.9}
        for d in state["documents"]
    ]
    return state

def should_retry(state: AdvancedRAGState):
    avg_grade = sum(g["relevance"] for g in state["grades"]) / len(state["grades"])
    return "generate" if avg_grade > 0.5 else "rewrite"
```

---

## Lesson 6.4: Retrieval State Management (45 minutes)

State tracks document flow through pipeline:

```python
class FullRAGState(TypedDict):
    query: str
    retrieved_docs: list
    ranked_docs: list
    final_docs: list
    response: str
    retrieval_metadata: dict

def track_retrieval(state: FullRAGState):
    state["retrieval_metadata"]["total_retrieved"] = len(state["retrieved_docs"])
    return state
```

---

## Lesson 6.5: RAG Agent Tools (60 minutes)

Build agents with multiple retrieval tools:

```python
class ToolBasedRAGState(TypedDict):
    query: str
    tool_calls: list
    search_results: str
    synthesis_results: str
    final_answer: str

def plan_retrieval(state: ToolBasedRAGState):
    state["tool_calls"] = ["search_documents", "search_web"]
    return state

def execute_tools(state: ToolBasedRAGState):
    for tool in state["tool_calls"]:
        if tool == "search_documents":
            state["search_results"] = "Found internal docs"
        elif tool == "search_web":
            state["search_results"] += ", and web results"
    return state

def synthesize(state: ToolBasedRAGState):
    state["final_answer"] = f"Synthesized from: {state['search_results']}"
    return state
```

---

## Module 6 Assessment

### Project: Production RAG System

Build a system with:
- Vector store integration
- Agentic retrieval decisions
- Query rewriting
- Document grading
- Multi-source synthesis

---

## Key Takeaways

1. **Retrieval grounds responses:** Authoritative documents improve accuracy
2. **Agentic retrieval:** Let agents decide when to retrieve
3. **Query optimization:** Rewriting improves retrieval quality
4. **Ranking matters:** Grade relevance to filter poor results

---

# Module 7: Production Deployment & Advanced Features

**Duration:** 2-3 hours

---

## Module Overview

This module teaches deployment, monitoring, and optimization for production systems. You'll learn compilation best practices, persistence strategies, LangSmith integration, and scaling considerations.

---

## Lesson 7.1: Graph Compilation & Optimization (45 minutes)

```python
# Compilation with checkpointing
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_node"]
)

# Optimization techniques
# 1. Batch processing
# 2. Caching repeated computations
# 3. Reducing state size
# 4. Parallel execution where possible
```

---

## Lesson 7.2: Persistence & Checkpointing (45 minutes)

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Persistent checkpointing
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
app = graph.compile(checkpointer=checkpointer)

# Recovery from failures
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(input_data, config=config)
# Can resume from this checkpoint even after crash
```

---

## Lesson 7.3: Debugging with LangSmith (45 minutes)

```python
import os
os.environ["LANGSMITH_API_KEY"] = "..."

# Automatic tracing
result = app.invoke(input_data)

# View in LangSmith Studio:
# - Execution timeline
# - State changes
# - Tool calls
# - Costs
```

---

## Lesson 7.4: Scaling Agents (45 minutes)

**Scaling Considerations:**
- Distributed state management
- Load balancing
- Cost optimization
- Monitoring at scale

---

## Lesson 7.5: Tool Integration & Function Calling (60 minutes)

```python
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information"""
    return f"Results for {query}"

tools = [search_tool]

def tool_node(state):
    # Execute tool calls
    return state

# Add to graph
graph.add_node("tools", tool_node)
```

---

## Lesson 7.6: Custom Node Implementation (45 minutes)

```python
class CustomNode:
    def __init__(self, config: dict):
        self.config = config
    
    def __call__(self, state: dict) -> dict:
        # Custom node logic
        return state

graph.add_node("custom", CustomNode(config={"param": "value"}))
```

---

# Module 8: Advanced Agent Patterns & Architectures

**Duration:** 3-4 hours

---

## Module Overview

Advanced reasoning and self-improvement patterns that make agents more capable.

---

## Lesson 8.1: ReAct (Reasoning + Acting) (60 minutes)

```python
class ReActState(TypedDict):
    thoughts: str
    actions: str
    observations: str

def reason(state: ReActState):
    state["thoughts"] = "Let me think about this step by step..."
    return state

def act(state: ReActState):
    state["actions"] = "I will use tool X"
    return state

def observe(state: ReActState):
    state["observations"] = "Tool returned..."
    return state
```

---

## Lesson 8.2: Reflection & Reflexion (60 minutes)

Agents evaluate their own outputs:

```python
class ReflectionState(TypedDict):
    response: str
    critique: str
    revised_response: str

def generate(state: ReflectionState):
    state["response"] = "Initial response"
    return state

def critique(state: ReflectionState):
    state["critique"] = "This needs improvement because..."
    return state

def revise(state: ReflectionState):
    state["revised_response"] = "Improved response..."
    return state
```

---

## Lesson 8.3: Supervisor Agents (60 minutes)

Already covered in Module 4 - Supervisors routing to specialists

---

## Lesson 8.4: Essay Writing & Complex Generation (60 minutes)

Multi-step content generation:

```python
class EssayState(TypedDict):
    topic: str
    outline: str
    sections: dict
    final_essay: str

def create_outline(state: EssayState):
    state["outline"] = "1. Introduction\n2. Body\n3. Conclusion"
    return state

def write_sections(state: EssayState):
    sections = {}
    for section in state["outline"].split("\n"):
        sections[section] = f"Content for {section}"
    state["sections"] = sections
    return state

def compile_essay(state: EssayState):
    state["final_essay"] = "\n".join(state["sections"].values())
    return state
```

---

## Lesson 8.5: Conditional Workflows (45 minutes)

Complex branching:

```python
def complex_router(state):
    if condition1:
        return path1
    elif condition2:
        return path2
    else:
        return path3
```

---

## Lesson 8.6: Cycles & Loops (45 minutes)

Iterative agents:

```python
def should_continue(state):
    if state["iterations"] < 3:
        return "continue"
    return "end"

graph.add_conditional_edges("process", should_continue, {
    "continue": "process",
    "end": END
})
```

---

# Module 9: Integration & Ecosystem

**Duration:** 2-3 hours

---

## Module Overview

Integrate LangGraph with ecosystem tools and deploy to production platforms.

---

## Lesson 9.1: LangChain Integration

Migrate from LangChain agents to LangGraph:

```python
# LangChain agent becomes LangGraph graph
# More explicit control flow
# Better testability
```

---

## Lesson 9.2: Vector Database Integration

```python
from langchain_community.vectorstores import Pinecone

vector_store = Pinecone.from_documents(docs, embeddings)
```

---

## Lesson 9.3: External APIs & Services

```python
@tool
def api_call(endpoint: str) -> str:
    import requests
    response = requests.get(endpoint)
    return response.json()
```

---

## Lesson 9.4: Multi-LLM Strategies

```python
class MultiLLMState(TypedDict):
    query: str
    model_choice: str
    response: str

def choose_model(state: MultiLLMState):
    if "complex" in state["query"]:
        state["model_choice"] = "gpt-4"
    else:
        state["model_choice"] = "gpt-3.5"
    return state
```

---

## Lesson 9.5: LangGraph Cloud & Deployment

```python
# Deploy to LangGraph Cloud
# Automatic scaling
# Built-in monitoring
# Managed checkpointing
```

---

# Module 10: Capstone Project & Assessment

**Duration:** 2-3 hours

---

## Module Overview

Apply everything in a comprehensive capstone project.

---

## Lesson 10.1: Project Ideation & Planning

Plan a real-world AI system using LangGraph.

---

## Lesson 10.2: Implementation Guidance

Step-by-step guidance through implementation.

---

## Lesson 10.3: Optimization & Refinement

Performance profiling and optimization.

---

## Lesson 10.4: Deployment & Monitoring

Move system to production with monitoring.

---

## Lesson 10.5: Capstone Presentation & Feedback

Present project, receive feedback, iterate.

---

## Capstone Project Requirements

- Use 5+ module concepts
- Include multi-agent patterns
- Implement memory/persistence
- Deploy to production platform
- Document thoroughly

---

## Success Metrics

- ✓ System works end-to-end
- ✓ Code is production-quality
- ✓ Documentation is clear
- ✓ Performance is acceptable
- ✓ Demonstration is compelling