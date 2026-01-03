# Module 4: Multi-Agent & Advanced Patterns

**Duration:** 4-5 hours  
**Target Audience:** Advanced developers with Modules 1-3 knowledge  
**Learning Outcomes:** Build sophisticated multi-agent systems, implement parallelization and hierarchies

---

## Module Overview

This module teaches you to think beyond single agents. You'll build systems with multiple specialized agents working in parallel, orchestrator agents directing worker agents, and hierarchical agent architectures. These patterns solve complex problems that single agents can't handle efficiently.

---

## Lesson 4.1: Parallelization (60 minutes)

### Learning Objectives
- Execute nodes in parallel
- Merge results from parallel branches
- Implement fan-out/fan-in patterns

### Key Concepts

**Parallelization Pattern:**
```
        ┌─→ [Worker1] ┐
[Start]─┼─→ [Worker2] ├─→ [Merge] → [End]
        └─→ [Worker3] ┘
```

All workers run simultaneously, then results merge.

### Hands-On Activity: Parallel Document Processing

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict
from typing import Annotated
from operator import add

class ParallelState(TypedDict):
    documents: list[str]
    analyses: Annotated[list[dict], add]
    summary: str

def distribute_documents(state: ParallelState):
    """Send each document to a worker"""
    return [
        Send("analyze", {"doc": doc, "doc_id": i})
        for i, doc in enumerate(state["documents"])
    ]

def analyze_document(state: dict):
    """Worker analyzes one document"""
    # Simulate analysis
    return {"analyses": [{"doc_id": state["doc_id"], "result": f"Analysis of {state['doc'][:20]}..."}]}

def merge_analyses(state: ParallelState):
    """Combine all analyses"""
    state["summary"] = f"Analyzed {len(state['analyses'])} documents"
    return state

graph = StateGraph(ParallelState)
graph.add_node("distribute", distribute_documents)
graph.add_node("analyze", analyze_document)
graph.add_node("merge", merge_analyses)

graph.add_edge(START, "distribute")
graph.add_edge("distribute", "analyze")  # Sends to multiple instances
graph.add_edge("analyze", "merge")
graph.add_edge("merge", END)

app = graph.compile()

result = app.invoke({
    "documents": ["Doc1", "Doc2", "Doc3"],
    "analyses": [],
    "summary": ""
})

print(f"Processed {len(result['analyses'])} documents")
```

**Key Pattern:** Use `Send()` to distribute work to multiple node instances

---

## Lesson 4.2: Sub-graphs (60 minutes)

### Learning Objectives
- Compose graphs from other graphs
- Encapsulate functionality in sub-graphs
- Manage sub-graph state

### Key Concepts

**Sub-graph Composition:**
```
[Main Graph]
    │
    ├─→ [Sub-Graph 1]
    │   ├─→ [Node A]
    │   └─→ [Node B]
    │
    └─→ [Sub-Graph 2]
        ├─→ [Node C]
        └─→ [Node D]
```

### Hands-On Activity: Hierarchical System

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Sub-graph for text processing
class ProcessingState(TypedDict):
    text: str
    cleaned: str
    analyzed: str

def clean(state: ProcessingState):
    state["cleaned"] = state["text"].lower().strip()
    return state

def analyze(state: ProcessingState):
    state["analyzed"] = f"Analysis: {len(state['cleaned'].split())} words"
    return state

processing_graph = StateGraph(ProcessingState)
processing_graph.add_node("clean", clean)
processing_graph.add_node("analyze", analyze)
processing_graph.add_edge(START, "clean")
processing_graph.add_edge("clean", "analyze")
processing_graph.add_edge("analyze", END)
processing_app = processing_graph.compile()

# Main graph using sub-graph
class MainState(TypedDict):
    input_text: str
    processing_result: str
    final_output: str

def preprocess(state: MainState):
    result = processing_app.invoke({"text": state["input_text"]})
    state["processing_result"] = result["analyzed"]
    return state

def postprocess(state: MainState):
    state["final_output"] = f"Final: {state['processing_result']}"
    return state

main_graph = StateGraph(MainState)
main_graph.add_node("preprocess", preprocess)
main_graph.add_node("postprocess", postprocess)
main_graph.add_edge(START, "preprocess")
main_graph.add_edge("preprocess", "postprocess")
main_graph.add_edge("postprocess", END)

app = main_graph.compile()
result = app.invoke({"input_text": "Hello World"})
```

---

## Lesson 4.3: Map-Reduce Patterns (60 minutes)

### Learning Objectives
- Distribute and aggregate work
- Handle variable-sized inputs
- Implement dynamic task creation

### Key Concepts

**Map-Reduce:**
```
[Input List] → [Map: Process Each] → [Reduce: Aggregate] → [Output]
```

### Hands-On Activity

```python
from langgraph.types import Send

class MapReduceState(TypedDict):
    items: list
    mapped_results: Annotated[list, add]
    final_result: str

def mapper(state: dict):
    """Process one item"""
    return {"mapped_results": [{"item": state["item"], "result": f"Processed: {state['item']}"}]}

def reducer(state: MapReduceState):
    """Combine all results"""
    state["final_result"] = f"Combined {len(state['mapped_results'])} results"
    return state

def distribute(state: MapReduceState):
    """Map phase: distribute items"""
    return [Send("mapper", {"item": item}) for item in state["items"]]

graph = StateGraph(MapReduceState)
graph.add_node("distribute", distribute)
graph.add_node("mapper", mapper)
graph.add_node("reducer", reducer)

graph.add_edge(START, "distribute")
graph.add_edge("distribute", "mapper")
graph.add_edge("mapper", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()

result = app.invoke({
    "items": ["A", "B", "C", "D", "E"],
    "mapped_results": [],
    "final_result": ""
})
```

---

## Lesson 4.4: Research Assistant Architecture (60 minutes)

### Learning Objectives
- Build complex multi-step workflows
- Coordinate planning and execution
- Implement iterative refinement

### Key Concepts

**Research Loop:**
```
[Plan] → [Research] → [Analyze] → [Refine] → [Done?]
                                       ↓
                                   [Yes: Output]
```

### Hands-On Activity: Multi-Step Research

```python
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

class ResearchState(TypedDict):
    question: str
    research_plan: str
    findings: list[str]
    analysis: str
    iterations: int
    final_report: str

def planner(state: ResearchState):
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke([{"role": "user", "content": f"Plan research for: {state['question']}"}])
    state["research_plan"] = response.content
    return state

def researcher(state: ResearchState):
    # Simulate finding sources
    state["findings"] = ["Finding 1", "Finding 2", "Finding 3"]
    return state

def analyzer(state: ResearchState):
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke([{"role": "user", "content": f"Analyze: {str(state['findings'])}"}])
    state["analysis"] = response.content
    state["iterations"] += 1
    return state

def should_refine(state: ResearchState):
    if state["iterations"] < 2:
        return "research"
    return "finalize"

def finalize(state: ResearchState):
    state["final_report"] = f"Report:\n{state['analysis']}"
    return state

graph = StateGraph(ResearchState)
graph.add_node("plan", planner)
graph.add_node("research", researcher)
graph.add_node("analyze", analyzer)
graph.add_node("finalize", finalize)

graph.add_edge(START, "plan")
graph.add_edge("plan", "research")
graph.add_edge("research", "analyze")
graph.add_conditional_edges("analyze", should_refine, {
    "research": "research",
    "finalize": "finalize"
})
graph.add_edge("finalize", END)

app = graph.compile()

result = app.invoke({
    "question": "What is AI?",
    "research_plan": "",
    "findings": [],
    "analysis": "",
    "iterations": 0,
    "final_report": ""
})
```

---

## Lesson 4.5: Multi-Agent Orchestration (75 minutes)

### Learning Objectives
- Implement supervisor pattern
- Coordinate multiple specialized agents
- Handle inter-agent communication

### Key Concepts

**Supervisor Pattern:**
```
                    ┌─→ [SearchAgent] ┐
[UserQuery] → [Supervisor] ┼─→ [AnalysisAgent] ├─→ [Synthesize]
                    └─→ [WritingAgent] ┘
```

### Hands-On Activity: Supervisor System

```python
from langgraph.types import Send

class SupervisorState(TypedDict):
    query: str
    search_results: str
    analysis: str
    final_answer: str

def supervisor(state: SupervisorState):
    """Route query to appropriate agents"""
    if "how" in state["query"]:
        return ["search", "analysis"]
    return ["search"]

def search_agent(state: SupervisorState):
    state["search_results"] = "Found information..."
    return state

def analysis_agent(state: SupervisorState):
    state["analysis"] = f"Analyzed: {state['search_results']}"
    return state

def synthesize(state: SupervisorState):
    state["final_answer"] = f"{state['analysis']}"
    return state

graph = StateGraph(SupervisorState)
graph.add_node("search", search_agent)
graph.add_node("analysis", analysis_agent)
graph.add_node("synthesize", synthesize)

def route_supervisor(state: SupervisorState):
    return [Send(agent, state) for agent in supervisor(state)]

graph.add_conditional_edges(START, route_supervisor)
graph.add_edge("search", "synthesize")
graph.add_edge("analysis", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()

result = app.invoke({"query": "How does photosynthesis work?", "search_results": "", "analysis": "", "final_answer": ""})
```

---

## Lesson 4.6: Hierarchical Agent Systems (45 minutes)

### Learning Objectives
- Build deep hierarchies
- Delegate tasks through levels
- Manage complexity scaling

### Key Concepts

**Hierarchical Levels:**
```
Level 1: Executive (decision-making)
         ↓
Level 2: Managers (coordination)
         ↓
Level 3: Workers (execution)
```

---

## Module 4 Assessment

### Project: Multi-Agent Research System

Build a system with:
- Supervisor agent
- 3+ specialized workers
- Parallelization
- Result synthesis

### Success Criteria
- ✓ Agents run in parallel when possible
- ✓ Supervisor correctly routes
- ✓ Results merge properly
- ✓ Final output is coherent

---

## Key Takeaways

1. **Parallelization scales:** Use `Send()` for fan-out/fan-in
2. **Sub-graphs enable modularity:** Encapsulate reusable patterns
3. **Supervisors coordinate:** Smart routing multiplies capability
4. **Hierarchies manage complexity:** Layers of abstraction help scaling

---

## Resources

- [LangGraph Parallelization](https://langchain-ai.github.io/langgraph/concepts/low_level_conceptual_model/)
- [Sub-graphs Documentation](https://langchain-ai.github.io/langgraph/)
- [Multi-Agent Patterns](https://blog.langchain.com/)