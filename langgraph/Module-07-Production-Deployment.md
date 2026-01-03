# Module 7: Production Deployment & Advanced Features

**Duration:** 2-3 hours  
**Target Audience:** Advanced developers ready for production systems  
**Learning Outcomes:** Deploy agents at scale with monitoring and optimization

---

## Module Overview

This module teaches the final mile: taking tested graphs and turning them into production systems. You'll learn compilation optimizations, persistent storage strategies, LangSmith integration for monitoring, and practical scaling considerations.

---

## Lesson 7.1: Graph Compilation & Optimization (45 minutes)

### Learning Objectives
- Understand compilation process
- Optimize performance
- Prepare graphs for production

### Key Concepts

**Compilation:** Converts StateGraph into executable application

### Hands-On Activity

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class OptimizedState(TypedDict):
    query: str
    response: str
    execution_time: float

def fast_processor(state: OptimizedState):
    # Optimized logic
    state["response"] = "Fast response"
    return state

graph = StateGraph(OptimizedState)
graph.add_node("process", fast_processor)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Compile
app = graph.compile()

# Test performance
import time
start = time.time()
result = app.invoke({"query": "test"})
state["execution_time"] = time.time() - start

print(f"Execution time: {result['execution_time']:.3f}s")
```

**Optimization Techniques:**
1. **Caching:** Cache repeated computations
2. **Batching:** Process multiple items together
3. **Parallelization:** Run nodes concurrently
4. **Streaming:** Return results incrementally

---

## Lesson 7.2: Persistence & Checkpointing (45 minutes)

### Learning Objectives
- Implement persistent checkpointing
- Choose storage backends
- Handle recovery

### Hands-On Activity

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver

# Option 1: In-memory (development)
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Option 2: PostgreSQL (production)
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/langgraph"
)
app = graph.compile(checkpointer=checkpointer)

# Usage with persistence
config = {"configurable": {"thread_id": "user_session_123"}}

# Invocation 1
result1 = app.invoke({"query": "Hello"}, config=config)

# Later: Resume same thread
result2 = app.invoke({"query": "Remember me?"}, config=config)
# All previous state is available

# Access checkpoint history
saved_values = app.get_state(config)
print(f"Saved state: {saved_values}")
```

**Storage Options:**

| Option | Pros | Cons | Use Case |
|--------|------|------|----------|
| **InMemory** | Fast, simple | Lost on restart | Development, testing |
| **Postgres** | Persistent, scalable | Network overhead | Production |
| **SQLite** | Persistent, simple | Single file | Single-machine prod |
| **Cloud** | Scalable, managed | Vendor lock-in | SaaS, multi-region |

---

## Lesson 7.3: Debugging with LangSmith (45 minutes)

### Learning Objectives
- Set up LangSmith tracing
- Visualize graph execution
- Monitor production systems

### Hands-On Activity

```python
import os
from langsmith import Client

# Setup
os.environ["LANGSMITH_API_KEY"] = "your_api_key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-langgraph-app"

# All invocations automatically traced
result = app.invoke({"query": "test"})

# View in LangSmith:
# 1. Go to smith.langchain.com
# 2. Navigate to your project
# 3. See execution traces with:
#    - Timeline of node execution
#    - State changes at each node
#    - Token usage
#    - Costs
#    - LLM inputs/outputs

# Programmatic access
client = Client()
runs = client.list_runs(project_name="my-langgraph-app")

for run in runs:
    print(f"Run: {run.name}")
    print(f"Duration: {run.end_time - run.start_time}")
    print(f"Tokens: {run.total_tokens}")
```

**LangSmith Features:**
- **Tracing:** Automatic execution logging
- **Debugging:** Step through execution
- **Evaluation:** Test agent quality
- **Monitoring:** Production metrics
- **Analytics:** Cost and performance trends

---

## Lesson 7.4: Scaling Agents (45 minutes)

### Learning Objectives
- Understand scaling challenges
- Design for distributed execution
- Optimize costs at scale

### Key Considerations

**Stateful vs. Stateless:**
- Stateful (LangGraph): More complex but powerful
- Stateless: Simpler to scale horizontally

**Scaling Strategy:**
```
[Load Balancer]
      ↓
    [App 1] ← [Shared Postgres] ← [Checkpointer]
    [App 2]
    [App 3]
```

### Hands-On Activity

```python
class ScalableDeployment:
    """Deployment pattern for multiple instances"""
    
    def __init__(self, db_url: str):
        from langgraph.checkpoint.postgres import PostgresSaver
        
        # All instances share same database
        self.checkpointer = PostgresSaver.from_conn_string(db_url)
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
    
    def _build_graph(self):
        """Build graph once, reuse everywhere"""
        graph = StateGraph(State)
        # ... add nodes and edges
        return graph
    
    def invoke(self, input_data: dict, thread_id: str):
        """Invoke with thread ID for persistence"""
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.invoke(input_data, config=config)

# Usage in multiple processes
import multiprocessing

deployment = ScalableDeployment("postgresql://...")

def worker(thread_id: str):
    result = deployment.invoke({"query": "test"}, thread_id)
    return result

if __name__ == "__main__":
    # Multiple workers handling requests
    threads = [
        multiprocessing.Process(target=worker, args=(f"user_{i}",))
        for i in range(10)
    ]
    
    for t in threads:
        t.start()
```

**Cost Optimization:**
- Use cheaper models for simpler tasks
- Implement caching to avoid redundant API calls
- Batch operations when possible
- Monitor token usage

---

## Lesson 7.5: Tool Integration & Function Calling (60 minutes)

### Learning Objectives
- Create reusable tools
- Implement function calling
- Handle tool execution errors

### Hands-On Activity

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import Send

# Define tools
@tool
def calculator(expression: str) -> str:
    """Execute a mathematical expression"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search internal knowledge base"""
    # Simulate search
    return f"Found {top_k} results for: {query}"

@tool  
def send_notification(message: str, recipient: str) -> str:
    """Send a notification to user"""
    return f"Notification sent to {recipient}: {message}"

tools = [calculator, search_knowledge_base, send_notification]

class ToolUsingState(TypedDict):
    query: str
    tool_calls: list[dict]
    tool_results: list[str]
    final_response: str

def agent_with_tools(state: ToolUsingState):
    """Agent decides which tools to use"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [{"role": "user", "content": state["query"]}]
    response = llm_with_tools.invoke(messages)
    
    # Extract tool calls
    tool_calls = []
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append({
                "tool": tc["name"],
                "args": tc["args"]
            })
    
    state["tool_calls"] = tool_calls
    return state

def execute_tools(state: ToolUsingState):
    """Execute requested tools"""
    tool_results = []
    
    for call in state["tool_calls"]:
        tool_name = call["tool"]
        tool_args = call["args"]
        
        # Execute tool
        if tool_name == "calculator":
            result = calculator(tool_args.get("expression", ""))
        elif tool_name == "search_knowledge_base":
            result = search_knowledge_base(tool_args.get("query", ""))
        elif tool_name == "send_notification":
            result = send_notification(
                tool_args.get("message", ""),
                tool_args.get("recipient", "")
            )
        
        tool_results.append(result)
    
    state["tool_results"] = tool_results
    return state

def generate_response(state: ToolUsingState):
    """Synthesize final response"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    tool_context = "\n".join(state["tool_results"]) if state["tool_results"] else "No tools used"
    
    messages = [
        {"role": "user", "content": state["query"]},
        {"role": "system", "content": f"Tool results:\n{tool_context}"}
    ]
    
    response = llm.invoke(messages)
    state["final_response"] = response.content
    return state

# Build graph
graph = StateGraph(ToolUsingState)
graph.add_node("agent", agent_with_tools)
graph.add_node("tools", execute_tools)
graph.add_node("generate", generate_response)

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", "generate")
graph.add_edge("generate", END)

app = graph.compile()

result = app.invoke({
    "query": "Calculate 5 + 3 and search for Python tutorials",
    "tool_calls": [],
    "tool_results": [],
    "final_response": ""
})
```

---

## Lesson 7.6: Custom Node Implementation (45 minutes)

### Learning Objectives
- Create stateful node classes
- Implement error handling
- Add logging and monitoring

### Hands-On Activity

```python
from typing import Any

class CustomProcessingNode:
    """Custom node with state and configuration"""
    
    def __init__(self, model_name: str = "default", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.execution_count = 0
    
    def __call__(self, state: dict) -> dict:
        """Execute node logic"""
        self.execution_count += 1
        
        if self.verbose:
            print(f"[{self.__class__.__name__}] Execution #{self.execution_count}")
            print(f"  Input state keys: {list(state.keys())}")
        
        try:
            # Node logic
            result = self._process(state)
            
            if self.verbose:
                print(f"  Success: {len(result)} outputs")
            
            return result
        
        except Exception as e:
            print(f"  Error: {e}")
            raise
    
    def _process(self, state: dict) -> dict:
        """Override this in subclasses"""
        return state
    
    def get_stats(self) -> dict:
        """Get node statistics"""
        return {
            "executions": self.execution_count,
            "model": self.model_name
        }

class AnalysisNode(CustomProcessingNode):
    """Example custom node for analysis"""
    
    def _process(self, state: dict) -> dict:
        text = state.get("text", "")
        state["analysis"] = f"Analyzed: {len(text)} chars"
        return state

# Use custom node in graph
graph = StateGraph(State)
graph.add_node("analyze", AnalysisNode(verbose=True))
# ... rest of graph

# Access node stats later
node = AnalysisNode()
graph.add_node("my_node", node)
app = graph.compile()
app.invoke({"text": "test"})
print(node.get_stats())  # {"executions": 1, "model": "default"}
```

---

## Module 7 Assessment

### Project: Production-Ready Agent System

**Requirements:**
- Persistent checkpointing
- LangSmith integration  
- Multiple tools with error handling
- Performance monitoring
- Deployment documentation

### Success Criteria
- ✓ State persists across invocations
- ✓ Execution traces visible in LangSmith
- ✓ Tools execute correctly
- ✓ Error handling prevents crashes
- ✓ System handles concurrent requests

---

## Key Takeaways

1. **Compilation prepares for production:** Not just execution, but optimization
2. **Persistence enables reliability:** Recover from failures
3. **Monitoring is essential:** See what's happening at scale
4. **Tools multiply capability:** Let agents use external services
5. **Custom nodes add flexibility:** Implement domain-specific logic

---

## Next Steps

Module 8 explores advanced reasoning patterns that make agents smarter and more capable.

---

## Resources

- [LangGraph Deployment Guide](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Tool Use Best Practices](https://python.langchain.com/docs/concepts/tool_calling/)
- [Production Considerations](https://blog.langchain.com/)