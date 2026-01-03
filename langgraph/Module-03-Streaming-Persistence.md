# Module 3: Streaming, Persistence & Human Control

**Duration:** 3-4 hours  
**Target Audience:** Intermediate developers with Modules 1-2 knowledge  
**Learning Outcomes:** Stream outputs in real-time, implement checkpointing, build human-in-the-loop systems

---

## Module Overview

Real agents don't run in isolation—they need to stream results to users in real-time, persist their state for long-running tasks, and pause for human approval. This module teaches you the production patterns that make agents trustworthy and interactive. You'll learn streaming for real-time feedback, checkpointing for resilience, breakpoints for control, and human-in-the-loop workflows.

---

## Lesson 3.1: Streaming Outputs (45 minutes)

### Learning Objectives
- Implement token-level streaming for real-time responses
- Understand event-based streaming architecture
- Build responsive user interfaces

### Key Concepts

**Why Stream?**
- **Latency:** Show results as they arrive instead of waiting
- **UX:** Users see progress, not a spinning wheel
- **Cost transparency:** Users see token usage in real-time

**Streaming Architecture:**
```
[Node 1] ──token──→ [Stream Queue] ──token──→ [User UI]
         ──token──→                ──token──→
```

### Hands-On Activity: Real-time Streaming

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

class StreamState(TypedDict):
    prompt: str
    full_response: str

def streaming_node(state: StreamState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Collect full response
    full_text = ""
    for chunk in llm.stream(state["prompt"]):
        full_text += chunk.content
    
    state["full_response"] = full_text
    return state

graph = StateGraph(StreamState)
graph.add_node("stream", streaming_node)
graph.add_edge(START, "stream")
graph.add_edge("stream", END)
app = graph.compile()

# Stream results
for output in app.stream({"prompt": "Write a poem about AI"}):
    print(output)  # See real-time output
```

### Key Streaming Patterns
- **Token streaming:** Use `.stream()` method
- **Event streaming:** Use `stream_mode="values"` or `"updates"`
- **Custom streaming:** Emit events in nodes

---

## Lesson 3.2: Breakpoints & Interrupts (60 minutes)

### Learning Objectives
- Implement static breakpoints for controlled pausing
- Use checkpointing for state persistence
- Resume execution from saved states

### Key Concepts

**Breakpoint Types:**

| Type | Use Case | Syntax |
|------|----------|--------|
| **interrupt_before** | Pause before node runs | `interrupt_before=["node"]` |
| **interrupt_after** | Pause after node runs | `interrupt_after=["node"]` |

### Hands-On Activity: Checkpoint System

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class ApprovalState(TypedDict):
    request: str
    approval_required: bool
    approved: bool
    result: str

def prepare_request(state: ApprovalState):
    state["approval_required"] = "high_value" in state["request"]
    return state

def execute_request(state: ApprovalState):
    if not state["approved"]:
        raise ValueError("Not approved!")
    state["result"] = f"Executed: {state['request']}"
    return state

# Build graph with breakpoint
graph = StateGraph(ApprovalState)
graph.add_node("prepare", prepare_request)
graph.add_node("execute", execute_request)
graph.add_edge(START, "prepare")
graph.add_edge("prepare", "execute")
graph.add_edge("execute", END)

# Compile with checkpointing
checkpointer = InMemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute"]  # Pause before execution
)

# First run: Prepare but don't execute
thread_id = "approval_1"
try:
    result = app.invoke(
        {"request": "high_value transaction", "approved": False},
        config={"configurable": {"thread_id": thread_id}}
    )
except ValueError:
    print("Execution blocked - awaiting approval")

# Human approves
app.update_state(
    {"configurable": {"thread_id": thread_id}},
    {"approved": True}
)

# Resume from checkpoint
result = app.invoke(None, config={"configurable": {"thread_id": thread_id}})
print(result["result"])  # "Executed: ..."
```

---

## Lesson 3.3: Human Feedback Integration (60 minutes)

### Learning Objectives
- Request human input during execution
- Implement approval workflows
- Handle state modifications from humans

### Key Concepts

**Human-in-the-Loop Pattern:**
```
[Agent] → [Breakpoint] → [Wait for Human] → [Resume]
```

### Hands-On Activity: Approval Workflow

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    task: str
    tool_calls: list[dict]
    human_feedback: str
    status: str

def plan_task(state: WorkflowState):
    """Agent plans what to do"""
    state["tool_calls"] = [
        {"tool": "search", "query": "python basics"},
        {"tool": "summarize", "content": "..."},
        {"tool": "post", "platform": "twitter"}
    ]
    state["status"] = "awaiting_approval"
    return state

def execute_approved(state: WorkflowState):
    """Execute only approved tools"""
    approved_tools = state.get("human_feedback", "")
    for tool_call in state["tool_calls"]:
        if tool_call["tool"] in approved_tools:
            print(f"Executing: {tool_call}")
    state["status"] = "completed"
    return state

graph = StateGraph(WorkflowState)
graph.add_node("plan", plan_task)
graph.add_node("execute", execute_approved)
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

checkpointer = InMemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_after=["plan"]  # Pause for approval
)

# Run planning phase
thread_id = "workflow_1"
app.invoke(
    {"task": "Generate content"},
    config={"configurable": {"thread_id": thread_id}}
)

# Human reviews and approves specific tools
app.update_state(
    {"configurable": {"thread_id": thread_id}},
    {"human_feedback": "search,summarize"}  # Approve these, skip post
)

# Resume
result = app.invoke(
    None,
    config={"configurable": {"thread_id": thread_id}}
)
```

---

## Lesson 3.4: Dynamic Breakpoints (45 minutes)

### Learning Objectives
- Create context-aware interrupt logic
- Implement conditional breakpoints
- Handle complex interrupt scenarios

### Key Concepts

**Dynamic Interrupts:** Interrupt based on state, not just node name

### Hands-On Activity

```python
class AdvancedState(TypedDict):
    request_amount: float
    requires_approval: bool
    decision: str

def check_needs_approval(state: AdvancedState):
    """Dynamically decide if approval needed"""
    if state["request_amount"] > 1000:
        state["requires_approval"] = True
    else:
        state["requires_approval"] = False
    return state

def approval_node(state: AdvancedState):
    """Only runs if requires_approval is True"""
    if not state.get("decision"):
        raise ValueError("Awaiting approval decision")
    return state

# Add custom interrupt logic
from langgraph.types import Interrupt

def maybe_interrupt(state: AdvancedState):
    if state.get("requires_approval"):
        raise Interrupt("Approval required for high-value requests")
    return state
```

---

## Lesson 3.5: Time Travel & State History (30 minutes)

### Learning Objectives
- Access execution history
- Revert to previous states
- Use time-travel debugging

### Key Concepts

**State History:** Checkpointer saves every state version

```python
# Get all checkpoints
checkpoints = checkpointer.list_tuples(config)

# Revert to earlier checkpoint
app.invoke(
    None,
    config={**config, "checkpoint_id": earlier_checkpoint_id}
)
```

---

## Module 3 Assessment

### Project: Human-Approved Task Executor

Build a system that:
- Plans tasks (3+ steps)
- Pauses for human approval
- Allows humans to modify the plan
- Executes approved steps only

### Success Criteria
- ✓ Execution pauses at checkpoints
- ✓ State persists correctly
- ✓ Human can modify state
- ✓ Resumption works properly

---

## Key Takeaways

1. **Streaming improves UX:** Real-time feedback matters
2. **Checkpoints provide safety:** Can always resume or rollback
3. **Human control is essential:** Safety-critical systems need approval
4. **State modifications enable workflow:** Humans can guide execution

---

## Next Steps

Module 4 explores sophisticated multi-agent architectures using these persistence and control techniques to build coordinated systems.

---

## Resources

- [LangGraph Streaming Guide](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- [Checkpointing Documentation](https://langchain-ai.github.io/langgraph/concepts/persistent_storage/)
- [Human-in-the-Loop Patterns](https://langchain-ai.github.io/langgraph/tutorials/human-in-the-loop/)