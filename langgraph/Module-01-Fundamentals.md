# Module 1: LangGraph Fundamentals & Core Concepts

**Duration:** 4-5 hours  
**Target Audience:** Beginners with Python basics and LLM knowledge  
**Learning Outcomes:** Build basic graphs, understand core components, implement simple agents

---

## Module Overview

This module introduces the graph-based architecture that makes LangGraph powerful. Instead of traditional sequential function calls, you'll learn how to think in terms of **states**, **nodes**, and **edges**—the building blocks of intelligent agent systems. By the end of this module, you'll have built your first working agent that can use tools and maintain conversation state.

---

## Lesson 1.1: Introduction & Motivation (30 minutes)

### Learning Objectives
- Understand why graph-based agent architecture is necessary
- Recognize limitations of traditional LLM application design
- Appreciate the benefits of state management in agent systems

### Key Concepts

**The Problem with Traditional Approaches:**
- **Stateless chains:** Each call is independent; no memory between invocations
- **Complex loops:** Managing agent loops manually is error-prone
- **Tool calling chaos:** Coordinating multiple tools and retries becomes spaghetti code
- **Debugging nightmare:** Hard to trace execution flow and state changes

**Why LangGraph Solves This:**
- **Explicit state management:** All agent data flows through a defined state structure
- **Visual architecture:** Graphs can be visualized, understood, and debugged easily
- **Built-in patterns:** Common agent patterns (routing, loops, parallelization) are native
- **Deterministic execution:** Clear control flow makes systems predictable and testable

### Discussion Topics
- Compare traditional nested function calls vs. graph-based execution
- Think about a real-world problem you'd want to solve with an agent
- How would state management help in that problem?

### Code Exploration
Show students the contrast:

**Traditional approach (messy):**
```python
response = chain.invoke(input)
if "tool_call" in response:
    tool_result = execute_tool(response["tool"])
    response = chain.invoke({**input, "tool_result": tool_result})
    # What if this needs retrying? What if multiple tools?
```

**LangGraph approach (clean):**
```python
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")
# Control flow is explicit and visual
```

---

## Lesson 1.2: Simple Graph Architecture (45 minutes)

### Learning Objectives
- Understand the three core components of a graph
- Build your first "Hello World" LangGraph
- Understand the graph lifecycle from creation to execution

### Key Concepts

**Core Components:**

| Component | Purpose | Example |
|-----------|---------|---------|
| **State** | Shared data structure | `{"messages": [], "count": 0}` |
| **Nodes** | Computational units | `def process(state): return {...}` |
| **Edges** | Connections defining flow | `graph.add_edge("node1", "node2")` |

**Graph Lifecycle:**
1. Define state schema
2. Create StateGraph instance
3. Add nodes (computational units)
4. Add edges (control flow)
5. Compile the graph
6. Invoke with input

### Hands-On Activity: Hello World Graph

**Starter Code:**
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Step 1: Define State
class State(TypedDict):
    name: str
    greeting: str

# Step 2: Define Nodes (functions that process state)
def greet(state: State):
    state["greeting"] = f"Hello, {state['name']}!"
    return state

# Step 3: Build Graph
graph = StateGraph(State)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

# Step 4: Compile
app = graph.compile()

# Step 5: Run
result = app.invoke({"name": "World"})
print(result)  # Output: {"name": "World", "greeting": "Hello, World!"}
```

**Guided Exploration:**
- Modify to add a second node that exclaims the greeting
- Add another node that counts characters in the greeting
- Connect nodes in different orders

### Visualization
Show the graph structure visually (nodes as boxes, edges as arrows):
```
[START] → [greet] → [END]
```

### Key Takeaways
- State flows through the graph unchanged unless nodes modify it
- Each node is a pure function (same input = same output)
- Order matters: edges define the execution sequence

---

## Lesson 1.3: LangGraph Core Components (45 minutes)

### Learning Objectives
- Master the five core building blocks: State, Nodes, Edges, START, END
- Understand how data flows through each component
- Build confidence with component assembly

### Key Concepts

**1. State: The Contract**
```python
from typing_extensions import TypedDict

class ConversationState(TypedDict):
    messages: list        # What messages exist?
    current_task: str     # What are we working on?
    tool_calls: list      # What tools were called?
```

State is like a contract between nodes. Each node receives it, can modify it, must return it.

**2. Nodes: The Workers**
```python
# Function-based node
def process_node(state: State):
    # Read from state
    messages = state["messages"]
    # Do work
    result = analyze(messages)
    # Update and return state
    state["messages"].append(result)
    return state

# Every node: State → State
```

**3. Edges: The Highways**
```python
# Simple edge: A always leads to B
graph.add_edge("node_a", "node_b")

# Conditional edge: A leads to different nodes based on state
def router(state):
    if state["task"] == "generate":
        return "generator"
    else:
        return "retriever"

graph.add_conditional_edges("router", router)
```

**4. START & END: The Boundaries**
```python
graph.add_edge(START, "first_node")    # Where does execution begin?
graph.add_edge("last_node", END)       # Where does it finish?
```

### Hands-On Activity: Multi-Component System

**Build a processing pipeline:**
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class PipelineState(TypedDict):
    raw_text: str
    cleaned_text: str
    analysis: str

def clean_text(state: PipelineState):
    state["cleaned_text"] = state["raw_text"].lower().strip()
    return state

def analyze_text(state: PipelineState):
    state["analysis"] = f"Length: {len(state['cleaned_text'])}"
    return state

graph = StateGraph(PipelineState)
graph.add_node("clean", clean_text)
graph.add_node("analyze", analyze_text)
graph.add_edge(START, "clean")
graph.add_edge("clean", "analyze")
graph.add_edge("analyze", END)

app = graph.compile()
result = app.invoke({"raw_text": "  HELLO WORLD  "})
```

**Assignments:**
- Add a third node that reverses the cleaned text
- Create a node that counts words
- Connect them all in a meaningful order

### Component Assembly Checklist
- [ ] State defined with all necessary fields
- [ ] Each node is a pure function
- [ ] START connects to first node
- [ ] All nodes connect to something
- [ ] END is reachable from some node
- [ ] Graph compiles without errors

---

## Lesson 1.4: Building Chains with LangGraph (60 minutes)

### Learning Objectives
- Build sequential multi-step workflows
- Understand linear execution models
- Implement your first data processing chain

### Key Concepts

**Chain Pattern: Sequential Processing**
```
[START] → [Step1] → [Step2] → [Step3] → [END]
```

Each step processes the output of the previous step. This is the foundation for more complex patterns.

### Hands-On Activity: Data Processing Chain

**Build a 3-step text processing chain:**

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class TextChainState(TypedDict):
    input_text: str
    step1_output: str
    step2_output: str
    step3_output: str

def step1_uppercase(state: TextChainState):
    """Step 1: Convert to uppercase"""
    state["step1_output"] = state["input_text"].upper()
    return state

def step2_reverse(state: TextChainState):
    """Step 2: Reverse the text"""
    state["step2_output"] = state["step1_output"][::-1]
    return state

def step3_add_marker(state: TextChainState):
    """Step 3: Add a marker"""
    state["step3_output"] = f"[PROCESSED] {state['step2_output']}"
    return state

# Build the chain
graph = StateGraph(TextChainState)
graph.add_node("step1", step1_uppercase)
graph.add_node("step2", step2_reverse)
graph.add_node("step3", step3_add_marker)

# Connect sequentially
graph.add_edge(START, "step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)

# Compile and run
app = graph.compile()
result = app.invoke({"input_text": "hello"})
print(result["step3_output"])  # [PROCESSED] OLLEH
```

**Project Assignment:**
Create a 4-step chain for:
- Reading CSV data
- Filtering by condition
- Transforming each row
- Generating a summary

### Chain Best Practices
- ✓ Each step should be independently testable
- ✓ State should have clear input/output for each step
- ✓ Avoid side effects (don't call external APIs without state tracking)
- ✓ Include error handling in each node

---

## Lesson 1.5: Router Patterns (45 minutes)

### Learning Objectives
- Implement conditional branching in graphs
- Create decision points based on state
- Build adaptive workflows

### Key Concepts

**Router Pattern: Decision-Based Routing**
```
          ├─→ [Path A]
[START] → [Router] ┤
          ├─→ [Path B]
          └─→ [Path C]
```

A router node inspects state and directs execution to different paths.

### Hands-On Activity: Intelligent Router

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class RouterState(TypedDict):
    query: str
    intent: str
    result: str

def classify_intent(state: RouterState):
    """Determine what the user wants"""
    query = state["query"].lower()
    if "weather" in query:
        state["intent"] = "weather"
    elif "news" in query:
        state["intent"] = "news"
    else:
        state["intent"] = "general"
    return state

def get_weather(state: RouterState):
    state["result"] = "🌤️ Sunny, 72°F"
    return state

def get_news(state: RouterState):
    state["result"] = "📰 Top headlines today..."
    return state

def general_response(state: RouterState):
    state["result"] = "ℹ️ I can help with that"
    return state

# Router function
def route_intent(state: RouterState):
    if state["intent"] == "weather":
        return "weather"
    elif state["intent"] == "news":
        return "news"
    else:
        return "general"

# Build graph with conditional edges
graph = StateGraph(RouterState)
graph.add_node("classify", classify_intent)
graph.add_node("weather", get_weather)
graph.add_node("news", get_news)
graph.add_node("general", general_response)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_intent, {
    "weather": "weather",
    "news": "news",
    "general": "general"
})
graph.add_edge("weather", END)
graph.add_edge("news", END)
graph.add_edge("general", END)

app = graph.compile()
result = app.invoke({"query": "What's the weather like?"})
print(result["result"])  # 🌤️ Sunny, 72°F
```

**Project Assignment:**
Build a router for an e-commerce system that directs queries to:
- Product search
- Order tracking
- Returns processing
- Customer support

---

## Lesson 1.6: Agents from Scratch (60 minutes)

### Learning Objectives
- Understand agent loop mechanics
- Implement LLM decision-making in nodes
- Build your first functional agent

### Key Concepts

**Agent Loop: The Core Pattern**
```
[Agent thinks] → [Decides action] → [Uses tool] → [Observes] → [Repeat]
```

An agent continuously loops: LLM decides what to do, system executes it, agent observes the result, repeats until done.

### Hands-On Activity: Calculator Agent

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # Accumulate messages
    current_action: str

def calculator_agent(state: AgentState):
    """LLM decides what to do"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # LLM processes the conversation
    messages = state["messages"]
    response = llm.invoke(messages)
    
    # Determine next action from LLM response
    if "add" in response.content.lower():
        state["current_action"] = "add"
    elif "multiply" in response.content.lower():
        state["current_action"] = "multiply"
    else:
        state["current_action"] = "respond"
    
    # Add LLM response to messages
    state["messages"].append(response)
    return state

def add_tool(state: AgentState):
    """Execute addition"""
    last_msg = state["messages"][-1].content
    # Parse numbers and add
    result = 5 + 3  # Simplified for demo
    state["messages"].append({"role": "tool", "content": f"Result: {result}"})
    return state

def multiply_tool(state: AgentState):
    """Execute multiplication"""
    last_msg = state["messages"][-1].content
    result = 5 * 3  # Simplified for demo
    state["messages"].append({"role": "tool", "content": f"Result: {result}"})
    return state

def should_continue(state: AgentState):
    """Should agent keep running?"""
    if state["current_action"] in ["add", "multiply"]:
        return "tools"
    return "end"

# Build agent graph
graph = StateGraph(AgentState)
graph.add_node("agent", calculator_agent)
graph.add_node("add", add_tool)
graph.add_node("multiply", multiply_tool)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "add",  # Simplified: just use add
    "end": END
})
graph.add_edge("add", "agent")  # Loop back to agent
graph.add_edge("multiply", "agent")  # Loop back to agent

app = graph.compile()

# Run agent
initial_state = {
    "messages": [{"role": "user", "content": "What is 5 + 3?"}]
}
result = app.invoke(initial_state)
```

**Key Agent Concepts:**
- **State accumulates:** Messages list grows with each interaction
- **Agent loops:** After using a tool, it loops back to itself
- **Termination:** Agent decides when to stop
- **Tool calling:** Agent requests specific actions

**Project Assignment:**
Build an agent that:
- Receives user questions
- Decides which of 3 tools to use
- Executes the tool
- Provides a response
- Knows when to stop

---

## Lesson 1.7: Agents with Memory (45 minutes)

### Learning Objectives
- Persist agent state across invocations
- Maintain conversation history
- Build agents that "remember" user context

### Key Concepts

**Persistence Problem:**
Without memory, each invocation starts fresh. With memory (checkpointing), agents maintain continuity.

**Two Patterns:**
1. **In-memory:** State stored in Python (lost on restart)
2. **Persistent:** State stored in database/file (survives restarts)

### Hands-On Activity: Memory-Enabled Chat Agent

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from operator import add
from langchain_openai import ChatOpenAI

class ChatState(TypedDict):
    messages: Annotated[list, add]

def chat_node(state: ChatState):
    """Simple chat that remembers history"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # LLM has access to full message history
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    return state

# Build graph
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Compile with in-memory checkpointer
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Use a thread_id to maintain conversation
thread_id = "user_123"

# First interaction
result1 = app.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config={"configurable": {"thread_id": thread_id}}
)

# Second interaction - agent remembers
result2 = app.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config={"configurable": {"thread_id": thread_id}}
)
# LLM will remember: "Your name is Alice"
```

**Memory Architecture:**
```
Invocation 1: [User msg] → [LLM] → [Response] → [Checkpoint]
                                                       ↓
Invocation 2: [User msg] ← [Loaded checkpoint]
             All previous messages available to LLM
```

**Project Assignment:**
Build a personalized recommendation agent that:
- Remembers user preferences from past interactions
- Uses that history to make better recommendations
- Demonstrates memory across multiple invocations

---

## Lesson 1.8: LangSmith Studio Integration (Optional, 30 minutes)

### Learning Objectives
- Visualize graphs in LangSmith Studio
- Debug agent execution with traces
- Monitor agent behavior

### Key Concepts

**LangSmith Studio:** Web-based tool for:
- Visualizing graph structure
- Tracing execution in real-time
- Debugging state changes
- Testing agents

### Hands-On Activity

1. **Sign up:** Create free account at smith.langchain.com
2. **Set API key:** `export LANGSMITH_API_KEY=...`
3. **Run your graph:** Automatically traces to LangSmith
4. **View in Studio:** See visualized execution

**What You'll See:**
- Graph structure diagram
- Node execution timeline
- State changes at each step
- LLM inputs/outputs
- Tool execution results

---

## Module 1 Assessment

### Knowledge Check Quiz
1. What are the three core components of a LangGraph?
2. How is state passed between nodes?
3. What's the difference between simple edges and conditional edges?
4. Why would you use checkpointing in an agent?
5. Draw the execution flow for a 3-node chain

### Hands-On Project: Build a Simple Agent

**Requirements:**
- Define a clear state structure
- Create 3-5 nodes with specific purposes
- Implement conditional routing
- Demonstrate state accumulation
- Test with multiple invocations

**Rubric:**
- Code correctness (25%): Code runs without errors
- Architecture (25%): Clear state design, logical node structure
- Memory (25%): State properly flows through nodes
- Documentation (25%): Comments explain each component

### Success Criteria
- ✓ Agent responds to inputs correctly
- ✓ State accumulates across interactions
- ✓ Can explain each component's purpose
- ✓ Code is readable and well-documented

---

## Key Takeaways

1. **Graphs provide structure:** Explicit nodes, edges, and state make systems understandable
2. **State is central:** All data flows through the shared state
3. **Nodes are isolated:** Each node is a pure function
4. **Agents loop:** The pattern is: Think → Act → Observe → Repeat
5. **Memory matters:** Checkpointing enables persistent agent behavior

---

## Next Steps

Module 2 builds on these fundamentals by diving deep into **state management**, showing how to design complex state structures, implement custom reducers, and integrate external memory systems for sophisticated agents.

---

## Resources

- [LangGraph Official Docs](https://langchain-ai.github.io/langgraph/)
- [StateGraph API Reference](https://langchain-ai.github.io/langgraph/concepts/low_level_conceptual_model/)
- [LangSmith Setup Guide](https://docs.smith.langchain.com/)
- [LangChain Python Documentation](https://python.langchain.com/)

---

## Appendix: Code Templates

### Minimal Graph Template
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class MyState(TypedDict):
    data: str

def my_node(state: MyState):
    return state

graph = StateGraph(MyState)
graph.add_node("node", my_node)
graph.add_edge(START, "node")
graph.add_edge("node", END)
app = graph.compile()
result = app.invoke({"data": "test"})
```

### Agent Loop Template
```python
def agent_node(state):
    # Agent decision logic
    return state

def tool_node(state):
    # Tool execution
    return state

def should_continue(state):
    # Termination logic
    return "continue" or "end"

graph.add_conditional_edges("agent", should_continue, {
    "continue": "tool",
    "end": END
})
graph.add_edge("tool", "agent")  # Loop back
```

### Router Template
```python
def router_fn(state):
    if condition1:
        return "path1"
    elif condition2:
        return "path2"
    else:
        return "path3"

graph.add_conditional_edges("router", router_fn, {
    "path1": "node1",
    "path2": "node2",
    "path3": "node3"
})
```