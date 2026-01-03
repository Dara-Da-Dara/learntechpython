# LangGraph Course: Quick Reference Guide for Instructors

## Module-by-Module Teaching Guide

### Module 1: LangGraph Fundamentals & Core Concepts (4-5 hours)
**Learning Objectives:** Students will understand graph-based agent architecture and build their first working agents.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 1.1 | Motivation & problems solved | Discussion: Compare traditional vs. graph-based agents | 30 min |
| 1.2 | Graph structure (nodes/edges) | Code along: Build Hello World graph | 45 min |
| 1.3 | Core components assembly | Code along: Create StateGraph with multiple components | 45 min |
| 1.4 | Sequential chains | Project: Build 3-step data processing chain | 60 min |
| 1.5 | Router patterns | Project: Implement branching logic | 45 min |
| 1.6 | Manual agent building | Project: Build basic agent with tool calling | 60 min |
| 1.7 | Agent memory intro | Project: Add conversation memory to agent | 45 min |
| 1.8 | LangSmith Studio (opt.) | Demo & exploration: Trace agent execution | 30 min |

**Assessment:** Quiz on components + Working agent code with tools

---

### Module 2: Advanced State Management (4-5 hours)
**Learning Objectives:** Students will design complex state systems and implement custom reducers for multi-agent scenarios.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 2.1 | TypedDict schemas | Code along: Design states for 3 scenarios | 45 min |
| 2.2 | Reducer functions | Code lab: Implement 5 different reducers | 60 min |
| 2.3 | Multiple schemas | Project: Build system with nested states | 60 min |
| 2.4 | Message management | Code lab: Implement token-aware trimming | 45 min |
| 2.5 | Summarization | Project: Build chatbot with auto-summarization | 60 min |
| 2.6 | External memory | Project: Integrate with vector database | 60 min |

**Assessment:** Design document for state schema + Working chatbot with external memory

---

### Module 3: Streaming, Persistence & Human Control (3-4 hours)
**Learning Objectives:** Students will implement human-in-the-loop systems and handle long-running, persistent workflows.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 3.1 | Streaming outputs | Code lab: Implement token-level streaming | 45 min |
| 3.2 | Breakpoints & interrupts | Project: Build persistent agent with checkpoints | 60 min |
| 3.3 | Human feedback | Project: Implement approval workflow | 60 min |
| 3.4 | Dynamic interrupts | Code lab: Context-aware interrupt logic | 45 min |
| 3.5 | Time travel debugging | Demo & exploration: LangSmith time travel | 30 min |

**Assessment:** Working human-in-the-loop system + Code that demonstrates interrupt/resume

---

### Module 4: Multi-Agent & Advanced Patterns (4-5 hours)
**Learning Objectives:** Students will architect and implement sophisticated multi-agent systems with complex coordination.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 4.1 | Parallelization | Project: Parallel document processor | 60 min |
| 4.2 | Sub-graphs | Project: Build hierarchical agent system | 60 min |
| 4.3 | Map-reduce | Project: Dynamic task distribution system | 60 min |
| 4.4 | Research assistant | Project: Build multi-step research agent | 60 min |
| 4.5 | Multi-agent orchestration | Project: Supervisor-worker system | 75 min |
| 4.6 | Hierarchical systems | Code lab: Deep hierarchy implementation | 45 min |

**Assessment:** Working multi-agent system + Architecture diagram

---

### Module 5: Memory Systems & Long-Term Context (3-4 hours)
**Learning Objectives:** Students will implement sophisticated memory architectures supporting long-term learning and context preservation.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 5.1 | Memory types | Comparison chart exercise: Design trade-offs | 30 min |
| 5.2 | LangGraph Store | Code lab: Store CRUD operations | 45 min |
| 5.3 | Memory + profile | Project: User-aware agent system | 60 min |
| 5.4 | Memory + collection | Project: Multi-collection data organization | 60 min |
| 5.5 | Long-term agents | Project: Build learning/adaptive agent | 75 min |

**Assessment:** Agent that learns and improves from interactions + Memory architecture document

---

### Module 6: Retrieval-Augmented Generation (3-4 hours)
**Learning Objectives:** Students will build production-grade RAG systems that intelligently retrieve and use context.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 6.1 | RAG fundamentals | Code along: Basic RAG pipeline | 45 min |
| 6.2 | Agentic RAG | Project: Agent deciding when to retrieve | 60 min |
| 6.3 | Advanced patterns | Project: Query rewriting + retry logic | 60 min |
| 6.4 | State management | Code lab: Complex RAG state flow | 45 min |
| 6.5 | Multi-tool RAG | Project: Multi-document RAG system | 60 min |

**Assessment:** Working RAG agent with quality metrics + Performance analysis

---

### Module 7: Production Deployment & Features (2-3 hours)
**Learning Objectives:** Students will deploy agents at scale with proper monitoring and optimization.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 7.1 | Compilation & optimization | Code lab: Optimize agent performance | 45 min |
| 7.2 | Persistence strategies | Project: Setup production checkpointing | 45 min |
| 7.3 | LangSmith debugging | Workshop: Instrument agent with LangSmith | 45 min |
| 7.4 | Scaling strategies | Architecture workshop: Design for scale | 45 min |
| 7.5 | Tool integration | Project: Build sophisticated tool system | 60 min |
| 7.6 | Custom nodes | Code lab: Implement stateful nodes | 45 min |

**Assessment:** Production-ready agent deployed to cloud + Monitoring dashboard

---

### Module 8: Advanced Agent Patterns (3-4 hours)
**Learning Objectives:** Students will implement sophisticated reasoning and self-improvement patterns.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 8.1 | ReAct pattern | Project: Build explicit reasoning agent | 60 min |
| 8.2 | Reflection & refinement | Project: Self-improving agent system | 60 min |
| 8.3 | Supervisor agents | Project: Complex task delegation | 60 min |
| 8.4 | Essay writer | Project: Multi-section generation agent | 60 min |
| 8.5 | Complex conditionals | Code lab: Advanced routing logic | 45 min |
| 8.6 | Cycles & loops | Project: Iterative refinement system | 45 min |

**Assessment:** Agent demonstrating advanced pattern + Pattern documentation

---

### Module 9: Integration & Ecosystem (2-3 hours)
**Learning Objectives:** Students will integrate LangGraph with ecosystem tools and deploy production systems.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 9.1 | LangChain integration | Code lab: Migrate LangChain agents | 45 min |
| 9.2 | Vector databases | Project: Multi-DB retrieval system | 60 min |
| 9.3 | External APIs | Project: Build API-calling agents | 60 min |
| 9.4 | Multi-LLM strategies | Project: Intelligent model selection | 45 min |
| 9.5 | Cloud deployment | Workshop: Deploy to LangGraph Cloud | 45 min |

**Assessment:** Deployed system using multiple integrations + Technical documentation

---

### Module 10: Capstone Project (2-3 hours)
**Learning Objectives:** Students will apply all concepts to build production-grade AI system.

| Lesson | Key Concepts | Hands-On Activity | Time |
|--------|--------------|------------------|------|
| 10.1 | Project planning | Workshop: Scope and plan capstone | 45 min |
| 10.2 | Implementation | Guided implementation with instructor support | 90 min |
| 10.3 | Optimization | Code review and optimization workshop | 60 min |
| 10.4 | Deployment | Deployment checklist completion | 45 min |
| 10.5 | Presentation | Student presentations + peer review | 90 min |

**Assessment:** Working deployed system + Project presentation + Technical documentation

---

## Code Example Templates by Lesson

### Module 1, Lesson 1.2: Simple Graph Starter
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list

def process(state: State):
    # Process logic here
    return state

graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()
```

### Module 2, Lesson 2.2: Reducer Pattern
```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add_messages]
    counter: Annotated[int, add]
    items: Annotated[list, lambda x, y: x + y]
```

### Module 3, Lesson 3.2: Breakpoint Pattern
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_node"],
    interrupt_after=["human_decision"]
)
```

### Module 4, Lesson 4.1: Parallelization
```python
from langgraph.types import Send

def process_tasks(state):
    return [Send("worker", {"task": t}) for t in state["tasks"]]

graph.add_conditional_edges("dispatcher", process_tasks, ["worker"])
```

### Module 6, Lesson 6.2: Agentic RAG
```python
def should_retrieve(state):
    last_msg = state["messages"][-1]
    if "retrieve" in last_msg.content.lower():
        return "retrieve"
    return "generate"

graph.add_conditional_edges("agent", should_retrieve, 
    {"retrieve": "retriever", "generate": "generator"})
```

---

## Assessment Rubric Framework

### Code Quality (25%)
- Correctness: Does code work as intended?
- Clarity: Is code readable and well-organized?
- Documentation: Are complex sections explained?
- Best Practices: Does code follow LangGraph patterns?

### Conceptual Understanding (25%)
- Explanation: Can student explain what they built?
- Design Decisions: Are choices justified?
- Trade-offs: Does student understand alternatives?
- Extensions: Can student extend/modify work?

### Architecture (25%)
- State Design: Is state structure appropriate?
- Flow Control: Are edges and routing logical?
- Scalability: Can system handle growth?
- Integration: Does it work with other components?

### Production Readiness (25%)
- Error Handling: Are failures managed?
- Monitoring: Is system observable?
- Performance: Is system optimized?
- Documentation: Are docs complete?

---

## Classroom Discussion Prompts

### Module 1
- "How is LangGraph's approach different from traditional programming?"
- "When would you use cycles in a graph?"
- "What's the relationship between state and behavior?"

### Module 2
- "Why do reducer functions matter in multi-agent systems?"
- "How would you design state for a complex domain?"
- "What are trade-offs between different schema approaches?"

### Module 3
- "How does human-in-the-loop change agent design?"
- "What's the value of time-travel debugging?"
- "How do you balance automation with human control?"

### Module 4
- "How do you coordinate multiple agents effectively?"
- "What's the difference between orchestrator and supervisor patterns?"
- "How do you handle conflicting agent outputs?"

### Module 5
- "Why is long-term memory critical for agents?"
- "How should agents use historical data?"
- "What security considerations exist for memory?"

### Module 6
- "When should retrieval be agentic vs. automatic?"
- "How do you improve retrieval quality?"
- "What are RAG limitations and when to avoid?"

### Module 7
- "What makes an agent production-ready?"
- "How do you monitor distributed agents?"
- "What are cost implications of different approaches?"

### Module 8
- "How do you implement complex reasoning?"
- "Can agents truly improve themselves?"
- "What are limitations of reflection patterns?"

### Module 9
- "How do you choose between LLM providers?"
- "What integration challenges exist?"
- "How do you handle vendor lock-in?"

### Module 10
- "What surprised you most about building agents?"
- "What would you do differently starting over?"
- "How would you extend your project further?"

---

## Common Student Mistakes & How to Address

| Mistake | Why It Happens | Solution |
|---------|---------------|----------|
| Putting all logic in one node | Easier initially | Show modular decomposition benefits early |
| Ignoring state structure | Seems like internal detail | Emphasize state as core design tool |
| Not using checkpointing | Adds "complexity" | Show persistence saves debugging time |
| Over-paralelizing | Looks efficient | Teach state merging complexities |
| Static graphs for dynamic problems | Easier to code | Show Send() API for dynamic patterns |
| Forgetting about error handling | Works in happy path | Code agents that fail gracefully |
| Not testing state updates | Obvious they work | Show subtle bugs with custom reducers |
| Ignoring costs at scale | Focus on functionality | Calculate actual token/API costs |

---

## Extension Topics for Advanced Students

### Performance Optimization
- State serialization performance
- Caching strategies
- Vector database optimization
- Token counting and cost prediction

### Enterprise Patterns
- Rate limiting and quotas
- Multi-tenant architectures
- Audit logging and compliance
- Cost allocation

### Advanced Reasoning
- Chain-of-thought variants
- Tree-of-thought patterns
- Multi-strategy agent selection
- Monte Carlo tree search in agents

### Specialized Domains
- Code generation agents
- Scientific research agents
- Financial analysis agents
- Customer service systems

---

## Resource Links for Students

### Official Documentation
- https://langchain-ai.github.io/langgraph/ - Main LangGraph docs
- https://python.langchain.com/ - LangChain documentation
- https://smith.langchain.com/ - LangSmith platform

### Learning Platforms
- https://academy.langchain.com/ - LangChain Academy (free)
- https://www.deeplearning.ai/short-courses/ - DeepLearning.AI courses
- https://www.datacamp.com/courses - DataCamp courses

### Community
- https://discord.gg/langchain - LangChain Discord
- https://github.com/langchain-ai/langgraph - GitHub repository
- https://stackoverflow.com/questions/tagged/langgraph - Stack Overflow

---

## Recommended Teaching Schedule (6 weeks)

**Week 1:** Module 1 (Fundamentals) + half of Module 2  
**Week 2:** Finish Module 2 + Module 3 (State Management & Persistence)  
**Week 3:** Module 4 (Multi-Agent) + start Module 5 (Memory)  
**Week 4:** Finish Module 5 + Module 6 (RAG Systems)  
**Week 5:** Module 7 (Production) + Module 8 (Advanced Patterns)  
**Week 6:** Module 9 (Integration) + Module 10 (Capstone)

### Async Option (Self-Paced)
- Students work through modules at own pace
- Weekly office hours for Q&A
- Peer code reviews via GitHub
- Asynchronous project feedback
- Monthly cohort presentations

