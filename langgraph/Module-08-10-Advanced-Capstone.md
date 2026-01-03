# Module 8: Advanced Agent Patterns & Architectures

**Duration:** 3-4 hours  
**Target Audience:** Expert developers building sophisticated systems  
**Learning Outcomes:** Implement advanced reasoning and self-improvement patterns

---

## Module Overview

This module teaches sophisticated agent patterns that enable reasoning, self-correction, and adaptive behavior. You'll implement ReAct (explicit reasoning), Reflection (self-evaluation), specialized agent architectures, and complex workflows.

---

## Lesson 8.1: ReAct (Reasoning + Acting) (60 minutes)

### Learning Objectives
- Implement explicit reasoning steps
- Separate thought from action
- Create interpretable agent traces

### Key Concepts

**ReAct Pattern:**
```
Thought: "I need to find information about..."
Action: Tool(search_documents)
Observation: "Found these results..."
Thought: "Based on these results, I should..."
Action: Tool(synthesize)
Final Answer: ...
```

### Hands-On Activity

```python
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

class ReActState(TypedDict):
    task: str
    thoughts: list[str]
    actions: list[str]
    observations: list[str]
    iteration: int
    final_answer: str

def reasoning_step(state: ReActState) -> ReActState:
    """Agent thinks about what to do"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    thought_prompt = f"""Given this task: {state['task']}
    
Previous thoughts: {state['thoughts'][-1] if state['thoughts'] else 'None'}

What should we think about next? Be specific."""
    
    response = llm.invoke([{"role": "user", "content": thought_prompt}])
    state["thoughts"].append(response.content)
    return state

def action_step(state: ReActState) -> ReActState:
    """Agent decides on action"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    action_prompt = f"""Based on this thought: {state['thoughts'][-1]}

What action should we take? Choose from:
1. search(query)
2. analyze(data)
3. conclude(answer)

Respond with just the action."""
    
    response = llm.invoke([{"role": "user", "content": action_prompt}])
    state["actions"].append(response.content)
    return state

def observation_step(state: ReActState) -> ReActState:
    """Execute action and observe result"""
    action = state["actions"][-1]
    
    # Simulate action execution
    if "search" in action:
        observation = "Found relevant information about the topic"
    elif "analyze" in action:
        observation = "Analysis shows key patterns in the data"
    else:
        observation = "Ready to provide final answer"
    
    state["observations"].append(observation)
    state["iteration"] += 1
    return state

def should_continue(state: ReActState):
    """Check if we should continue reasoning"""
    if state["iteration"] >= 3:
        return "conclude"
    if "conclude" in state["actions"][-1].lower():
        return "conclude"
    return "reason"

def conclude(state: ReActState) -> ReActState:
    """Generate final answer"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    history = "\n".join([
        f"Thought: {t}\nAction: {a}\nObservation: {o}"
        for t, a, o in zip(state["thoughts"], state["actions"], state["observations"])
    ])
    
    conclude_prompt = f"""Based on this reasoning process:

{history}

Provide a final answer to the task: {state['task']}"""
    
    response = llm.invoke([{"role": "user", "content": conclude_prompt}])
    state["final_answer"] = response.content
    return state

# Build ReAct graph
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ReActState)
graph.add_node("reason", reasoning_step)
graph.add_node("act", action_step)
graph.add_node("observe", observation_step)
graph.add_node("conclude", conclude)

graph.add_edge(START, "reason")
graph.add_edge("reason", "act")
graph.add_edge("act", "observe")

def route_after_observe(state: ReActState):
    return should_continue(state)

graph.add_conditional_edges("observe", route_after_observe, {
    "reason": "reason",
    "conclude": "conclude"
})
graph.add_edge("conclude", END)

app = graph.compile()

result = app.invoke({
    "task": "What is machine learning?",
    "thoughts": [],
    "actions": [],
    "observations": [],
    "iteration": 0,
    "final_answer": ""
})

print("Reasoning trace:")
for i, (t, a, o) in enumerate(zip(result["thoughts"], result["actions"], result["observations"])):
    print(f"\nStep {i+1}:")
    print(f"  Thought: {t}")
    print(f"  Action: {a}")
    print(f"  Observation: {o}")
print(f"\nFinal Answer: {result['final_answer']}")
```

---

## Lesson 8.2: Reflection & Reflexion (60 minutes)

### Learning Objectives
- Implement self-evaluation
- Generate critique for improvement
- Iterate on outputs

### Key Concepts

**Reflection Loop:**
```
[Generate] → [Evaluate] → [Good?] 
                            ├─→ Yes: [Done]
                            └─→ No: [Revise] → [Generate]
```

### Hands-On Activity

```python
class ReflectionState(TypedDict):
    query: str
    attempts: int
    current_response: str
    critique: str
    is_good: bool
    final_response: str

def generate_response(state: ReflectionState) -> ReflectionState:
    """Generate initial or revised response"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    if state["attempts"] == 0:
        prompt = state["query"]
    else:
        prompt = f"""Query: {state['query']}

Previous response was critiqued as: {state['critique']}

Please provide a better response."""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["current_response"] = response.content
    state["attempts"] += 1
    return state

def critique_response(state: ReflectionState) -> ReflectionState:
    """Evaluate the response quality"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    critique_prompt = f"""Evaluate this response to the query.

Query: {state['query']}
Response: {state['current_response']}

Provide a brief critique:
1. Is it accurate?
2. Is it complete?
3. Any improvements needed?

Also rate overall quality 1-10."""
    
    response = llm.invoke([{"role": "user", "content": critique_prompt}])
    state["critique"] = response.content
    
    # Simple heuristic: if response mentions "good" or "excellent", mark as good
    state["is_good"] = "good" in response.content.lower() and "8" in response.content or "9" in response.content
    return state

def should_revise(state: ReflectionState):
    """Decide whether to revise or accept"""
    if state["is_good"] or state["attempts"] >= 3:
        return "accept"
    return "revise"

def accept_response(state: ReflectionState) -> ReflectionState:
    """Accept the response"""
    state["final_response"] = state["current_response"]
    return state

# Build reflection graph
graph = StateGraph(ReflectionState)
graph.add_node("generate", generate_response)
graph.add_node("critique", critique_response)
graph.add_node("accept", accept_response)

graph.add_edge(START, "generate")
graph.add_edge("generate", "critique")

graph.add_conditional_edges("critique", should_revise, {
    "revise": "generate",
    "accept": "accept"
})
graph.add_edge("accept", END)

app = graph.compile()

result = app.invoke({
    "query": "Explain quantum computing",
    "attempts": 0,
    "current_response": "",
    "critique": "",
    "is_good": False,
    "final_response": ""
})
```

---

## Lesson 8.3: Supervisor Agents (60 minutes)

### Learning Objectives
- Route tasks to specialized agents
- Coordinate multiple specialized workers
- Synthesize results

### Hands-On Activity

(See Module 4 for detailed supervisor pattern implementation)

---

## Lesson 8.4: Essay Writing & Complex Generation (60 minutes)

### Learning Objectives
- Build multi-step generation pipelines
- Implement section-by-section writing
- Synthesize coherent long-form content

### Hands-On Activity

```python
class EssayState(TypedDict):
    topic: str
    thesis: str
    outline: list[dict]  # {section, description}
    sections: dict  # section -> content
    transitions: dict  # section_pair -> transition
    final_essay: str
    word_count: int

def create_thesis(state: EssayState) -> EssayState:
    """Generate thesis statement"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    response = llm.invoke([
        {"role": "user", "content": f"Create a thesis for essay on: {state['topic']}"}
    ])
    state["thesis"] = response.content
    return state

def create_outline(state: EssayState) -> EssayState:
    """Create essay outline"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    outline_prompt = f"""Create a 4-section outline for this essay:
Topic: {state['topic']}
Thesis: {state['thesis']}

Sections: Introduction, Body 1, Body 2, Conclusion"""
    
    response = llm.invoke([{"role": "user", "content": outline_prompt}])
    
    # Parse into outline
    state["outline"] = [
        {"section": "Introduction", "description": "Introduce topic and thesis"},
        {"section": "Body 1", "description": "First main argument"},
        {"section": "Body 2", "description": "Second main argument"},
        {"section": "Conclusion", "description": "Summarize and conclude"}
    ]
    return state

def write_sections(state: EssayState):
    """Write each section from outline"""
    from langgraph.types import Send
    
    return [
        Send("write_section", {"section_info": section})
        for section in state["outline"]
    ]

def write_section(state: dict) -> dict:
    """Write a single section"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    response = llm.invoke([
        {"role": "user", "content": f"Write content for: {state['section_info']['section']}"}
    ])
    
    return {"sections": {state['section_info']['section']: response.content}}

def synthesize_essay(state: EssayState) -> EssayState:
    """Combine sections into complete essay"""
    essay_parts = []
    for section in state["outline"]:
        section_name = section["section"]
        if section_name in state["sections"]:
            essay_parts.append(state["sections"][section_name])
    
    state["final_essay"] = "\n\n".join(essay_parts)
    state["word_count"] = len(state["final_essay"].split())
    return state

# Build essay writing graph
graph = StateGraph(EssayState)
graph.add_node("thesis", create_thesis)
graph.add_node("outline", create_outline)
graph.add_node("write_sections", write_sections)
graph.add_node("write_section", write_section)
graph.add_node("synthesize", synthesize_essay)

graph.add_edge(START, "thesis")
graph.add_edge("thesis", "outline")
graph.add_edge("outline", "write_sections")
graph.add_edge("write_sections", "write_section")
graph.add_edge("write_section", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()
```

---

## Lesson 8.5: Conditional Workflows (45 minutes)

Complex branching based on state:

```python
def complex_router(state: dict):
    if state.get("requires_research"):
        if state.get("is_urgent"):
            return "fast_research"
        return "thorough_research"
    elif state.get("requires_approval"):
        return "approval"
    else:
        return "direct_response"

graph.add_conditional_edges("analyze", complex_router, {
    "fast_research": "research",
    "thorough_research": "research",
    "approval": "approval",
    "direct_response": "response"
})
```

---

## Lesson 8.6: Cycles & Loops (45 minutes)

Iterative agents that improve solutions:

```python
def iteration_limit(state: dict):
    if state["iterations"] < state["max_iterations"]:
        return "improve"
    return "finalize"

graph.add_conditional_edges("evaluate", iteration_limit, {
    "improve": "improve",
    "finalize": "final"
})
graph.add_edge("improve", "evaluate")
```

---

## Module 8 Assessment

### Project: Advanced Agent System

Build using 3+ advanced patterns:
- ReAct reasoning
- Reflection/self-improvement
- Complex multi-step generation
- Conditional workflows

---

## Key Takeaways

1. **Reasoning transparency:** ReAct makes agent thinking visible
2. **Self-improvement:** Reflection enables better outputs
3. **Specialization:** Experts are better than generalists
4. **Iteration:** Complex tasks need multiple passes
5. **Patterns enable complex behavior:** Combine patterns for power

---

# Module 9: Integration & Ecosystem

**Duration:** 2-3 hours

---

## Module Overview

Integrate LangGraph with the broader ecosystem of tools and deploy to production platforms.

---

## Lesson 9.1: LangChain Integration

```python
# Migrate from LangChain AgentExecutor to LangGraph
# Better control flow, easier debugging, explicit state
```

---

## Lesson 9.2: Vector Database Integration

Support for Pinecone, Weaviate, Chroma:

```python
from langchain_community.vectorstores import Pinecone

vector_store = Pinecone.from_documents(docs, embeddings, index_name="my-index")
```

---

## Lesson 9.3: External APIs & Services

```python
@tool
def call_external_api(endpoint: str, params: dict) -> str:
    import requests
    response = requests.post(endpoint, json=params)
    return response.json()
```

---

## Lesson 9.4: Multi-LLM Strategies

```python
def choose_model(state: dict):
    complexity = evaluate_complexity(state["query"])
    if complexity > 0.8:
        return "gpt-4"
    return "gpt-3.5-turbo"
```

---

## Lesson 9.5: LangGraph Cloud Deployment

Deploy to LangGraph Cloud for:
- Automatic scaling
- Built-in monitoring
- Managed checkpointing
- Cost tracking

---

# Module 10: Capstone Project & Assessment

**Duration:** 2-3 hours

---

## Module Overview

Bring together everything learned into a comprehensive capstone project demonstrating mastery of LangGraph.

---

## Capstone Project Requirements

- **Scope:** Build a production-quality AI system
- **Modules:** Use concepts from 5+ modules
- **Patterns:** Implement 3+ architectural patterns
- **Features:** Include memory, persistence, tools, monitoring
- **Code Quality:** Professional-grade implementation
- **Documentation:** Comprehensive docs and deployment guide

---

## Project Ideas

1. **Research Assistant Agent**
   - Literature search
   - Paper summarization
   - Research synthesis
   - Multi-agent coordination

2. **Customer Support System**
   - Intent detection
   - Multi-step resolution
   - Knowledge base integration
   - Escalation handling

3. **Content Generation Pipeline**
   - Topic planning
   - Multi-section writing
   - Iterative refinement
   - Quality evaluation

4. **Data Analysis Agent**
   - Data exploration
   - Statistical analysis
   - Visualization recommendations
   - Insight generation

5. **E-commerce Recommendation Engine**
   - User profiling
   - Product retrieval
   - Personalization
   - Cross-sell/upsell

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Functionality** | 25% | System works end-to-end, handles edge cases |
| **Architecture** | 25% | Good state design, modular patterns, scalable |
| **Integration** | 20% | Multiple ecosystem components, robust APIs |
| **Production Ready** | 20% | Error handling, monitoring, documentation |
| **Innovation** | 10% | Creative use of patterns, novel approach |

---

## Presentation Requirements

- 10-minute demo
- Architecture diagram
- Code walkthrough
- Performance metrics
- Lessons learned
- Future improvements

---

## Final Assessment Checklist

- ✓ Uses 5+ module concepts
- ✓ Includes 3+ architectural patterns
- ✓ Persistent state management
- ✓ Multiple tools or integrations
- ✓ Error handling
- ✓ Monitoring/logging
- ✓ Documentation
- ✓ Deployment guide
- ✓ Performance acceptable
- ✓ Code is clean and maintainable

---

## Certificate of Completion

Upon successful capstone completion:
- Understanding of LangGraph fundamentals and advanced patterns
- Ability to design and build production-grade agentic systems
- Experience with ecosystem integration and deployment
- Portfolio project demonstrating expertise

---

## Path Forward

**Next Steps After This Course:**
- Deploy your capstone to production
- Contribute to LangGraph open source
- Build specialized agent systems for your domain
- Explore emerging agent patterns
- Join the LangGraph community

---

## Resources

- **Official Docs:** https://langchain-ai.github.io/langgraph/
- **Community:** LangChain Discord, GitHub Discussions
- **Examples:** https://github.com/langchain-ai/langgraph/tree/main/examples
- **Blog:** https://blog.langchain.com/
- **Academy:** https://academy.langchain.com/