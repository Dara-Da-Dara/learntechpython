# Module 5: Memory Systems & Long-Term Context

**Duration:** 3-4 hours  
**Learning Outcomes:** Build adaptive agents with long-term memory, implement sophisticated memory schemas

---

## Module Overview

This module teaches sophisticated memory architectures that enable agents to learn and adapt from interactions. You'll build systems with dual-layer memory (short-term conversation + long-term knowledge), user profiles that evolve, and memory retrieval that informs better decisions.

---

## Lesson 5.1: Short-term vs. Long-term Memory (30 minutes)

### Key Concepts

**Memory Layers:**
- **Working Memory:** Current conversation (messages list)
- **Episodic Memory:** What happened in past interactions (stored interactions)
- **Semantic Memory:** Facts and knowledge (vector database)
- **Procedural Memory:** How to do things (learned behaviors)

```python
from typing_extensions import TypedDict

class MemoryState(TypedDict):
    # Working memory (current)
    current_messages: list
    
    # Long-term memory (persistent)
    user_profile: dict          # Preferences, history
    interaction_history: list   # Past conversations
    learned_preferences: dict   # Patterns about user
```

---

## Lesson 5.2: LangGraph Store (45 minutes)

### Learning Objectives
- Use LangGraph's native Store for persistence
- Organize data with namespaces
- Query and retrieve efficiently

### Hands-On Activity

```python
from langgraph.store import InMemoryStore

store = InMemoryStore()

# Save data
store.put(("user_data", "user_123"), "preferences", {
    "language": "en",
    "timezone": "EST",
    "communication_style": "concise"
})

# Retrieve
preferences = store.get(("user_data", "user_123"), "preferences")

# List all for user
all_user_data = store.get_all(("user_data", "user_123"))

# Use in graph
class StoreState(TypedDict):
    user_id: str
    query: str
    user_profile: dict

def load_profile(state: StoreState, store: InMemoryStore):
    state["user_profile"] = store.get(("user_data", state["user_id"]), "profile") or {}
    return state

def save_profile(state: StoreState, store: InMemoryStore):
    store.put(("user_data", state["user_id"]), "profile", state["user_profile"])
    return state
```

---

## Lesson 5.3: Memory Schema + Profile (60 minutes)

### Learning Objectives
- Design user profile schemas
- Update profiles from interactions
- Use profiles to personalize responses

### Hands-On Activity

```python
class UserProfile(TypedDict):
    user_id: str
    name: str
    interests: list[str]
    interaction_count: int
    last_interaction: str
    preferred_topics: list[str]
    interaction_style: str

class ProfileState(TypedDict):
    user_id: str
    query: str
    user_profile: UserProfile
    response: str

def extract_interests(state: ProfileState):
    """Extract interests from conversation"""
    # Use LLM to identify interests from query
    if "python" in state["query"].lower():
        if "python" not in state["user_profile"]["interests"]:
            state["user_profile"]["interests"].append("python")
    state["user_profile"]["interaction_count"] += 1
    return state

def personalize_response(state: ProfileState):
    """Tailor response to user profile"""
    topics = ", ".join(state["user_profile"]["preferred_topics"])
    state["response"] = f"Based on your interest in {topics}: ..."
    return state
```

---

## Lesson 5.4: Memory Schema + Collection (60 minutes)

### Learning Objectives
- Organize related data in collections
- Implement scalable memory systems
- Query across collections

### Hands-On Activity

```python
from typing_extensions import TypedDict

class ConversationMemory(TypedDict):
    conversation_id: str
    timestamp: str
    messages: list
    summary: str

class DocumentMemory(TypedDict):
    doc_id: str
    content: str
    embedding: list[float]
    relevance_tags: list[str]

class MemorySystemState(TypedDict):
    query: str
    relevant_conversations: list[ConversationMemory]
    relevant_documents: list[DocumentMemory]
    synthesis: str

def retrieve_relevant_memories(state: MemorySystemState, store):
    """Find relevant past conversations and documents"""
    # Retrieve similar conversations
    conversations = store.get_all(("conversations", "all"))
    state["relevant_conversations"] = conversations[:3]
    
    # Retrieve related documents
    documents = store.get_all(("documents", "all"))
    state["relevant_documents"] = documents[:3]
    
    return state

def synthesize_with_memory(state: MemorySystemState):
    """Use memories to inform response"""
    memory_context = f"Prior conversations: {len(state['relevant_conversations'])}"
    memory_context += f"\nRelevant docs: {len(state['relevant_documents'])}"
    state["synthesis"] = f"Based on memory, {state['query']} - {memory_context}"
    return state
```

---

## Lesson 5.5: Building Long-term Memory Agents (75 minutes)

### Learning Objectives
- Implement complete memory system
- Build agents that learn from history
- Demonstrate adaptive behavior

### Hands-On Activity: Learning Agent

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

class LearningAgentState(TypedDict):
    user_id: str
    query: str
    interaction_history: list
    learned_patterns: dict
    response: str
    memory_updated: bool

def recall_learning(state: LearningAgentState):
    """Use past learnings to inform current response"""
    patterns = state["learned_patterns"]
    history_summary = f"Interaction count: {len(state['interaction_history'])}"
    return state

def generate_response(state: LearningAgentState):
    """Generate response informed by history"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    context = f"History: {state['interaction_history'][-5:]}"  # Last 5
    response = llm.invoke([{"role": "user", "content": f"{context}\nQuery: {state['query']}"}])
    state["response"] = response.content
    return state

def update_learnings(state: LearningAgentState):
    """Learn from this interaction"""
    state["learned_patterns"]["queries"].append(state["query"])
    state["learned_patterns"]["last_response"] = state["response"]
    state["memory_updated"] = True
    return state

graph = StateGraph(LearningAgentState)
graph.add_node("recall", recall_learning)
graph.add_node("generate", generate_response)
graph.add_node("update", update_learnings)

graph.add_edge(START, "recall")
graph.add_edge("recall", "generate")
graph.add_edge("generate", "update")
graph.add_edge("update", END)

app = graph.compile()

# Use with persistent store
initial_state = {
    "user_id": "user_123",
    "query": "Tell me about ML",
    "interaction_history": [],
    "learned_patterns": {"queries": [], "last_response": ""},
    "response": "",
    "memory_updated": False
}

result = app.invoke(initial_state)
```

---

## Module 5 Assessment

### Project: Adaptive Personal Assistant

Build an agent that:
- Maintains user profile
- Learns preferences from interactions
- Uses memories to personalize responses
- Demonstrates improvement over interactions

### Success Criteria
- ✓ Profile updates correctly
- ✓ Memories inform responses
- ✓ Agent shows learning
- ✓ Data persists correctly

---

## Key Takeaways

1. **Multi-layer memory:** Combine short and long-term storage
2. **Profiles enable personalization:** User data informs better responses
3. **Learning from history:** Agents improve through interaction
4. **Structured storage:** Organized memory is retrievable memory

---

## Resources

- [LangGraph Store Documentation](https://langchain-ai.github.io/langgraph/)
- [Memory Management Guide](https://langchain-ai.github.io/langgraph/concepts/)
- [User Profile Patterns](https://blog.langchain.com/)