# Module 2: Advanced State Management

**Duration:** 4-5 hours  
**Target Audience:** Intermediate developers with Module 1 knowledge  
**Learning Outcomes:** Design complex state structures, implement custom reducers, manage sophisticated memory systems

---

## Module Overview

State is the heart of LangGraph. While Module 1 introduced basic state, this module teaches you to **architect state** for complex systems. You'll master TypedDict schemas, reducer functions, message management, and external memory integration. By the end, you'll be able to design state structures that elegantly support multi-agent systems, conversation history, and persistent knowledge bases.

---

## Lesson 2.1: State Schema Design (45 minutes)

### Learning Objectives
- Design TypedDict schemas for complex scenarios
- Understand state as a contract between nodes
- Apply schema design patterns and best practices

### Key Concepts

**State Schema: The Foundation**

State is the "contract" every node must honor. A well-designed schema makes graphs clear, maintainable, and testable.

```python
from typing_extensions import TypedDict

class ConversationState(TypedDict):
    messages: list              # All conversation history
    current_query: str          # User's current question
    context: str                # Retrieved context
    final_response: str         # Agent's response
```

**Design Principles:**

| Principle | Good | Bad |
|-----------|------|-----|
| **Clear naming** | `user_query` | `q` |
| **Single responsibility** | Separate `messages` and `metadata` | One big `data` field |
| **Immutability-friendly** | Use lists/dicts consistently | Mix types |
| **Documentation** | Comments on complex fields | Unexplained fields |
| **Type hints** | `str`, `list[dict]`, `Optional[str]` | No type hints |

### Hands-On Activity: Schema Design Workshop

**Scenario 1: Research Assistant**
```python
from typing_extensions import TypedDict
from typing import Optional

class ResearchState(TypedDict):
    research_query: str                    # What to research?
    search_results: list[dict]             # Raw search results
    analyzed_papers: list[str]             # Summaries of papers
    current_section: Optional[str]         # What are we writing now?
    outline: dict                          # Research outline
    draft: str                             # In-progress draft
    final_report: Optional[str]            # Completed report
    citation_count: int                    # Number of citations
    metadata: dict                         # Graph execution metadata
```

**Scenario 2: E-commerce Assistant**
```python
class ShoppingState(TypedDict):
    user_id: str
    query: str                             # What user wants
    product_catalog: list[dict]            # Available products
    search_results: list[dict]             # Filtered results
    user_preferences: dict                 # Learned preferences
    selected_product: Optional[dict]       # Current selection
    recommendation_reason: str             # Why this product?
    cart: list[dict]                       # Items to buy
    order_status: str                      # pending/confirmed
    conversation_history: list[dict]       # Full chat history
```

**Scenario 3: Multi-Step Workflow**
```python
class WorkflowState(TypedDict):
    workflow_id: str
    step_number: int
    step_status: str                       # pending/running/completed
    steps_completed: list[str]
    current_data: dict                     # Current transformation
    errors: list[str]                      # Any errors encountered
    logs: list[str]                        # Execution logs
    final_output: Optional[dict]
    execution_time_ms: float
```

**Schema Design Checklist:**
- [ ] Every field serves a clear purpose
- [ ] Fields are named descriptively
- [ ] Types are explicit (no `Any`)
- [ ] Field relationships are clear
- [ ] Could this schema be used by any node?
- [ ] Can you explain why each field exists?

### Project Assignment
Design state schemas for:
1. Customer support chatbot
2. Content generation pipeline
3. Data analysis workflow

For each, identify:
- What data needs persistence?
- What's input vs. internal?
- How do nodes modify state?
- What's the "flow" of data?

---

## Lesson 2.2: Reducer Functions (60 minutes)

### Learning Objectives
- Understand reducer functions for state updates
- Implement custom reducers for lists and counters
- Use `add_messages` for conversation history

### Key Concepts

**The Problem Reducers Solve:**

Without reducers, updating lists becomes messy:
```python
# Manual (error-prone)
state["messages"].append(new_message)  # But what if parallelizing?
state["messages"] += new_messages       # Inconsistent with other fields

# With reducers (declarative)
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Messages accumulate
```

**What is a Reducer?**

A reducer is a function that defines how to merge new values with existing values:
```python
# Reducer signature: (existing_value, new_value) → merged_value

# Example: Addition reducer
def add_reducer(a: int, b: int) -> int:
    return a + b

# Example: List concatenation reducer  
def concat_reducer(a: list, b: list) -> list:
    return a + b
```

**Built-in Reducers:**

| Reducer | Use Case | Merges How |
|---------|----------|-----------|
| `operator.add` | Accumulate counts, append lists | `a + b` |
| `add_messages` | Conversation history | Deduplicates by ID |
| `operator.mul` | Multiply numbers | `a * b` |
| Custom | Domain-specific logic | Your function |

### Hands-On Activity: Reducer Implementation

**Pattern 1: Message Accumulation with `add_messages`**

```python
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import add_messages

class ConversationState(TypedDict):
    # Messages automatically accumulate without duplication
    messages: Annotated[list, add_messages]
    
def agent_node(state: ConversationState):
    # Add new messages
    return {"messages": [new_message]}
    # LangGraph merges using add_messages reducer

# Result: Each invocation adds to the message history
```

**Pattern 2: Counter Reducer**

```python
from typing_extensions import TypedDict
from typing import Annotated
from operator import add

class AnalysisState(TypedDict):
    text: str
    word_count: Annotated[int, add]         # Accumulate counts
    sentence_count: Annotated[int, add]

def count_words(state: AnalysisState):
    count = len(state["text"].split())
    return {"word_count": count}

def count_sentences(state: AnalysisState):
    count = len(state["text"].split("."))
    return {"sentence_count": count}
```

**Pattern 3: Custom Reducer**

```python
from typing import Annotated

def merge_lists(existing: list, new: list) -> list:
    """Merge lists, keeping unique items"""
    return list(set(existing + new))

class DataState(TypedDict):
    items: Annotated[list, merge_lists]
    
def collect_items(state: DataState):
    return {"items": [1, 2, 3]}

def collect_more(state: DataState):
    return {"items": [2, 3, 4]}  # Duplicates removed: {1,2,3,4}
```

**Pattern 4: Complex Reducer**

```python
def merge_metadata(existing: dict, new: dict) -> dict:
    """Merge metadata dicts, new values override old"""
    result = existing.copy()
    result.update(new)
    # Preserve timestamps
    result["last_updated"] = datetime.now()
    return result

class TaskState(TypedDict):
    metadata: Annotated[dict, merge_metadata]
    
# Metadata updates accumulate while preserving structure
```

**Pattern 5: Conditional Reducer**

```python
def smart_merge_strings(existing: str, new: str) -> str:
    """Merge strings intelligently"""
    if new.startswith("+++"):
        return existing + "\n" + new[3:]
    else:
        return new  # Replace mode

class DocumentState(TypedDict):
    content: Annotated[str, smart_merge_strings]
```

### Project Assignment: Build a Tracking System

Create a state with multiple reducers that tracks:
- A growing list of events (with deduplication)
- Cumulative metrics (counters)
- A merged configuration object
- A conversation history using `add_messages`

Test by having different nodes update different fields and verify they merge correctly.

---

## Lesson 2.3: Multiple State Schemas (60 minutes)

### Learning Objectives
- Design systems with nested state structures
- Compose multiple schemas
- Handle complex hierarchical data

### Key Concepts

**When to Use Multiple Schemas:**

```python
# Simple system: Single schema
class SimpleState(TypedDict):
    query: str
    response: str

# Complex system: Multiple schemas
class MainState(TypedDict):
    """Top-level state for orchestrator"""
    query: str
    worker_states: list["WorkerState"]
    final_response: str

class WorkerState(TypedDict):
    """Sub-state for worker agents"""
    task_id: str
    input_data: dict
    result: str
    status: str
```

**Schema Composition Patterns:**

**Pattern 1: Nested Dictionaries**
```python
class NestedState(TypedDict):
    user: dict  # Contains name, id, preferences
    session: dict  # Contains start_time, interactions
    analytics: dict  # Contains metrics

# Nodes work with nested structure
def process_user(state: NestedState):
    state["user"]["interactions"] = 42
    return state
```

**Pattern 2: List of Schemas**
```python
class DocumentState(TypedDict):
    documents: list[dict]  # Each doc: {id, content, metadata}
    processing_queue: list[str]  # Doc IDs to process
    results: list[dict]  # Results: {doc_id, analysis}

def process_all_docs(state: DocumentState):
    for doc in state["documents"]:
        state["results"].append(analyze_doc(doc))
    return state
```

**Pattern 3: Multi-Level Hierarchy**
```python
class HierarchicalState(TypedDict):
    level1: dict  # Top-level context
    level2: dict  # Intermediate data
    level3: dict  # Detailed processing
    metadata: dict  # Shared metadata

# Nodes focus on specific levels
def process_level1(state: HierarchicalState):
    return state  # Modifies level1

def process_level2(state: HierarchicalState):
    return state  # Modifies level2 based on level1
```

### Hands-On Activity: Building Hierarchical Systems

**Build a Multi-Agent Research System:**

```python
from typing_extensions import TypedDict
from typing import Annotated
from operator import add

class ResearcherState(TypedDict):
    """Individual researcher state"""
    researcher_id: str
    assigned_topic: str
    findings: list[str]
    papers_analyzed: int
    quality_score: float

class MainResearchState(TypedDict):
    """Orchestrator state combining researchers"""
    research_question: str
    researchers: list[ResearcherState]      # Multiple sub-states
    consolidated_findings: str
    consensus_reached: bool
    research_metadata: dict

def researcher_agent(state: MainResearchState, researcher_idx: int):
    """Individual researcher processes topic"""
    researcher = state["researchers"][researcher_idx]
    # Analyze papers
    researcher["findings"].append("Finding 1")
    researcher["papers_analyzed"] += 1
    return state

def consolidate_findings(state: MainResearchState):
    """Combine all researchers' findings"""
    all_findings = []
    for researcher in state["researchers"]:
        all_findings.extend(researcher["findings"])
    state["consolidated_findings"] = "; ".join(all_findings)
    return state

def reach_consensus(state: MainResearchState):
    """Check if all researchers agree"""
    avg_quality = sum(r["quality_score"] for r in state["researchers"]) / len(state["researchers"])
    state["consensus_reached"] = avg_quality > 0.8
    return state
```

**Graph Structure:**
```
[START] → [Researcher1] ┐
          [Researcher2] ├→ [Consolidate] → [Consensus] → [END]
          [Researcher3] ┘
```

### Project Assignment
Design a system with nested state for:
1. Multi-team project management
2. Hierarchical document analysis
3. Distributed data processing

---

## Lesson 2.4: Message Trimming & Filtering (45 minutes)

### Learning Objectives
- Manage conversation memory size
- Implement windowing strategies
- Filter irrelevant messages

### Key Concepts

**The Memory Problem:**

Conversations grow indefinitely:
```
Messages after 10 exchanges: ~2000 tokens
Messages after 100 exchanges: ~20,000 tokens → EXPENSIVE
```

Solutions:
- Keep only recent N messages (windowing)
- Summarize old messages
- Filter less relevant messages
- Use external storage

### Hands-On Activity: Message Management

**Pattern 1: Keep Last N Messages**

```python
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import add_messages

def trim_messages(state: dict, msgs: list, max_tokens: int = 2000):
    """Keep messages until reaching token limit"""
    total_tokens = 0
    keep_msgs = []
    
    # Iterate from newest to oldest
    for msg in reversed(msgs):
        tokens = len(msg.content.split())  # Rough estimate
        if total_tokens + tokens > max_tokens:
            break
        keep_msgs.append(msg)
        total_tokens += tokens
    
    return list(reversed(keep_msgs))

class TrimmedState(TypedDict):
    messages: Annotated[list, add_messages]

def trim_node(state: TrimmedState):
    state["messages"] = trim_messages(state, state["messages"])
    return state
```

**Pattern 2: Message Filtering**

```python
def filter_messages(messages: list, exclude_types: list[str]) -> list:
    """Remove certain message types"""
    return [
        msg for msg in messages 
        if msg.get("type") not in exclude_types
    ]

# Remove debug messages
important_msgs = filter_messages(messages, ["debug", "internal"])
```

**Pattern 3: Summary Window**

```python
class SummaryState(TypedDict):
    recent_messages: list  # Last N messages (full detail)
    old_summary: str       # Summary of earlier conversation
    summary_cutoff: int    # After how many messages summarize?

def check_need_summary(state: SummaryState):
    if len(state["recent_messages"]) > state["summary_cutoff"]:
        return "summarize"
    return "continue"

def summarize_old_messages(state: SummaryState):
    """Create summary of old messages"""
    old_text = "\n".join([m["content"] for m in state["recent_messages"][:-10]])
    # Generate summary (could use LLM)
    state["old_summary"] = f"Earlier discussion covered: {old_text[:500]}..."
    state["recent_messages"] = state["recent_messages"][-10:]
    return state
```

### Project Assignment
Build a memory-efficient chatbot that:
- Keeps recent messages (last 10)
- Summarizes older messages
- Tracks conversation metadata
- Never exceeds token limit

---

## Lesson 2.5: Chatbot with Summarization (60 minutes)

### Learning Objectives
- Build production chatbots with memory management
- Implement conversation summarization
- Handle long-running conversations

### Key Concepts

**Summarization Architecture:**
```
[Long conversation] → [Summarizer] → [Summary + Recent] → [LLM sees both]
```

### Hands-On Activity: Building a Smart Chatbot

```python
from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import add_messages, BaseMessage

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    conversation_summary: str
    message_count: int

def chat_node(state: ChatbotState) -> ChatbotState:
    """Main chat node"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Build context: summary + recent messages
    context_messages = []
    if state["conversation_summary"]:
        context_messages.append({
            "role": "system",
            "content": f"Summary of earlier conversation: {state['conversation_summary']}"
        })
    
    # Add recent messages (last 5)
    context_messages.extend(state["messages"][-5:])
    
    # Get response
    response = llm.invoke(context_messages)
    
    # Track count
    state["message_count"] += 1
    
    return {"messages": [response]}

def should_summarize(state: ChatbotState):
    """Decide if we need summarization"""
    if state["message_count"] > 10 and len(state["messages"]) > 15:
        return "summarize"
    return "chat"

def summarize_node(state: ChatbotState) -> ChatbotState:
    """Create summary of conversation"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Summarize all but last 5 messages
    old_messages = state["messages"][:-5]
    text_to_summarize = "\n".join([m.content for m in old_messages])
    
    summary = llm.invoke([{
        "role": "user",
        "content": f"Summarize this conversation:\n{text_to_summarize}"
    }])
    
    return {
        "conversation_summary": summary.content,
        "messages": state["messages"][-5:]  # Keep only recent
    }

# Build graph
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ChatbotState)
graph.add_node("chat", chat_node)
graph.add_node("summarize", summarize_node)

graph.add_edge(START, "chat")

def route_after_chat(state: ChatbotState):
    return should_summarize(state)

graph.add_conditional_edges("chat", route_after_chat, {
    "chat": "chat",
    "summarize": "summarize"
})

graph.add_edge("summarize", "chat")  # Continue chatting after summarizing

app = graph.compile()

# Use it
initial_state = {
    "messages": [{"role": "user", "content": "Hello!"}],
    "conversation_summary": "",
    "message_count": 0
}

for i in range(20):
    result = app.invoke(initial_state)
    initial_state = result
    print(f"Exchange {i+1}: Messages count={len(result['messages'])}")
```

### Real-World Considerations
- **Token budgeting:** Calculate actual token costs
- **Summary quality:** Test that summaries preserve key information
- **Latency:** Summarization adds overhead
- **Accuracy:** LLM summaries can lose nuance

### Project Assignment
Build a customer support chatbot that:
- Maintains conversation memory
- Summarizes after N exchanges
- Includes ticket metadata
- Handles escalation to human agent

---

## Lesson 2.6: External Memory Integration (60 minutes)

### Learning Objectives
- Persist state to databases
- Integrate vector stores for semantic search
- Build agents with long-term memory

### Key Concepts

**Memory Architecture:**
```
[Short-term: Current messages in RAM]
                    ↓
[Long-term: Database/Vector Store]
```

### Hands-On Activity: Database-Backed Memory

**Pattern 1: Postgres for Structured Data**

```python
import psycopg2
from typing_extensions import TypedDict

class ConversationWithDB(TypedDict):
    conversation_id: str
    messages: list
    user_profile: dict

def save_to_db(state: ConversationWithDB):
    """Save state to Postgres"""
    conn = psycopg2.connect("dbname=chatdb")
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO conversations (id, messages, profile)
        VALUES (%s, %s, %s)
        ON CONFLICT(id) DO UPDATE SET messages=EXCLUDED.messages
    """, (
        state["conversation_id"],
        str(state["messages"]),
        str(state["user_profile"])
    ))
    
    conn.commit()
    conn.close()
    return state

def load_from_db(conversation_id: str) -> ConversationWithDB:
    """Load state from Postgres"""
    conn = psycopg2.connect("dbname=chatdb")
    cur = conn.cursor()
    
    cur.execute("SELECT messages, profile FROM conversations WHERE id=%s", 
                (conversation_id,))
    messages, profile = cur.fetchone()
    conn.close()
    
    return {
        "conversation_id": conversation_id,
        "messages": eval(messages),
        "user_profile": eval(profile)
    }
```

**Pattern 2: Vector Store for Semantic Search**

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class SemanticMemoryState(TypedDict):
    query: str
    relevant_memories: list[str]
    vector_store: Chroma

def retrieve_memories(state: SemanticMemoryState) -> SemanticMemoryState:
    """Find semantically similar past interactions"""
    results = state["vector_store"].similarity_search(
        state["query"],
        k=3  # Get top 3 similar memories
    )
    
    state["relevant_memories"] = [r.page_content for r in results]
    return state

# Setup
embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings, collection_name="memories")

# Usage
state = {
    "query": "How do I reset my password?",
    "relevant_memories": [],
    "vector_store": vector_store
}

# Find similar past interactions
result = retrieve_memories(state)
print(result["relevant_memories"])  # [Similar past questions]
```

**Pattern 3: LangGraph Store (Native)**

```python
from langgraph.store import InMemoryStore

# LangGraph's built-in memory store
store = InMemoryStore()

class StateWithStore(TypedDict):
    user_id: str
    query: str
    user_memories: list[dict]

def save_memory(state: StateWithStore, store: InMemoryStore):
    """Save to LangGraph store"""
    store.put(
        ("memories", state["user_id"]),
        state["query"],  # key
        {"content": state["query"], "timestamp": "2024-01-01"}
    )
    return state

def recall_memory(state: StateWithStore, store: InMemoryStore):
    """Retrieve from LangGraph store"""
    memories = store.get_all(("memories", state["user_id"]))
    state["user_memories"] = memories
    return state
```

### Project Assignment
Build an intelligent assistant that:
- Stores user interactions in a database
- Retrieves similar past interactions using vector search
- Uses memories to personalize responses
- Maintains both short-term (messages) and long-term (database) memory

---

## Module 2 Assessment

### Knowledge Check
1. What is a reducer function and why is it useful?
2. How would you trim a conversation to stay under a token limit?
3. Design a state schema for a multi-step workflow
4. When would you use `add_messages` vs. a custom reducer?
5. How do you integrate an external database?

### Project: Advanced Chatbot System

**Requirements:**
- State schema with 5+ fields
- At least 2 different reducer types
- Message trimming logic
- Summarization node
- Optional: External memory integration

**Rubric:**
- State Design (25%): Clear, well-organized, properly typed
- Reducers (25%): Correct merging behavior, handles edge cases
- Memory Management (25%): Efficiently trims/summarizes
- Integration (25%): External memory working correctly

### Success Criteria
- ✓ Chatbot handles 50+ message exchanges
- ✓ Memory stays under token limit
- ✓ Summaries preserve key information
- ✓ State properly persists across invocations

---

## Key Takeaways

1. **State schema is the contract:** Every node must understand and respect it
2. **Reducers enable accumulation:** Use them for messages, counts, and complex merging
3. **Memory management is critical:** Long conversations need windowing/summarization
4. **External storage scales:** Use databases for true persistence
5. **Composition patterns help:** Nest state for complex systems

---

## Next Steps

Module 3 builds on state management by adding **persistence, streaming, and human oversight**—turning your stateful graphs into truly production-ready systems.

---

## Resources

- [TypedDict Documentation](https://docs.python.org/3/library/typing.html#typing.TypedDict)
- [Operator Module Docs](https://docs.python.org/3/library/operator.html)
- [LangGraph Message Management](https://langchain-ai.github.io/langgraph/concepts/messages/)
- [Vector Store Integration](https://python.langchain.com/docs/integrations/vectorstores/)