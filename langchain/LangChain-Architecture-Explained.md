# LangChain: Complete Architecture, Components & Guide
## With Visual Architecture Diagram & Detailed Explanations

## Table of Contents
1. [Introduction](#introduction)
2. [LangChain Architecture Diagram](#langchain-architecture-diagram)
3. [Why LangChain?](#why-langchain)
4. [Core Components Overview](#core-components-overview)
5. [Component Deep Dive](#component-deep-dive)
6. [Architecture Explanation](#architecture-explanation)
7. [Data Flow & Workflows](#data-flow--workflows)
8. [LangChain Ecosystem](#langchain-ecosystem)
9. [When to Use LangChain](#when-to-use-langchain)
10. [Getting Started](#getting-started)
11. [Code Examples](#code-examples)

---

## Introduction

**LangChain** is an open-source Python framework for building applications powered by Large Language Models (LLMs). It provides modular abstractions and tools that simplify the process of integrating language models with external data, APIs, memory systems, and autonomous agents.

### What Makes LangChain Special?

- **Modular Design** – Pick and use only the components you need
- **Provider Agnostic** – Works with OpenAI, Anthropic, Google, local models, etc.
- **Production Ready** – Built-in patterns for memory, retrieval, agents, and deployment
- **Highly Composable** – Combine components into complex workflows
- **Active Ecosystem** – Rich integrations with 100+ services and tools

---

## LangChain Architecture Diagram

### Visual Overview

Below is the comprehensive LangChain architecture showing all 7 core components, their relationships, and integration layers:

**[See Professional Architecture Diagram - File: LangChain-Full-Architecture.png]**

*(Note: Insert the architecture image here when viewing in a visual markdown renderer)*

---

## Architecture Explanation

### Understanding the LangChain Architecture Diagram

The architecture diagram illustrates LangChain's structure as a **three-layer system**. Let's break down what each layer represents and how the components interact:

### Layer 1: Application Layer (Top)

**Position:** The topmost layer of the architecture

**Purpose:** This is where your actual applications live - the user-facing code that end-users interact with.

**Components:**
- **Chat UIs** – User interfaces for conversational applications
- **Backend APIs** – Server-side logic for web applications
- **Workflows** – Automated business processes
- **Services** – Microservices that use LangChain

**Example:** If you're building a customer support chatbot, the chat interface would be in this layer.

**Key Point:** This layer doesn't directly interact with LLMs. Instead, it calls the LangChain framework below.

### Layer 2: LangChain Orchestration Layer (Middle)

**Position:** The core of the system - where all the magic happens

**Purpose:** Manages the coordination between LLMs, data, tools, and your application logic

**The 7 Core Components:**

#### **Component 1: Models/LLMs**
```
Position: Top-left of orchestration layer
Purpose: Unified interface to language models
```

**What it does:**
- Abstracts different LLM providers (OpenAI, Anthropic, Google, etc.)
- Provides a consistent interface regardless of which LLM you use
- Handles model configuration (temperature, token limits, etc.)

**Examples:**
- ChatOpenAI – GPT-4, GPT-3.5-turbo
- ChatAnthropic – Claude family models
- ChatGoogleGenerativeAI – Gemini models
- ChatOllama – Local open-source models

**Data Flow:** Receives formatted prompts from the Chains component, sends back generated responses.

#### **Component 2: Prompts & Templates**
```
Position: Top-middle of orchestration layer
Purpose: Input formatting and reusability
```

**What it does:**
- Standardizes how you structure prompts
- Makes prompts reusable across different inputs
- Supports template variables for dynamic content
- Includes chat message formatting

**Types:**
- **PromptTemplate** – Simple text templates
- **ChatPromptTemplate** – Multi-message conversation templates
- **Partial Templates** – Pre-fill some variables

**Data Flow:** Takes input variables → formats them into structured prompts → sends to Chains/LLMs.

#### **Component 3: Chains**
```
Position: Top-right of orchestration layer
Purpose: Multi-step workflows
```

**What it does:**
- Orchestrates multiple components into workflows
- Passes outputs from one step as inputs to the next
- Manages the execution flow

**Types:**
- **LLMChain** – One LLM call with a prompt
- **SequentialChain** – Multiple steps in order
- **ConversationChain** – Stateful conversations with memory
- **RAGChain** – Retrieval + LLM generation

**Data Flow:** Receives user input → manages flow through templates, memory, retrievers → sends to LLM → returns final output.

#### **Component 4: Memory**
```
Position: Middle-left of orchestration layer
Purpose: Conversation state persistence
```

**What it does:**
- Stores conversation history
- Injects past context into new prompts
- Manages different memory strategies

**Types:**
- **ConversationBufferMemory** – Stores all messages
- **ConversationSummaryMemory** – Summarizes conversations
- **ConversationTokenBufferMemory** – Token-aware buffer
- **VectorStoreMemory** – Semantic similarity search
- **ConversationEntityMemory** – Tracks important entities

**Data Flow:** 
1. New message comes in → Memory retrieves past context
2. Past context is injected into the prompt
3. After LLM response → New exchange is saved to memory

**Key Point:** Without memory, each conversation would be independent. Memory enables the model to understand context across multiple turns.

#### **Component 5: Retrievers & Indexes (RAG)**
```
Position: Middle-center of orchestration layer
Purpose: Enable Retrieval-Augmented Generation
```

**What it does:**
- Loads external documents
- Creates searchable indexes (vector databases)
- Retrieves relevant documents when user asks questions
- Passes retrieved context to LLM

**RAG Pipeline:**
1. **Load** documents from files, PDFs, web pages
2. **Split** documents into manageable chunks
3. **Embed** chunks into numerical vectors
4. **Store** vectors in a vector database
5. **Retrieve** relevant chunks when user queries

**Supported Vector Stores:**
- Chroma – Lightweight, great for development
- Pinecone – Managed, production-ready
- Weaviate – Open-source, flexible
- FAISS – Meta's high-performance framework
- Milvus – Scalable vector database

**Data Flow:** User query → embedded as vector → searched in vector database → top-k relevant documents retrieved → added to LLM prompt → LLM generates grounded answer.

**Key Point:** This solves the "knowledge cutoff" problem. Instead of relying on training data, RAG lets LLMs answer questions about your specific documents.

#### **Component 6: Tools**
```
Position: Middle-right of orchestration layer
Purpose: Give agents external capabilities
```

**What it does:**
- Defines functions or APIs that agents can call
- Acts as an interface between LLM and external systems
- Provides structured way to pass information to agents

**Tool Types:**
- **HTTP APIs** – Web search, weather, news
- **Database Queries** – Read/write to databases
- **Python Functions** – Custom calculations
- **System Operations** – File I/O, shell commands
- **Custom APIs** – Your own business logic

**Examples:**
- Search tool (Google, DuckDuckGo)
- Calculator tool
- Database lookup tool
- Weather API tool

**Data Flow:** Agent decides → decides which tool to use → calls tool with parameters → receives tool output → incorporates into next reasoning step.

#### **Component 7: Agents**
```
Position: Bottom of orchestration layer
Purpose: Autonomous decision-making
```

**What it does:**
- Uses LLM to decide which action to take
- Calls tools based on reasoning
- Iterates through think-act-observe cycles
- Autonomously completes multi-step tasks

**Agent Types:**
- **ReAct (Reasoning + Acting)** – Think step-by-step, then act
- **Tool-using Agents** – Choose which tool to call
- **Planning Agents** – Multi-step planning
- **Conversational Agents** – Stateful with memory

**Agent Workflow:**
```
Input (User Task)
    ↓
Think (LLM decides what to do)
    ↓
Act (Call appropriate tool)
    ↓
Observe (Get tool results)
    ↓
Repeat until task complete
    ↓
Final Answer
```

**Data Flow:** User task → Agent LLM reasoning → selects tool → tool execution → results back to agent → repeat until done.

**Key Point:** Agents are autonomous. Unlike chains where you define the steps, agents use LLM reasoning to decide the next action dynamically.

### Layer 3: Integration Layer (Bottom)

**Position:** The bottommost layer where external services live

**Purpose:** Connects LangChain to external systems and services

**Categories:**

#### **LLM Providers (20+)**
```
OpenAI (GPT-4, GPT-3.5)
Anthropic (Claude)
Google (Gemini, PaLM)
Cohere
Mistral
Meta (Llama)
Local models (Ollama, Hugging Face)
...and more
```

**Purpose:** Provides the actual language models that power LangChain applications.

#### **Vector Stores (15+)**
```
Pinecone (Managed)
Chroma (Open-source, lightweight)
Weaviate (Open-source, flexible)
FAISS (Meta's framework)
Milvus (High-performance)
Cassandra (Distributed)
...and more
```

**Purpose:** Stores embeddings and enables semantic search for RAG systems.

#### **Data Sources (30+)**
```
PDF documents
Text files
Word documents
Web pages (URLs)
CSV files
SQL databases
NoSQL databases
Google Drive
Notion
GitHub
...and more
```

**Purpose:** Provides documents and data that RAG systems retrieve from.

#### **APIs & External Services**
```
Web search (Google, DuckDuckGo)
Weather services
News APIs
Calculation engines
Database services
Custom business APIs
...and more
```

**Purpose:** Gives agents the ability to perform actions outside of text generation.

---

## Component Interaction Flow

### How Data Flows Through the Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────┐
│ Application Layer receives input │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Memory Module (if enabled)       │
│ Retrieves conversation history   │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Retrievers/RAG (if enabled)      │
│ Fetches relevant documents       │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Prompts & Templates              │
│ Formats input with context       │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Models/LLMs                      │
│ Generates response               │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Tools & Agents (if needed)       │
│ Takes additional actions         │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Memory Storage (if enabled)      │
│ Saves new interaction            │
└────────────┬─────────────────────┘
             │
             ▼
Response to User
```

---

## Why LangChain?

### Problems It Solves

#### Problem 1: Stateless Interactions
```
Raw API Call:
User: "My name is Alice"
LLM: "Nice to meet you!"

Next Call:
User: "What's my name?"
LLM: "I don't know" ❌
```

LLM APIs don't remember conversations. Each call starts fresh.

**LangChain Solution:** Memory component automatically stores and retrieves conversation history.

#### Problem 2: No Data Access
Raw LLMs cannot:
- Read your databases
- Access PDF documents
- Retrieve from knowledge bases
- Integrate with company systems

**LangChain Solution:** Retrievers & RAG components integrate seamlessly with vector databases.

#### Problem 3: No Action Capability
```python
# LLM can only generate text:
response = llm.generate("Send an email")
# Output: "To send an email, use: send_email(recipient, message)"
# But LLM can't actually send it! ❌
```

**LangChain Solution:** Tools and Agents enable LLMs to call external APIs and functions.

#### Problem 4: Complex Boilerplate
Without LangChain, you manually handle:
- Prompt formatting
- Conversation history tracking
- Document retrieval pipelines
- Output parsing and validation
- Error handling and retries

**LangChain Solution:** All components provide abstraction over common patterns.

### How LangChain Solves These

| Problem | Solution |
|---------|----------|
| No memory | **Memory module** – automatically stores & injects conversation history |
| No data access | **Retrieval & RAG** – document loading, embeddings, vector stores |
| No action capability | **Tools & Agents** – define APIs, let agents decide when to call them |
| Complex boilerplate | **Chains & Components** – reusable, composable building blocks |

---

## Core Components Overview

LangChain consists of **7 major components**:

```
┌─────────────────────────────────────────────────────┐
│         LangChain Core Components (7)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Models/LLMs          → LLM interface layer     │
│  2. Prompts/Templates    → Input formatting        │
│  3. Chains               → Multi-step workflows    │
│  4. Memory               → Conversation state      │
│  5. Retrievers/Indexes   → Data search & RAG      │
│  6. Tools                → APIs & functions        │
│  7. Agents               → Autonomous decision     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. Models / LLMs

**Purpose:** Unified interface to different language models.

**What it provides:**
- Chat models (ChatGPT, Claude, Gemini)
- Completion models
- Embedding models
- Unified method signatures across providers

**Key Classes:**
- `ChatOpenAI` – OpenAI's GPT models
- `ChatAnthropic` – Anthropic's Claude
- `ChatGoogleGenerativeAI` – Google's Gemini
- `ChatOllama` – Local/open-source models

**Basic Code:**
```python
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic

# OpenAI
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key="your_key"
)

# Anthropic
claude = ChatAnthropic(
    model="claude-3-opus-20240229",
    max_tokens=1024
)

# Get response
response = llm.invoke("What is Python?")
print(response.content)
```

**Key Parameters:**
- `model` – Model identifier
- `temperature` – Randomness (0=deterministic, 1=creative)
- `max_tokens` – Maximum response length
- `top_p`, `top_k` – Sampling parameters

---

### 2. Prompts & Templates

**Purpose:** Standardize and reuse prompt formatting.

**What it provides:**
- PromptTemplate – For simple text prompts
- ChatPromptTemplate – For multi-turn conversations
- Dynamic variable injection
- Partial functions
- Prompt composition

**Basic Code:**
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Simple prompt template
simple_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms suitable for a 10-year-old."
)

formatted = simple_prompt.format(topic="Machine Learning")
print(formatted)

# Chat prompt template (multi-turn)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Python tutor."),
    ("human", "What is a {concept}?"),
])

messages = chat_prompt.format_messages(concept="decorator")
print(messages)
```

**Best Practices:**
- Use templates for consistency
- Include role/persona for better outputs
- Add context and examples
- Define output format clearly

---

### 3. Chains

**Purpose:** Combine multiple components into multi-step workflows.

**What it provides:**
- **LLMChain** – Prompt + LLM in one call
- **SequentialChain** – Run chains in sequence
- **ConversationChain** – Stateful conversations with memory
- **RAGChain** – Retrieval + generation
- **Custom chains** – Build your own

**Chain Types & Code:**

#### 3a. LLMChain (Single Step)
```python
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = PromptTemplate(
    template="Write a {style} poem about {topic}",
    input_variables=["style", "topic"]
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(style="haiku", topic="Python")
print(result)
```

#### 3b. SequentialChain (Multiple Steps)
```python
from langchain.chains import SequentialChain, LLMChain

# Step 1: Generate outline
outline_prompt = PromptTemplate(
    template="Create an outline for an article about {topic}",
    input_variables=["topic"]
)
outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")

# Step 2: Write article
article_prompt = PromptTemplate(
    template="Write a detailed article based on:\n{outline}",
    input_variables=["outline"]
)
article_chain = LLMChain(llm=llm, prompt=article_prompt, output_key="article")

# Combine into sequential chain
overall = SequentialChain(
    chains=[outline_chain, article_chain],
    input_variables=["topic"],
    output_variables=["outline", "article"]
)

result = overall({"topic": "Artificial Intelligence"})
print(result["article"])
```

#### 3c. ConversationChain (With Memory)
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
conversation.run("Hi! My name is Alice")
conversation.run("What programming language should I learn?")
conversation.run("You mentioned Python earlier, right?")
# Model remembers due to memory!
```

---

### 4. Memory

**Purpose:** Persist and manage conversation state across interactions.

**Memory Types:**

| Type | Use Case | Code |
|------|----------|------|
| **BufferMemory** | Short conversations | Store all messages verbatim |
| **SummaryMemory** | Long conversations | Summarize to save tokens |
| **TokenBufferMemory** | Token-aware | Keep below token limit |
| **VectorMemory** | Semantic search | Find similar past interactions |
| **EntityMemory** | Fact tracking | Remember entities (names, dates) |

**Code Examples:**

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory
)

# 1. Buffer Memory (Simple)
memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Hi, my name is Bob"},
    {"output": "Nice to meet you Bob!"}
)
print(memory.buffer)

# 2. Summary Memory (Efficient for long conversations)
summary_memory = ConversationSummaryMemory(llm=llm)
summary_memory.save_context(
    {"input": "I love Python and machine learning"},
    {"output": "That's great! Those are powerful skills."}
)

# 3. Token Buffer Memory (Token-aware)
token_memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=1000  # Keep under 1000 tokens
)
```

**Memory Best Practices:**
- Use BufferMemory for simple chatbots
- Use SummaryMemory for long conversations
- Use TokenBufferMemory to control costs
- Clear memory when starting new topics

---

### 5. Retrievers & Indexes (RAG)

**Purpose:** Enable LLMs to answer questions about external documents.

**RAG Pipeline:**
1. Load documents
2. Split into chunks
3. Create embeddings
4. Store in vector database
5. Retrieve relevant chunks on query
6. Pass to LLM with context

**Code Example:**

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Step 1: Load documents
loader = TextLoader("company_docs.txt")
documents = loader.load()

# Step 2: Split into chunks
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 documents
)

# Step 5: Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 6: Ask questions
result = qa_chain("What is our company's return policy?")
print(result["result"])
print(result["source_documents"])
```

**Vector Store Options:**
- **Chroma** – Lightweight, open-source, great for development
- **Pinecone** – Managed, production-ready
- **Weaviate** – Open-source, flexible
- **FAISS** – Meta's framework, fast similarity search
- **Milvus** – High-performance, scalable

---

### 6. Tools

**Purpose:** Give agents the ability to call external APIs and functions.

**Types of Tools:**
- HTTP APIs (Web search, weather)
- Database queries
- Mathematical calculations
- Python code execution
- Custom functions

**Code Example:**

```python
from langchain.tools import Tool, tool
import requests

# Method 1: Using @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city"""
    response = requests.get(f"https://api.weather.com/city/{city}")
    return response.json()["temp"]

# Method 2: Using Tool class
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Use this to perform mathematical calculations. Input: math expression"
)

# Method 3: Structured tool with validation
from langchain.tools import StructuredTool

def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information"""
    # Implementation
    return f"Results for: {query}"

search_tool = StructuredTool.from_function(
    func=search_web,
    name="WebSearch",
    description="Search the internet for information",
)

# Use tools
tools = [get_weather, calculator_tool, search_tool]
```

**Tool Best Practices:**
- Write clear descriptions for agent understanding
- Include type hints and docstrings
- Handle errors gracefully
- Return results in string format
- Test tools independently first

---

### 7. Agents

**Purpose:** Let LLMs autonomously decide which tools to use.

**Agent Types:**
- **ReAct** – Reasoning + Acting (most popular)
- **Tool-using** – Simple tool selection
- **Planner** – Multi-step planning
- **Conversational** – Stateful with memory

**Code Example:**

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchRun

# Define tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Search the internet"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Evaluate math expressions"
    )
]

# Create agent
agent = initialize_agent(
    tools,
    ChatOpenAI(model="gpt-4"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5
)

# Run agent
result = agent.run("Search for latest AI news and summarize")
print(result)
```

**Agent Workflow (ReAct):**
```
Thought: I need to find recent AI news
Action: Search
Observation: Found articles about GPT-4
Thought: Let me summarize what I found
Final Answer: [Summary]
```

---

## Data Flow & Workflows

### Workflow 1: Simple Q&A Chain

```
Question
   │
   ▼
┌─────────────────────┐
│ Create Prompt       │ "Answer this: {question}"
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Send to LLM         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Get Response        │
└──────────┬──────────┘
           │
           ▼
       Answer
```

**Code:**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

chain = LLMChain(llm=llm, prompt=PromptTemplate(...))
result = chain.run(question="What is Python?")
```

### Workflow 2: RAG (Retrieval-Augmented Generation)

```
Question
   │
   ▼
┌──────────────────────┐
│ Embed Question       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Search Vector Store  │ Find similar documents
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Get Top Documents    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Build Prompt         │ "Based on: [docs]\nAnswer: {question}"
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Send to LLM          │
└──────────┬───────────┘
           │
           ▼
   Answer (grounded in documents)
```

---

### Workflow 3: Agent (Autonomous Decision-Making)

```
User Task: "Research AI trends and write summary"
   │
   ▼
┌──────────────────────────────────┐
│ Agent Thinks                     │
│ "I should search the web first"  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Agent Takes Action               │
│ Calls: search_tool("AI trends")  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Agent Observes Results           │
│ Gets: News articles, data        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Agent Thinks Again               │
│ "I have info, now write summary" │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Final Answer: Summary            │
└──────────────────────────────────┘
```

---

## LangChain Ecosystem

### Official Tools & Services

| Tool | Purpose | Link |
|------|---------|------|
| **LangSmith** | Debugging, testing, monitoring chains | Observability |
| **LangServe** | Deploy chains as REST APIs | Deployment |
| **LangGraph** | Build graph-based agent workflows | Orchestration |
| **LangChain Hub** | Share and discover prompts | Community |

### Supported Providers

**LLM Providers (20+):**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini, PaLM)
- Cohere
- Mistral
- Meta (Llama)
- Local models via Ollama, Hugging Face

**Vector Stores (15+):**
- Pinecone
- Chroma
- Weaviate
- FAISS
- Milvus
- Cassandra
- And more...

**Data Loaders (30+):**
- PDF, Text, Word, CSV
- Web pages, URLs
- GitHub, Notion, Google Drive
- SQL databases
- Custom sources

---

## When to Use LangChain

### ✅ Use LangChain When

- Building **conversational AI** with memory
- Implementing **RAG systems** over documents
- Creating **autonomous agents** with tools
- Need **multi-step workflows** (chains)
- Want to **switch LLM providers** easily
- Building for **production** (need patterns for reliability)
- Need **observability** (LangSmith integration)

### ❌ Don't Use LangChain When

- Making a **one-off API call** – use raw OpenAI/Anthropic SDK
- Doing **non-LLM ML** – use scikit-learn, PyTorch
- Need **tight research-level control** of sampling
- Just **training models** – use ML frameworks
- Building **lightweight edge apps** where framework overhead matters

---

## Getting Started

### Step 1: Install

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### Step 2: Set Up API Key

Create `.env` file:
```
OPENAI_API_KEY=sk-...
```

Load it:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### Step 3: Choose Your Path

**Path A: Simple Q&A**
```python
from langchain.chains import LLMChain
# See: Component Deep Dive → Chains section
```

**Path B: RAG over Documents**
```python
from langchain.chains import RetrievalQA
# See: Component Deep Dive → Retrievers section
```

**Path C: Autonomous Agent**
```python
from langchain.agents import initialize_agent
# See: Component Deep Dive → Agents section
```

**Path D: Conversational AI**
```python
from langchain.chains import ConversationChain
# See: Component Deep Dive → Chains section
```

---

## Code Examples

### Example 1: Simple Chatbot

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant."),
    ("human", "{input}")
])

chain = LLMChain(llm=llm, prompt=prompt)

# Chat
print(chain.run(input="How do I read a CSV in Python?"))
```

### Example 2: RAG System

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load documents
loader = TextLoader("company_docs.txt")
docs = loader.load()

# Split
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embeddings & Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
print(qa.run("What is our return policy?"))
```

### Example 3: Conversational Memory

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Multi-turn
conversation.run("My name is Bob and I love Python")
conversation.run("What programming language do I like?")
conversation.run("What's my name?")
# All remembered via memory!
```

### Example 4: Simple Agent

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Search the internet for current information"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Use this for math calculations"
    )
]

agent = initialize_agent(
    tools,
    ChatOpenAI(model="gpt-4"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run
result = agent.run("What's 25 * 4? Also search for latest Python version")
print(result)
```

### Example 5: Multi-Step Chain (Sequential)

```python
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# Step 1: Generate outline
outline_prompt = PromptTemplate(
    template="Create an article outline about {topic}",
    input_variables=["topic"]
)
outline_chain = LLMChain(
    llm=llm,
    prompt=outline_prompt,
    output_key="outline"
)

# Step 2: Write article
article_prompt = PromptTemplate(
    template="Write a detailed article based on this outline:\n{outline}",
    input_variables=["outline"]
)
article_chain = LLMChain(
    llm=llm,
    prompt=article_prompt,
    output_key="article"
)

# Step 3: Summarize
summary_prompt = PromptTemplate(
    template="Create a 2-sentence summary of:\n{article}",
    input_variables=["article"]
)
summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    output_key="summary"
)

# Combine
overall_chain = SequentialChain(
    chains=[outline_chain, article_chain, summary_chain],
    input_variables=["topic"],
    output_variables=["outline", "article", "summary"]
)

# Run
result = overall_chain({"topic": "Climate Change"})
print("Outline:", result["outline"])
print("Article:", result["article"])
print("Summary:", result["summary"])
```

### Example 6: Custom Tool for Agent

```python
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a company symbol"""
    # Mock implementation
    prices = {"AAPL": 150.25, "MSFT": 380.50, "GOOGL": 140.75}
    return f"Stock {symbol}: ${prices.get(symbol, 'Not found')}"

@tool
def calculate_returns(initial: float, final: float) -> str:
    """Calculate investment returns percentage"""
    returns = ((final - initial) / initial) * 100
    return f"Return: {returns:.2f}%"

tools = [get_stock_price, calculate_returns]

agent = initialize_agent(
    tools,
    ChatOpenAI(model="gpt-4"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Check Apple stock price and calculate if I invested $100 and it's now worth $150"
)
print(result)
```

### Example 7: RAG + Conversation (Advanced)

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Setup RAG
loader = TextLoader("docs.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Setup memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create chain
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Multi-turn conversation over documents
print(qa({"question": "What is the main topic?"}))
print(qa({"question": "Tell me more about that"}))  # References previous context!
print(qa({"question": "What did I ask first?"}))    # Remembers!
```

---

## Best Practices & Tips

### 1. Prompt Engineering
- Use clear, specific instructions
- Include examples (few-shot learning)
- Define output format explicitly
- Use system prompts for personality/role

### 2. Memory Management
- Start with BufferMemory for simplicity
- Switch to SummaryMemory for long conversations
- Use TokenBufferMemory to control costs
- Clear memory when changing topics

### 3. RAG Systems
- Chunk documents appropriately (500-1500 tokens)
- Use 10-20% overlap between chunks
- Test retrieval quality
- Use hybrid search when possible
- Monitor relevance scores

### 4. Agent Design
- Start with fewer tools (2-3)
- Write clear tool descriptions
- Test tools independently
- Set max_iterations to prevent infinite loops
- Use verbose=True for debugging

### 5. Error Handling
```python
try:
    result = chain.run(input)
except Exception as e:
    print(f"Error: {e}")
    # Fallback or retry logic
```

### 6. Cost Optimization
- Use token counting to estimate costs
- Batch requests when possible
- Use cheaper models for non-critical tasks
- Implement caching for repeated queries
- Monitor API usage

---

## Troubleshooting

### Issue: "API key not found"
```python
# Solution: Set environment variable
import os
os.environ["OPENAI_API_KEY"] = "your_key"
```

### Issue: "Token limit exceeded"
```python
# Solution: Use TokenBufferMemory
from langchain.memory import ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=1000)
```

### Issue: "Poor retrieval quality"
```python
# Solution: Improve chunking
splitter = CharacterTextSplitter(
    chunk_size=500,     # Smaller chunks
    chunk_overlap=100   # More overlap
)
```

### Issue: "Agent takes too long"
```python
# Solution: Limit iterations
agent = initialize_agent(
    tools, llm,
    max_iterations=3,   # Stop after 3 steps
    early_stopping_method="generate"
)
```

---

## Next Steps

1. **Complete the 50-hour training program** starting with Module 1
2. **Run all code examples** in this document
3. **Build a small project** (chatbot, RAG system, or agent)
4. **Join the LangChain community** on Discord
5. **Stay updated** with LangChain releases

---

**This is your complete guide to LangChain components and architecture. Download, save, and reference as needed!**

Version: 1.0  
Last Updated: December 2025  
Status: Production Ready
