# LangChain: Complete Architecture, Components & Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Why LangChain?](#why-langchain)
3. [Core Components Overview](#core-components-overview)
4. [Component Deep Dive](#component-deep-dive)
5. [Architecture Diagram](#architecture-diagram)
6. [Data Flow & Workflows](#data-flow--workflows)
7. [LangChain Ecosystem](#langchain-ecosystem)
8. [When to Use LangChain](#when-to-use-langchain)
9. [Getting Started](#getting-started)
10. [Code Examples](#code-examples)

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

#### Problem 2: No Data Access
Raw LLMs cannot:
- Read your databases
- Access PDF documents
- Retrieve from knowledge bases
- Integrate with company systems

#### Problem 3: No Action Capability
```python
# LLM can only generate text:
response = llm.generate("Send an email")
# Output: "To send an email, use: send_email(recipient, message)"
# But LLM can't actually send it! ❌
```

#### Problem 4: Complex Boilerplate
Without LangChain, you manually handle:
- Prompt formatting
- Conversation history tracking
- Document retrieval pipelines
- Output parsing and validation
- Error handling and retries

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
from langchain.schema import HumanMessage, SystemMessage

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

# Template with multiple variables
template = """You are an expert {role}.
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["role", "context", "question"]
)
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
    ConversationTokenBufferMemory,
    VectorStoreMemory,
    ConversationEntityMemory
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

# 4. Vector Memory (Semantic search)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
vector_memory = VectorStoreMemory(
    vectorstore=vectorstore,
    embedding_function=embeddings
)

# 5. Entity Memory (Track important facts)
entity_memory = ConversationEntityMemory(llm=llm)
entity_memory.save_context(
    {"input": "My boss is Alice and she works at TechCorp"},
    {"output": "Got it!"}
)
print(entity_memory.entity_cache["Alice"])
```

**Memory Best Practices:**
- Use BufferMemory for simple chatbots
- Use SummaryMemory for long, multi-session conversations
- Use TokenBufferMemory to stay within API limits
- Use VectorMemory for semantic relevance
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
from langchain.document_loaders import TextLoader, PDFLoader
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
from langchain.tools import DuckDuckGoSearchRun, BaseTool

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

## Architecture Diagram

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Your Application                      │
│         (Chatbot, RAG App, Agent, Workflow)            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LangChain Framework                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Models/LLMs  │  │   Prompts    │  │    Chains    │  │
│  │ • ChatOpenAI │  │ • Template   │  │ • LLMChain   │  │
│  │ • Claude     │  │ • ChatPrompt │  │ • Sequential │  │
│  │ • Gemini     │  │ • Partial    │  │ • RAG        │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    Memory    │  │  Retrievers  │  │    Tools     │  │
│  │ • Buffer     │  │ • Vector DB  │  │ • APIs       │  │
│  │ • Summary    │  │ • BM25       │  │ • Functions  │  │
│  │ • Token      │  │ • Ensemble   │  │ • Custom     │  │
│  │ • Vector     │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Agents (Autonomous Systems)            │  │
│  │   ReAct | Tool-using | Planning | Conversational │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│           Integration Layer (External Services)         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  LLM APIs     Vector Stores    Data Sources    APIs     │
│  • OpenAI     • Pinecone       • PDFs          • HTTP   │
│  • Anthropic  • Chroma         • Documents     • SQL    │
│  • Google     • Weaviate       • Web pages     • NoSQL  │
│  • Local      • FAISS          • CSVs          • Custom │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Component Interaction Map

```
┌─────────────┐
│  User Input │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Memory Module  │◄─── Load conversation history
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Retriever (RAG) │◄─── Fetch relevant documents
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Prompt Template  │◄─── Format with context
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Model/LLM      │◄─── Generate response
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Output Parser    │◄─── Structure output
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Agent/Tool Loop │◄─── (Optional) Call tools
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Memory Storage   │◄─── Save interaction
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Final Response  │
└──────────────────┘
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

**Code:**
```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)
result = qa.run("Question about documents")
```

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
│ Agent Takes Action               │
│ (No tool needed, just generate)  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Final Answer: Summary            │
└──────────────────────────────────┘
```

**Code:**
```python
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
result = agent.run("Research AI trends and write summary")
```

### Workflow 4: Conversational Agent (With Memory)

```
User: "Hi, my name is Alice"
   │
   ▼
┌──────────────────────────────────┐
│ Load Memory                      │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Chat with LLM                    │
│ "I'm Alice, nice to meet you"    │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Save to Memory                   │
│ Input: "Hi, my name is Alice"    │
│ Output: "I'm Alice, ..."         │
└──────────┬───────────────────────┘
           │
           ▼

User: "What's my name?"
   │
   ▼
┌──────────────────────────────────┐
│ Load Memory (has Alice!)         │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Chat with LLM                    │
│ "Your name is Alice"             │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Save to Memory                   │
└──────────────────────────────────┘
```

**Code:**
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

conversation.run("Hi, my name is Alice")
conversation.run("What's my name?")  # Remembers!
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
# See: Code Examples section
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
