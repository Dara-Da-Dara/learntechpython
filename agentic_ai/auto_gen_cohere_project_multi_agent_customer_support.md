# AutoGen + Cohere Project: Multi‑Agent Customer Support System

This project demonstrates how to build a **multi‑agent AI system using Microsoft AutoGen and Cohere API** for a realistic business use case: **Customer Support Automation**.

The system uses multiple agents (Manager, Support Agent, QA Agent) that collaborate to understand a customer query, generate a response, and validate its quality.

---

## 1. Business Use Case

**Problem**  
Customer support teams receive repetitive queries (refunds, order status, product issues). Human agents spend time answering common questions and checking response quality.

**Solution**  
Use a **multi‑agent AutoGen system** where:
- A **Manager Agent** decides what to do
- A **Support Agent** answers the customer
- A **QA Agent** checks correctness and tone

Cohere is used as the LLM for all agents.

---

## 2. Architecture

```
Customer Query
      │
      ▼
Manager Agent (AutoGen)
      │
      ├──► Support Agent (Cohere)
      │           │
      │           ▼
      └──► QA Agent (Cohere)
                  │
                  ▼
           Final Approved Response
```

---

## 3. Tech Stack

- Python 3.9+
- Microsoft AutoGen
- Cohere API (command‑r / command‑r‑plus)
- dotenv

---

## 4. Google Colab Setup

This project is **fully runnable on Google Colab** (no local setup needed).

### Step 1: Open Google Colab
- Go to **https://colab.research.google.com**
- Click **New Notebook**

### Step 2: Install Dependencies

```bash
!pip install -q pyautogen cohere python-dotenv
```

```bash
pip install pyautogen cohere python-dotenv
```

---

## 5. Environment Setup (Colab)

In Colab, securely set your Cohere API key using **Secrets** or environment variables.

### Option A: Using Colab Secrets (Recommended)
- Click 🔒 **Secrets** (left panel)
- Add:
  - Name: `COHERE_API_KEY`
  - Value: your Cohere API key

### Option B: Directly in Notebook (for testing only)

```python
import os
os.environ["COHERE_API_KEY"] = "your_cohere_api_key_here"
```

Create a `.env` file:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

---

## 6. Colab Notebook Structure

In Colab, everything runs inside **one notebook** (no separate files needed).

Suggested cell structure:

```
Cell 1: Install packages
Cell 2: Imports & config
Cell 3: Agent definitions
Cell 4: Run multi-agent workflow
```

```
autogen-cohere-support/
│
├── main.py
├── agents.py
├── config.py
├── .env
└── README.md
```

---

## 7. Configuration (Colab Cell)

```python
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

llm_config = {
    "model": "command-r-plus",
    "api_key": COHERE_API_KEY,
    "api_type": "cohere",
}
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

llm_config = {
    "model": "command-r-plus",
    "api_key": COHERE_API_KEY,
    "api_type": "cohere",
}
```

---

## 8. Agent Definitions (Colab Cell)

```python
from autogen import AssistantAgent, UserProxyAgent

# Manager Agent
manager_agent = AssistantAgent(
    name="ManagerAgent",
    system_message="You are a manager that decides how customer issues should be handled.",
    llm_config=llm_config,
)

# Support Agent
support_agent = AssistantAgent(
    name="SupportAgent",
    system_message="You are a helpful customer support agent. Be clear, polite, and concise.",
    llm_config=llm_config,
)

# QA Agent
qa_agent = AssistantAgent(
    name="QAAgent",
    system_message="You check responses for correctness, tone, and policy compliance.",
    llm_config=llm_config,
)

# Customer (User Proxy)
user_proxy = UserProxyAgent(
    name="Customer",
    human_input_mode="NEVER",
)
```

```python
from autogen import AssistantAgent, UserProxyAgent
from config import llm_config

# Manager Agent
manager_agent = AssistantAgent(
    name="ManagerAgent",
    system_message="You are a manager that decides how customer issues should be handled.",
    llm_config=llm_config,
)

# Support Agent
support_agent = AssistantAgent(
    name="SupportAgent",
    system_message="You are a helpful customer support agent. Be clear, polite, and concise.",
    llm_config=llm_config,
)

# QA Agent
qa_agent = AssistantAgent(
    name="QAAgent",
    system_message="You check responses for correctness, tone, and policy compliance.",
    llm_config=llm_config,
)

# User Proxy (Customer)
user_proxy = UserProxyAgent(
    name="Customer",
    human_input_mode="NEVER",
)
```

---

## 9. Run the Multi-Agent Flow (Colab Cell)

```python
# Customer query
customer_query = "I want a refund for my order. It arrived damaged."

# Start the conversation
user_proxy.initiate_chat(
    manager_agent,
    message=customer_query,
)

# Manager → Support Agent
manager_agent.send(
    "Generate a customer-friendly response to the refund request.",
    support_agent,
)

# Support → QA Agent
support_agent.send(
    "Review the response for tone and accuracy

```python
from agents import manager_agent, support_agent, qa_agent, user_proxy

# Customer query
customer_query = "I want a refund for my order. It arrived damaged."

# Step 1: Manager receives the query
user_proxy.initiate_chat(
    manager_agent,
    message=customer_query,
)

# Step 2: Manager asks Support Agent
manager_agent.send(
    "Generate a customer-friendly response to the refund request.",
    support_agent,
)

# Step 3: QA Agent reviews the response
support_agent.send(
    "Review the response for tone and accuracy.",
    qa_agent,
)

print("Multi-agent customer support flow completed.")
```

---

## 10. Expected Output

- A polite refund explanation
- Clear next steps for the customer
- QA-approved final answer

Example response:

> "We're sorry your order arrived damaged. You are eligible for a full refund. Please share your order ID, and we’ll process it within 3–5 business days."

---

## 11. Why AutoGen + Cohere?

- **AutoGen** → agent collaboration & orchestration
- **Cohere** → strong instruction-following and enterprise‑grade NLP
- Easy to extend with:
  - CRM tools
  - Order databases
  - Gradio / FastAPI UI

---

## 12. Possible Extensions

- Add tool‑calling for order lookup
- Add memory (FAISS) for past conversations
- Deploy with Gradio or FastAPI
- Add multilingual support

---

## 13. Summary

This project shows how to build a **production‑ready multi‑agent system** using AutoGen and Cohere for customer support automation. It is modular, extensible, and ideal for real business workflows.

---

If you want, I can:
- Convert this into a **downloadable Jupyter Notebook**
- Add **Gradio UI**
- Add **long‑term memory (FAISS)**
- Create a **diagram + deployment guide**

