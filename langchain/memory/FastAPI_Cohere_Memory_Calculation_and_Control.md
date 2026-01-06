
# Memory Calculation & Control in LLM Systems  
## (FastAPI + Cohere + Practical Architecture)

---

## 1. What is Memory in LLM Systems?

In Large Language Models (LLMs), **memory** refers to all textual information sent to the model **within its context window**, including:

- System prompt  
- Conversation history  
- Retrieved documents  
- User input  
- Expected model output  

> LLMs do NOT remember by default.  
> Memory must be **explicitly passed every time**.

---

## 2. Types of Memory (Conceptual View)

| Memory Type | Description | Stored Where |
|------------|------------|-------------|
| Parametric Memory | Knowledge inside model weights | Model parameters |
| Short-term Memory | Recent conversation | Prompt context |
| Summary Memory | Compressed past conversation | Prompt context |
| Long-term Memory | Facts, preferences | Vector DB / DB |
| Tool Memory | Logs, tool outputs | External storage |

---

## 3. Core Memory Calculation Formula

```
TOTAL TOKENS =
System Prompt
+ Stored Memory
+ User Input
+ Model Output
```

Constraint:

```
TOTAL TOKENS ≤ Model Context Window
```

---

## 4. Example: Token-Based Memory Calculation

Assume:
- Context window = 4096 tokens  
- System prompt = 300 tokens  
- User input = 200 tokens  
- Model output = 400 tokens  

```
Available memory =
4096 − (300 + 200 + 400)
= 3196 tokens
```

If each turn ≈ 400 tokens:

```
3196 / 400 ≈ 7 turns
```

---

## 5. Summary Memory Calculation

```
10 turns × 400 tokens = 4000 tokens
Summary size ≈ 300 tokens
Compression ≈ 13×
```

---

## 6. Vector Memory Calculation

Embedding:
- 1536 dimensions
- float32 (4 bytes)

```
1536 × 4 = 6144 bytes ≈ 6 KB per memory
10,000 memories ≈ 60 MB
```

---

## 7. When to Use Which Memory

| Requirement | Memory Type |
|------------|-------------|
| Exact conversation | Buffer memory |
| Long conversations | Summary memory |
| Permanent facts | Vector memory |
| Agent workflows | Hybrid memory |

---

## 8. FastAPI + Memory Size Control

FastAPI does **not** manage memory.  
You manage memory **before calling the LLM**.

Architecture:

```
Client → FastAPI → Token Budgeting → Memory Trim/Summary → Cohere LLM
```

---

## 9. Token Budget Configuration

```python
MAX_CONTEXT_TOKENS = 4096
SYSTEM_TOKENS = 300
RESPONSE_TOKENS = 400

AVAILABLE_MEMORY_TOKENS = (
    MAX_CONTEXT_TOKENS
    - SYSTEM_TOKENS
    - RESPONSE_TOKENS
)
```

---

## 10. Token Estimation Utility

```python
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)
```

---

## 11. Memory Trimming Logic

```python
def trim_memory(memory, max_tokens):
    total_tokens = 0
    trimmed = []

    for msg in reversed(memory):
        tokens = estimate_tokens(msg["content"])
        if total_tokens + tokens <= max_tokens:
            trimmed.insert(0, msg)
            total_tokens += tokens
        else:
            break

    return trimmed
```

---

## 12. FastAPI + Cohere Integration

### Install Dependencies

```bash
pip install fastapi uvicorn cohere
```

### Cohere Client

```python
import cohere
co = cohere.Client("YOUR_COHERE_API_KEY")
```

### FastAPI App

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
chat_memory = []
```

### Request Schema

```python
class ChatRequest(BaseModel):
    message: str
```

### Chat Endpoint

```python
@app.post("/chat")
def chat(req: ChatRequest):
    global chat_memory

    chat_memory.append({
        "role": "user",
        "content": req.message
    })

    safe_memory = trim_memory(
        chat_memory,
        AVAILABLE_MEMORY_TOKENS
    )

    chat_history = []
    for msg in safe_memory:
        chat_history.append({
            "role": "USER" if msg["role"] == "user" else "CHATBOT",
            "message": msg["content"]
        })

    response = co.chat(
        model="command",
        message=req.message,
        chat_history=chat_history,
        preamble="You are a helpful AI assistant.",
        max_tokens=RESPONSE_TOKENS
    )

    assistant_reply = response.text

    chat_memory.append({
        "role": "assistant",
        "content": assistant_reply
    })

    return {
        "response": assistant_reply,
        "memory_tokens_used": sum(
            estimate_tokens(m["content"]) for m in safe_memory
        )
    }
```

---

## 13. Summary Memory Integration

```python
def summarize_memory(memory):
    combined = " ".join(m["content"] for m in memory)
    return f"Summary: {combined[:600]}"
```

---

## 14. Hybrid Memory Architecture

```
Recent chat        → Buffer memory
Old conversation   → Summary memory
Facts & documents → Vector DB
User preferences  → Database
```

---

## 15. Interview-Ready Formula

```
LLM Memory =
System Prompt + Memory + User Input + Output ≤ Context Window
```

---

## 16. Key Takeaways

- FastAPI is a controller, not a memory manager
- Tokens are the real currency
- Best systems use hybrid memory architectures
