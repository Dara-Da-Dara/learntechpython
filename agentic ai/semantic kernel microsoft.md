# Agentic AI with Semantic Kernel (Python)

*A complete, class-ready Markdown guide combining concepts, architecture, and end-to-end Python code examples.*

---

## 1. What is Agentic AI?

**Agentic AI** refers to AI systems that act as **autonomous agents** rather than simple question–answer models.

An agentic system can:

* Understand a **goal**
* **Reason** about steps
* **Plan** actions
* **Choose tools** dynamically
* **Execute** those tools
* **Remember** past information
* **Reflect** and improve
* **Stop** when the goal is achieved

This mirrors how humans solve problems.

---

## 2. What is Semantic Kernel (SK)?

**Semantic Kernel** is Microsoft’s AI orchestration framework that enables agentic behavior.

It provides:

* LLM integration
* Tool / function calling
* Plugin (skill) architecture
* Planners for multi-step reasoning
* Memory (vector-based)
* Enterprise-ready orchestration

> **Semantic Kernel is the runtime that makes Agentic AI possible.**

---

## 3. Agentic Architecture Using Semantic Kernel

```
User Goal
   ↓
LLM Agent (Reasoning)
   ↓
Planner (Task Decomposition)
   ↓
Plugins / Skills (Actions)
   ↓
Memory (Recall & Learning)
   ↓
Reflection / Completion
```

---

## 4. Core Components

### 4.1 Agent (LLM)

Acts as the **decision-maker**.

### 4.2 Plugins / Skills

* **Native skills** → Python functions
* **Semantic skills** → Prompt-based functions

### 4.3 Planner

* Breaks goals into steps
* Selects tools automatically
* Executes sequentially

### 4.4 Memory

* Stores facts and experiences
* Enables long-term recall
* Improves context awareness

---

## 5. Installation

```bash
pip install semantic-kernel openai
```

---

## 6. Kernel Setup (Agent Brain)

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

kernel.add_chat_service(
    service_id="agent-llm",
    service=OpenAIChatCompletion(
        model_id="gpt-4",
        api_key="YOUR_OPENAI_API_KEY"
    )
)
```

**Explanation:**

* Kernel = agent runtime
* LLM = reasoning engine

---

## 7. Native Plugin (Tool Capability)

### Math Plugin

```python
class MathPlugin:
    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
```

### Register Plugin

```python
kernel.add_plugin(
    plugin_instance=MathPlugin(),
    plugin_name="math"
)
```

The agent can now **decide when to use math tools**.

---

## 8. Semantic Plugin (Prompt-Based Skill)

### Text Summarizer

```python
summarize_prompt = """
Summarize the following content in 3 bullet points:

{{$input}}
"""

summarize_function = kernel.create_semantic_function(
    prompt=summarize_prompt,
    function_name="summarize",
    plugin_name="text"
)
```

This allows the agent to perform **language-based actions**.

---

## 9. Memory (Semantic Memory Store)

```python
from semantic_kernel.memory import VolatileMemoryStore

kernel.register_memory_store(VolatileMemoryStore())
```

### Save Information

```python
await kernel.memory.save_information(
    collection="agent_memory",
    id="fact1",
    text="Company sales increased by 25% in Q3"
)
```

### Recall Information

```python
results = await kernel.memory.search(
    collection="agent_memory",
    query="sales growth",
    limit=1
)

print(results[0].text)
```

Memory enables **learning and contextual reasoning**.

---

## 10. Planner (Agentic Intelligence)

```python
from semantic_kernel.planning.basic_planner import BasicPlanner

planner = BasicPlanner()
```

The planner gives the agent **autonomy**.

---

## 11. Agentic Goal Execution

### Define Goal

```python
goal = """
Add two numbers,
explain the result,
and summarize it professionally
"""
```

### Create and Execute Plan

```python
plan = planner.create_plan(goal, kernel)

result = await plan.invoke_async(kernel)

print(result)
```

**What happens internally:**

1. Agent understands the goal
2. Chooses math + summarization tools
3. Executes steps automatically
4. Produces final output

---

## 12. Real-World Business Agent Example

### Goal

```python
goal = """
Recall sales data from memory,
analyze growth,
and generate an executive summary
"""
```

### Execution

```python
plan = planner.create_plan(goal, kernel)
output = await plan.invoke_async(kernel)

print(output)
```

This simulates a **business analyst AI agent**.

---

## 13. Reflection (Self-Evaluation Agent)

```python
reflection_prompt = """
Evaluate the quality of the following output.
Mention strengths and areas of improvement.

{{$input}}
"""

reflect_fn = kernel.create_semantic_function(
    prompt=reflection_prompt,
    function_name="reflect",
    plugin_name="agent"
)

reflection = await reflect_fn.invoke_async(
    input=str(output)
)

print(reflection)
```

Reflection enables **self-improving agents**.

---

## 14. Mapping to Agentic AI Concepts

| Agentic Concept | Semantic Kernel    |
| --------------- | ------------------ |
| Agent Brain     | LLM                |
| Tools           | Native Plugins     |
| Reasoning       | Planner            |
| Memory          | Vector Store       |
| Autonomy        | Plan Execution     |
| Reflection      | Semantic Functions |

---

## 15. Relation to Other Frameworks

| Framework        | Equivalent in SK     |
| ---------------- | -------------------- |
| ReAct            | Planner + Tool Calls |
| MCP              | External Plugins     |
| AutoGen          | Multi-agent SK setup |
| Chain-of-Thought | Planner Reasoning    |

---

## 16. One-Line Definition (Exam / Interview)

> **Agentic AI using Semantic Kernel is the design of autonomous, goal-driven AI agents that can reason, plan, use tools, recall memory, and execute multi-step tasks through an orchestration framework.**

---

## 17. Key Takeaways

* Semantic Kernel enables true agentic behavior
* Planner = autonomy
* Plugins = actions
* Memory = intelligence over time
* Reflection = self-improvement

---

**End of Document**
