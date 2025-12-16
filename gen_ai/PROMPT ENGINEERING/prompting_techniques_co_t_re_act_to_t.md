# Prompting Techniques with Examples (Chain-of-Thought, ReAct, Tree-of-Thought)

This document provides **clear prompt templates**, **code-style usage**, and **sample outputs** for three important reasoning techniques used with LLMs:

- Chain-of-Thought (CoT)
- ReAct (Reason + Act)
- Tree-of-Thought (ToT)

The examples are **teaching-friendly** and suitable for classroom demonstration.

---

## 1. Chain-of-Thought (CoT)

### Concept
Chain-of-Thought encourages the model to **reason step by step** before giving the final answer. It improves accuracy for logical and mathematical tasks.

---

### Prompt Example (Explicit CoT)

```text
You are a reasoning assistant.
Solve the problem step by step and explain your reasoning clearly.

Question:
If a company has revenue of 120, expenses of 75, and tax of 15, what is the net profit?
```

---

### Model Output

```text
Step 1: Revenue is 120.
Step 2: Expenses are 75.
Step 3: Tax is 15.
Step 4: Total deductions = 75 + 15 = 90.
Step 5: Net profit = 120 - 90 = 30.

Final Answer: The net profit is 30.
```

---

### Python-style Prompt Usage

```python
prompt = """
Solve step by step.
Revenue = 120
Expenses = 75
Tax = 15
Find net profit.
"""
```

---

## 2. ReAct (Reason + Act)

### Concept
ReAct combines **reasoning** with **actions** (such as calling tools, APIs, or functions). The model alternates between thinking and acting.

---

### Prompt Example

```text
You are an AI assistant that reasons and takes actions.

Task:
Find the average of numbers 10, 20, and 30.

Use the format:
Thought: reasoning
Action: calculation
Observation: result
Final Answer:
```

---

### Model Output

```text
Thought: I need to calculate the average of three numbers.
Action: (10 + 20 + 30) / 3
Observation: 60 / 3 = 20
Final Answer: The average is 20.
```

---

### ReAct with Tool (Pseudo-code)

```python
Thought: Need average
Action: calculator([10, 20, 30])
Observation: 20
Final Answer: 20
```

---

### Real-world Use Case
- Chatbots with API calls
- AI agents
- Function calling in LLMs

---

## 3. Tree-of-Thought (ToT)

### Concept
Tree-of-Thought explores **multiple reasoning paths** before choosing the best solution. Useful for **complex decision-making**.

---

### Prompt Example

```text
You are a strategic reasoning assistant.

Problem:
A student wants to choose a career. Options are:
1. Data Scientist
2. AI Engineer
3. Business Analyst

Generate multiple reasoning paths and select the best option based on long-term growth.
```

---

### Model Output

```text
Thought Path 1:
Data Scientist offers strong analytical growth and demand.

Thought Path 2:
AI Engineer has high demand, requires deep ML and system skills.

Thought Path 3:
Business Analyst focuses more on business insights with limited AI depth.

Evaluation:
AI Engineer has the highest long-term growth with advanced skill demand.

Final Answer:
AI Engineer is the best option for long-term growth.
```

---

### Tree Expansion View

```text
Root Problem
 ├─ Path 1: Data Scientist → Moderate growth
 ├─ Path 2: AI Engineer → High growth ✔
 └─ Path 3: Business Analyst → Stable growth
```

---

## Comparison Summary

| Technique | Purpose | Best For |
|--------|--------|---------|
| CoT | Step-by-step reasoning | Math, logic, explanations |
| ReAct | Reason + tools | Agents, APIs, automation |
| ToT | Multiple reasoning paths | Strategy, planning, decisions |

---

## Teaching Notes (Key Points)

- CoT improves **accuracy**
- ReAct enables **tool integration**
- ToT supports **complex reasoning**
- Use concise prompts for students
- Avoid exposing internal reasoning in production

---

### End of Document
