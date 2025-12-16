# **Prompt Engineering – Code Examples with Models and Outputs**

---

## **Purpose of This File**
This document provides **hands-on, executable-style code examples** demonstrating how prompts are used with **different LLM models**. Each section includes:
- Model used
- Prompt design
- Code snippet
- Expected output (illustrative)

This file is suitable for **labs, demonstrations, and exams**.

---

## **1. Zero-Shot Prompting (Text Generation)**

### **Model Used**
- GPT-style / Instruction-tuned LLM

### **Prompt**
```text
Explain overfitting in machine learning in simple terms.
```

### **Python Code Example**
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Explain overfitting in machine learning in simple terms."

output = generator(prompt, max_length=80)
print(output[0]['generated_text'])
```

### **Sample Output**
```text
Overfitting occurs when a machine learning model learns the training data too well, including noise, which reduces its performance on new data.
```

---

## **2. Few-Shot Prompting (Classification Task)**

### **Model Used**
- GPT / Instruction-based LLM

### **Prompt**
```text
Classify the sentiment:
Text: I love this course → Positive
Text: This lecture is confusing → Negative
Text: The exam was average → Neutral
Text: The software is slow →
```

### **Python Code Example**
```python
prompt = """
Classify the sentiment:
Text: I love this course → Positive
Text: This lecture is confusing → Negative
Text: The exam was average → Neutral
Text: The software is slow →
"""

output = generator(prompt, max_length=60)
print(output[0]['generated_text'])
```

### **Sample Output**
```text
Negative
```

---

## **3. Chain-of-Thought Prompting (Reasoning)**

### **Model Used**
- Reasoning-capable LLM

### **Prompt**
```text
Solve step by step.
If a pen costs ₹10, how much do 7 pens cost?
```

### **Python Code Example**
```python
prompt = "Solve step by step. If a pen costs ₹10, how much do 7 pens cost?"

output = generator(prompt, max_length=80)
print(output[0]['generated_text'])
```

### **Sample Output**
```text
One pen costs ₹10.
Seven pens cost 7 × 10 = ₹70.
```

---

## **4. Role-Based Prompting**

### **Model Used**
- Instruction-tuned LLM

### **Prompt**
```text
You are a university professor.
Explain Artificial Intelligence to first-year students.
```

### **Python Code Example**
```python
prompt = "You are a university professor. Explain Artificial Intelligence to first-year students."

output = generator(prompt, max_length=100)
print(output[0]['generated_text'])
```

### **Sample Output**
```text
Artificial Intelligence is a field of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence.
```

---

## **5. Tree of Thought Prompting (Decision Making)**

### **Model Used**
- Advanced reasoning LLM

### **Prompt**
```text
Generate three possible solutions.
Evaluate each.
Select the best one.

Problem: How should a student prepare for exams in limited time?
```

### **Python Code Example**
```python
prompt = """
Generate three possible solutions.
Evaluate each.
Select the best one.

Problem: How should a student prepare for exams in limited time?
"""

output = generator(prompt, max_length=150)
print(output[0]['generated_text'])
```

### **Sample Output**
```text
Option 1: Study all topics briefly – less depth.
Option 2: Focus on weak areas – efficient.
Option 3: Revise only strong topics – risky.
Best Option: Option 2.
```

---

## **6. ReAct Prompting (Reason + Action)**

### **Model Used**
- Agent-style LLM

### **Prompt**
```text
Thought: I need factual information.
Action: Retrieve data.
Observation: Analyze results.
Final Answer: Respond clearly.

Question: What is the capital of India?
```

### **Python Code Example**
```python
prompt = """
Thought: I need factual information.
Action: Retrieve data.
Observation: Analyze results.
Final Answer:
Question: What is the capital of India?
"""

output = generator(prompt, max_length=80)
print(output[0]['generated_text'])
```

### **Sample Output**
```text
Final Answer: The capital of India is New Delhi.
```

---

## **7. Prompt Self-Evaluation Example**

### **Prompt**
```text
Explain Generative AI.
Then evaluate your answer for accuracy and clarity.
```

### **Sample Output**
```text
Answer: Generative AI creates new content using learned data patterns.
Evaluation: Accurate and clearly explained.
```

---

## **Summary Table**

| Technique | Model Capability | Use Case |
|--------|----------------|----------|
| Zero-shot | Basic | Simple queries |
| Few-shot | Pattern learning | Classification |
| CoT | Reasoning | Math & logic |
| ToT | Decision making | Planning |
| ReAct | Tool use | AI agents |

---

## **Learning Outcomes**

Students will be able to:
- Implement prompt techniques using real LLM models
- Analyze model outputs based on prompt design
- Compare reasoning strategies across prompts
- Apply prompt engineering in practical applications

