# Module 2: Prompts and Prompt Engineering
**Duration:** 5 Hours | **Level:** Beginner

---

## 1. What is a Prompt?

### Definition
A **prompt** is the text input you give to an LLM to get a response. It's like asking a question or giving instructions.

### Simple Examples

```
Prompt 1: "What is Python?"
Response: "Python is a programming language..."

Prompt 2: "Write a joke about programming"
Response: "Why do programmers prefer dark mode? Because light attracts bugs!"
```

---

## 2. Why Prompts Matter

The quality of your prompt directly determines the quality of the response.

### Bad Prompt vs Better Prompt

**Bad:** "Tell me about AI"  
**Better:** "Explain how transformer-based LLMs work, in simple terms for a beginner, with examples"

### Prompt Engineering Value

- 5% improvement in prompt → 30% better results
- Free optimization (no model changes needed)
- Immediate impact (test right away)
- Reusable (use improved prompts everywhere)

---

## 3. Basic Prompt Types

### Type 1: Question Prompts
```
"What is machine learning?"
```

### Type 2: Instruction Prompts
```
"Write a Python function to calculate factorial"
```

### Type 3: Completion Prompts
```
"The capital of France is"
```

### Type 4: Conversation Prompts
```
User: "Hi, how are you?"
Assistant: "I'm doing well! How can I help?"
```

### Type 5: Role-Play Prompts
```
"You are an expert Python developer. Explain decorators in Python."
```

---

## 4. Prompt Engineering Techniques

### Technique 1: Be Specific and Clear

**Unclear:**
```
"Tell me about web development"
```

**Clear:**
```
"Explain the differences between frontend and backend web development,
with 2-3 examples of technologies used in each"
```

### Technique 2: Provide Context

**No Context:**
```
"Summarize this for me"
```

**With Context:**
```
"Summarize the following research paper in 3-4 bullet points,
focusing on the novel contributions and methodology"
```

### Technique 3: Define Output Format

**Undefined:**
```
"Give me a to-do list"
```

**Defined:**
```
"Create a JSON list with 5 tasks, each containing: id, title, description, priority"
```

### Technique 4: Use Examples (Few-Shot Learning)

```
"Convert numbers to English words. Follow this pattern:

Example 1:
Input: 5
Output: five

Example 2:
Input: 23
Output: twenty-three

Now convert:
Input: 42"
```

### Technique 5: Give Role and Persona

```
"You are an expert Python developer with 10 years of experience.
Explain the top 5 benefits of Python for beginners."
```

### Technique 6: Break Down Complex Tasks

Instead of one complex prompt, use multiple prompts in steps.

---

## 5. Prompt Templates in LangChain

### What is a Prompt Template?

A **PromptTemplate** is a reusable template for constructing prompts with variables.

### Basic Template Example

```python
from langchain.prompts import PromptTemplate

template = "What is the capital of {country}?"

prompt_template = PromptTemplate(
    input_variables=["country"],
    template=template
)

prompt = prompt_template.format(country="France")
print(prompt)
# Output: "What is the capital of France?"
```

### Why Use Templates?

- Reusability - Use same template with different inputs
- Consistency - Same format every time
- Maintainability - Update template in one place
- Testing - Easy to test different inputs

---

## 6. Chat Prompt Templates

For conversational applications, use **ChatPromptTemplate**.

### Why ChatPromptTemplate?

Conversations have multiple roles:
- **System** - Defines assistant behavior
- **Human** - User messages
- **AI** - Assistant responses

### Chat Template Example

```python
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Provide clear answers."),
    ("human", "Hello! I'm learning about {topic}"),
    ("ai", "That's great! {topic} is a fascinating field."),
    ("human", "{user_question}"),
])

messages = chat_template.format_messages(
    topic="machine learning",
    user_question="What's a neural network?"
)
```

---

## 7. Advanced Prompt Techniques

### Technique: Chain of Thought (CoT)

Make the LLM **think step by step** to get better reasoning.

**Without CoT:**
```
"Is 17 a prime number?"
Response: "Yes"
```

**With CoT:**
```
"Let's think step by step. Is 17 a prime number?

Step 1: A prime number is only divisible by 1 and itself
Step 2: Let me check divisors of 17
Step 3: 17 ÷ 2 = 8.5 (not divisible)
Step 4: No other divisors..."
```

### Technique: Few-Shot Prompting

Provide examples to guide the LLM.

### Technique: Role-Based Prompting

Give the LLM a specific role to enhance responses.

---

## 8. Common Prompt Mistakes

### Mistake 1: Being Too Vague
```
Avoid: "Write about technology"
Better: "Write a 500-word article about AI trends in 2025"
```

### Mistake 2: Asking Too Much at Once
Break complex tasks into steps.

### Mistake 3: Not Specifying Format
Always define what output format you want.

### Mistake 4: Ignoring Context
Provide necessary context for better responses.

### Mistake 5: Expecting Perfect First Try
Iterate: Ask, review, refine, try again.

---

## 9. Prompt Best Practices

### Best Practices Checklist

- Be specific - Give clear, detailed instructions
- Provide context - Explain the situation
- Define output - Show exactly what format you want
- Use examples - Few-shot learning helps
- Give role/persona - "You are an expert at..."
- Break down - Use steps for complex tasks
- Iterate - Refine prompts based on results
- Test - Try different formulations
- Use templates - Reuse for consistency
- Limit length - Keep prompts reasonable

---

## 10. Review Questions

1. What is a prompt?
2. Why do prompts matter?
3. What's the difference between PromptTemplate and ChatPromptTemplate?
4. Name 3 prompt engineering techniques.
5. What is Chain of Thought prompting?

---

## 11. Next Steps

You now understand:
- What prompts are
- Why they matter
- Prompt engineering techniques
- Templates in LangChain
- Best practices

**Next Module:** Module 3 - Chains and Sequences

---

**Module Summary**
- Prompts are inputs that determine LLM output quality
- Good prompt engineering significantly improves results
- Templates make prompts reusable and consistent
- Chat templates handle conversational interactions
- Advanced techniques improve reasoning

**Time spent:** 5 hours
