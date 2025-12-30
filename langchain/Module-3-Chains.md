# Module 3: Chains and Sequences
**Duration:** 5 Hours | **Level:** Beginner-Intermediate

---

## 1. What is a Chain?

### Definition
A **Chain** in LangChain is a structured workflow that combines multiple components (LLM, prompt, output parser) to accomplish a task with multiple steps.

### Simple Analogy
Think of a chain like a recipe:
- **Input:** Raw ingredients (your question)
- **Steps:** Follow recipe instructions (prompt → LLM → parse)
- **Output:** Final dish (LLM response)

---

## 2. Components of a Chain

### Basic Chain Structure

```
Input Variables
    ↓
Prompt Template (formats the input)
    ↓
LLM (generates response)
    ↓
Output Parser (processes output)
    ↓
Final Result
```

### Core Components

1. **Input Variables** - Variables you pass to the chain
2. **Prompt Template** - Formats input into a prompt
3. **LLM** - The language model (GPT-4, Claude, etc.)
4. **Output Parser** - Processes and validates output

---

## 3. Building Your First Chain

### Step-by-Step Example

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Step 1: Create the LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key="your_api_key"
)

# Step 2: Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in 2-3 sentences for a beginner."
)

# Step 3: Create the chain
chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

# Step 4: Run the chain
result = chain.run(topic="Blockchain")
print(result)
```

---

## 4. Sequential Chains

### What is a Sequential Chain?

A **Sequential Chain** combines multiple chains in order, where the output of one chain becomes the input to the next.

### Use Cases

- Multi-step workflows
- Progressive refinement
- Data transformation pipelines
- Complex reasoning tasks

### Example: Article Writing Pipeline

```python
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI()

# Step 1: Generate outline
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Create an outline for an article about {topic}"
)
outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")

# Step 2: Write article based on outline
article_prompt = PromptTemplate(
    input_variables=["outline"],
    template="Write a detailed article based on this outline:\n{outline}"
)
article_chain = LLMChain(llm=llm, prompt=article_prompt, output_key="article")

# Combine into sequential chain
overall_chain = SequentialChain(
    chains=[outline_chain, article_chain],
    input_variables=["topic"],
    output_variables=["outline", "article"]
)

result = overall_chain({"topic": "Artificial Intelligence"})
```

---

## 5. Chat-Based Chains

For conversational applications, use **ConversationChain**.

### What is ConversationChain?

A chain specifically designed for managing conversations with memory.

### Example: Simple Chatbot

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Create LLM
llm = ChatOpenAI(model="gpt-4")

# Create memory to store conversation
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
print(conversation.run(input="Hi! What's your name?"))
print(conversation.run(input="What are the benefits of Python?"))
print(conversation.run(input="You mentioned Python before, right?"))
```

---

## 6. Conversation Chain with System Prompt

Control the assistant's personality and behavior:

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    system_message="You are a helpful Python programming tutor."
)

response = conversation.run(input="What is a loop in Python?")
```

---

## 7. Router Chains

### What is a Router Chain?

A chain that **decides which sub-chain to run** based on input.

### Use Case

Different inputs need different processing:
- Math questions → Math solver
- Code questions → Code expert
- General questions → General assistant

---

## 8. Chain Configuration

### Setting Parameters

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,      # Randomness
    max_tokens=500,       # Max response length
    top_p=0.9            # Nucleus sampling
)

# Create chain
prompt = PromptTemplate(
    template="Write a {style} poem about {topic}",
    input_variables=["style", "topic"]
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(style="haiku", topic="Python")
```

### Verbose Mode

```python
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True  # Shows prompt and steps
)
```

---

## 9. Error Handling in Chains

### Handling Errors Gracefully

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI()
prompt = PromptTemplate(template="Solve: {problem}", input_variables=["problem"])
chain = LLMChain(llm=llm, prompt=prompt)

try:
    result = chain.run(problem="What is 2+2?")
    print(result)
except Exception as e:
    print(f"Error: {e}")
```

---

## 10. Chain Best Practices

### Do's

- Use templates for consistent prompts
- Add memory to conversational chains
- Test each step separately first
- Use descriptive output keys
- Handle errors gracefully

### Don'ts

- Don't hardcode prompts
- Don't ignore errors
- Don't make chains too long
- Don't forget to validate outputs

---

## 11. Review Questions

1. What is a chain in LangChain?
2. What's the difference between LLMChain and SequentialChain?
3. What does ConversationChain provide?
4. How does a Router Chain work?
5. Why use chains instead of raw API calls?

---

## 12. Next Steps

**Next Module:** Module 4 - Memory Management

---

**Module Summary**
- Chains combine prompts, LLMs, and parsers into workflows
- Sequential chains enable multi-step processing
- Conversation chains maintain memory for context
- Router chains select appropriate processing paths
- Chains are more structured and reusable than raw APIs

**Time spent:** 5 hours
