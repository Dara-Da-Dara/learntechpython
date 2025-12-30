# Module 6: Agents and Autonomous Systems
**Duration:** 6 Hours | **Level:** Intermediate-Advanced

---

## 1. What is an Agent?

### Definition
An **Agent** is an AI system that can **decide which tools to use** to accomplish a task, rather than following a predefined sequence.

### Agent vs Chain

**Chain:** Predetermined steps
```
Step 1 → Step 2 → Step 3 → Output
```

**Agent:** Dynamic decision-making
```
Analyze task → Decide which tool → Use tool → Repeat if needed → Output
```

---

## 2. Types of Agents

### ReAct (Reasoning + Acting)
Alternates between thinking and acting.

```
Thought: I need to search for information
Action: Use search tool
Observation: Found relevant information
Thought: Now I need to analyze this
Action: Use calculator
Observation: Got result
Thought: I can now answer the question
Final Answer: ...
```

### Tool-Using Agent
Decides which available tools to use.

### Zero-Shot Agent
Works without examples, uses tool descriptions.

### Few-Shot Agent
Works with examples to guide behavior.

---

## 3. Building a Simple Agent

### Basic Agent Example

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

# Define tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching the internet"
    )
]

# Initialize agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
result = agent.run("What is the latest news on AI?")
```

---

## 4. Creating Custom Tools

### Define Your Own Tools

```python
from langchain.tools import Tool
import math

def calculate_factorial(n: str) -> str:
    """Calculate factorial of a number"""
    try:
        result = math.factorial(int(n))
        return str(result)
    except:
        return "Invalid input"

factorial_tool = Tool(
    name="Factorial Calculator",
    func=calculate_factorial,
    description="Calculate factorial of n. Input: positive integer n"
)

tools = [factorial_tool]
```

---

## 5. Tool Calling in Agents

### How Agents Call Tools

```
1. Agent analyzes the task
2. Determines which tool to use
3. Formats tool input correctly
4. Executes tool
5. Observes result
6. Decides if more tools needed
7. Returns final answer
```

### Tool Format

Each tool needs:
- **name** - Identifier for the tool
- **description** - What the tool does (helps agent decide)
- **func** - The actual function to execute

---

## 6. Agent Memory

### Agents with Conversation Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
agent.run("My name is Alice")
agent.run("What's my name?")
# Agent remembers from memory
```

---

## 7. Agent Error Handling

### Handling Tool Errors

```python
try:
    result = agent.run("Complex question that might fail")
except Exception as e:
    print(f"Agent error: {e}")
    # Fallback approach
```

### Max Iterations

```python
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5,  # Prevent infinite loops
    early_stopping_method="generate"
)
```

---

## 8. Multi-Agent Systems

### Multiple Agents Working Together

```python
# Agent 1: Research specialist
research_agent = create_agent(research_tools, "Research")

# Agent 2: Analysis specialist
analysis_agent = create_agent(analysis_tools, "Analysis")

# Coordinator agent
coordinator = create_agent(
    [research_agent, analysis_agent],
    "Coordinator"
)

# Task
result = coordinator.run("Analyze market trends")
# Coordinator delegates to specialists
```

---

## 9. Practical Agent Examples

### Example 1: Research Assistant

```python
# Tools: Search, Web scraping, Document parsing
# Task: Research a topic and compile findings
```

### Example 2: Code Assistant

```python
# Tools: Code execution, Documentation lookup, Error debugging
# Task: Debug and fix code
```

---

## 10. Agent Best Practices

### Do's

- Define clear tool descriptions
- Provide few-shot examples
- Set max iterations
- Monitor token usage
- Test individual tools first

### Don'ts

- Don't create too many tools
- Don't use vague descriptions
- Don't ignore error handling
- Don't assume agent will work first try

---

## 11. Review Questions

1. What is an agent?
2. How is agent different from chain?
3. What is ReAct?
4. How do you create a custom tool?
5. How do agents handle errors?

---

## 12. Next Steps

**Next Module:** Module 7 - Tools, APIs & Function Calling

---

**Module Summary**
- Agents make dynamic decisions about which tools to use
- ReAct combines reasoning and acting
- Custom tools extend agent capabilities
- Agents can use memory for multi-turn conversations
- Multi-agent systems enable complex workflows

**Time spent:** 6 hours
