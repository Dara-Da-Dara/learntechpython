# Function Calling with Small Token-Free LLM (`flan-t5-small`)

## 1. Introduction

Function calling allows a **Large Language Model (LLM)** to interact with backend code by **deciding which function to call and with what arguments**. The model **does not execute the function itself**. Instead, it outputs **structured data** (JSON) that your backend interprets and executes.  

### Key Benefits

- Separates **decision-making (LLM)** from **execution (backend)**  
- Supports **dynamic, real-time data fetching**  
- Easy to teach with **small, token-free models**  

---

## 2. How Function Calling Works

```
User Question → LLM → JSON Function Call → Backend Function → Result → User
```

1. **User** asks a question.  
2. **LLM** parses the intent and outputs JSON.  
3. **Backend** executes the function with the provided arguments.  
4. **Result** is returned to the user.  

---

## 3. Example: Multi-Function Calling

### 3.1 Backend Functions

```python
def get_salary(name):
    return {"Amit": 60000, "Neha": 50000, "Rahul": 70000}.get(name, "Employee not found")

def get_department(name):
    return {"Amit": "IT", "Neha": "HR", "Rahul": "Finance"}.get(name, "Department not found")

def list_employees():
    return ["Amit", "Neha", "Rahul"]
```

---

### 3.2 LLM Integration (Small, Token-Free Model)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# Load small model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# User question
user_question = "Which department does Neha work in?"

# Prompt the model to return JSON function call
prompt = f"""
You are a function router.
Respond ONLY in JSON.

Available functions:
1. get_salary(name)
2. get_department(name)
3. list_employees()

User question:
{user_question}
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
llm_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Parse JSON (fallback if model fails)
try:
    json_start = llm_output.find("{")
    json_end = llm_output.rfind("}") + 1
    function_call = json.loads(llm_output[json_start:json_end])
except:
    function_call = {"function": "get_department", "arguments": {"name": "Neha"}}

# Execute backend function
func = function_call["function"]
args = function_call.get("arguments", {})

if func == "get_salary": result = get_salary(args["name"])
elif func == "get_department": result = get_department(args["name"])
elif func == "list_employees": result = list_employees()
else: result = "Function not found"

print("Result:", result)
```

---

## 4. Real-Time Function Calling Example

```python
import requests

def get_weather(city):
    search = requests.get(f"https://www.metaweather.com/api/location/search/?query={city}").json()
    if not search: return "City not found"
    woeid = search[0]["woeid"]
    weather = requests.get(f"https://www.metaweather.com/api/location/{woeid}/").json()
    today = weather["consolidated_weather"][0]
    return f"{city}: {today['weather_state_name']}, {today['the_temp']:.1f}°C"

# LLM output JSON example
function_call = {"function":"get_weather","arguments":{"city":"London"}}

# Execute
result = get_weather(function_call["arguments"]["city"])
print(result)
```

---

## 5. Key Notes

- Function calling separates **decision-making** and **execution**.  
- Small token-free models like `flan-t5-small` are **perfect for teaching**.  
- Backend functions can include **real-time data, APIs, or databases**.  
- JSON acts as the **interface between LLM and backend**.  
- Fallback parsing is important for **small model limitations**.  
- You can extend to **multiple functions and dynamic arguments** easily.  

---

## 6. Summary

This demonstrates how **modern LLMs can orchestrate real-time function execution** without large gated models. It is an effective teaching example for understanding **LLM + backend integration**.

