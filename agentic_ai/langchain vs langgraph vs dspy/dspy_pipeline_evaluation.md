# DSPy Pipeline and Model Evaluation

## 1. Introduction

DSPy is a framework for building structured, optimized, and evaluated
LLM pipelines. Instead of manually writing prompts, DSPy allows you to
define programs that automatically improve through optimization and
evaluation.

------------------------------------------------------------------------

## 2. DSPy Pipeline Overview

A typical DSPy pipeline follows this structure:

    Inputs → DSPy Modules → Optimizer → Evaluation → Optimized Program

### Key Components

  Component    Purpose
  ------------ ------------------------------------
  Signature    Defines input and output structure
  Module       Contains program logic
  Dataset      Training and evaluation examples
  Metric       Measures output quality
  Optimizer    Improves prompt and reasoning
  Evaluation   Measures performance

------------------------------------------------------------------------

## 3. DSPy Pipeline Implementation

### Step 1: Setup DSPy

``` python
import dspy

llm = dspy.OpenAI(
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY"
)

dspy.settings.configure(llm=llm)
```

------------------------------------------------------------------------

### Step 2: Define Signature

``` python
class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Correct factual answer")
```

------------------------------------------------------------------------

### Step 3: Define Module

``` python
class QAPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QA)

    def forward(self, question):
        return self.predict(question=question)
```

------------------------------------------------------------------------

## 4. Creating Dataset

``` python
trainset = [
    dspy.Example(
        question="What is RPA?",
        answer="Robotic Process Automation automates repetitive tasks."
    ).with_inputs("question"),

    dspy.Example(
        question="What is ETL?",
        answer="ETL stands for Extract, Transform, Load."
    ).with_inputs("question"),
]
```

------------------------------------------------------------------------

## 5. Model Evaluation Metrics

### Exact Match Metric

``` python
def exact_match(example, pred):
    return example.answer.lower() == pred.answer.lower()
```

### Containment Metric

``` python
def contains_answer(example, pred):
    return example.answer.lower() in pred.answer.lower()
```

------------------------------------------------------------------------

## 6. Baseline Model Evaluation

``` python
pipeline = QAPipeline()

evaluator = dspy.Evaluate(
    devset=trainset,
    metric=contains_answer,
    num_threads=1
)

baseline_score = evaluator(pipeline)
print("Baseline score:", baseline_score)
```

------------------------------------------------------------------------

## 7. Optimizing the Pipeline

``` python
optimizer = dspy.MIPRO(
    metric=contains_answer,
    max_bootstrapped_demos=2
)

optimized_pipeline = optimizer.compile(
    QAPipeline(),
    trainset=trainset
)
```

------------------------------------------------------------------------

## 8. Evaluate Optimized Model

``` python
optimized_score = evaluator(optimized_pipeline)
print("Optimized score:", optimized_score)
```

------------------------------------------------------------------------

## 9. Prediction Comparison

``` python
question = "Explain RPA"

print("Before optimization:")
print(pipeline(question).answer)

print("After optimization:")
print(optimized_pipeline(question).answer)
```

------------------------------------------------------------------------

## 10. DSPy vs Traditional Prompt Engineering

  Feature           DSPy           Traditional
  ----------------- -------------- -------------
  Prompt Writing    Automatic      Manual
  Optimization      Built-in       Manual
  Evaluation        Integrated     External
  Pipeline Design   Programmatic   Chain-based

------------------------------------------------------------------------

## 11. Real-world Applications

-   RAG-based Question Answering
-   ServiceNow Ticket Analysis
-   Agentic AI Systems
-   Chatbot Evaluation Pipelines
-   Knowledge Base Assistants

------------------------------------------------------------------------

## 12. Conclusion

DSPy provides a structured way to build, optimize, and evaluate LLM
pipelines. It enables scalable and production-ready AI systems with
minimal manual prompt engineering.
