# Langfuse: Overview and Integration Guide

## What is Langfuse?
Langfuse is an **open-source LLM engineering platform** designed to help developers build, monitor, debug, and optimize applications powered by large language models (LLMs) such as OpenAI, Anthropic, and Google Gemini.

It provides full **observability**, **prompt management**, **evaluation metrics**, and **tooling integrations** to improve LLM-powered applications.

---

## Key Features

- **Observability & Tracing**: Track all inputs, outputs, tool calls, retries, latencies, and costs.
- **Prompt Management**: Version control, test, and experiment with prompts.
- **Evaluation & Metrics**: Automated and human evaluations of output quality.
- **Integrated Tooling**: Dashboards, analytics, playground, SDKs for Python and JavaScript.
- **Open-Source & Self-Hostable**: MIT licensed, full control over deployment.
- **Multi-Model Support**: OpenAI, Anthropic, Google, Amazon Bedrock, etc.
- **Enterprise-Ready**: SOC 2, ISO 27001, GDPR/HIPAA compliant.

---

## Who Uses Langfuse?

- Developers and teams building production-grade LLM applications.
- Projects requiring prompt experimentation, debugging, performance tracking, and monitoring.

---

## Langfuse Integration with Python

### Installation
```bash
pip install langfuse
```

### Basic Setup
```python
from langfuse import LangfuseClient

# Initialize client with API key
client = LangfuseClient(api_key="YOUR_API_KEY")

# Track a simple LLM call
response = client.track_llm_call(
    model_name="gpt-4",
    input_text="Hello, Langfuse!",
)

print(response)
```

### Advanced Usage
- Track nested workflows and multiple tool calls.
- Attach metadata for experiments and evaluation.
- Monitor latency, errors, and cost for production LLM apps.

---

## Langfuse Integration with JavaScript

### Installation
```bash
npm install langfuse
```

### Basic Setup
```javascript
import { LangfuseClient } from 'langfuse';

const client = new LangfuseClient({ apiKey: 'YOUR_API_KEY' });

const response = await client.trackLlmCall({
  modelName: 'gpt-4',
  inputText: 'Hello, Langfuse!',
});

console.log(response);
```

---

## Resources
- [Langfuse Official Website](https://langfuse.com)
- [Langfuse Docs](https://langfuse.com/docs)
- [Langfuse GitHub](https://github.com/langfuse)
- [YCombinator Profile](https://www.ycombinator.com/companies/langfuse)

---

*This markdown file provides a quick overview of Langfuse and example integration code in Python and JavaScript for developers.*

