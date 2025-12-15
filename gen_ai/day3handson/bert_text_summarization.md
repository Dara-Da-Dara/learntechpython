# Text Summarization using BERT

## Step 1: Install Required Libraries
```bash
pip install bert-extractive-summarizer
pip install spacy
python -m spacy download en_core_web_sm
```

## Step 2: Python Code for Summarization
```python
from summarizer import Summarizer

# Initialize BERT model
model = Summarizer()

# Your text
text = """
Artificial Intelligence (AI) is transforming the world in numerous ways.
It is being used in healthcare, finance, transportation, and many other sectors.
AI can analyze large amounts of data quickly and efficiently.
However, it also raises ethical questions regarding privacy and decision-making.
Understanding AI is essential for preparing for future technological advancements.
"""

# Generate summary
summary = model(text)

print("Original Text:\n", text)
print("\nSummary:\n", summary)
```

## Output Example
```
Original Text:
Artificial Intelligence (AI) is transforming the world in numerous ways...

Summary:
Artificial Intelligence (AI) is transforming the world in numerous ways. It is being used in healthcare, finance, transportation, and many other sectors.
```

## Notes
- `bert-extractive-summarizer` is **extractive**, meaning it selects important sentences from the original text.
- For **longer texts**, it works very well with BERT embeddings.
- For **abstractive summarization** (generating a new summary), consider using models like **T5, BART, or Pegasus**.

