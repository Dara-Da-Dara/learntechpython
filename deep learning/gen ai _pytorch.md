# Install Hugging Face Transformers if not installed
# pip install transformers torch 

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # you can also use "gpt2-medium" or "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input prompt
prompt = "Artificial Intelligence is transforming"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(
    input_ids,
    max_length=100,      # total length of generated text
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)

# Decode generated text




### no pytorch 
# pip install transformers

from transformers import pipeline

# Create a text-generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text
prompt = "Artificial Intelligence is transforming"
result = generator(prompt, max_length=100, num_return_sequences=1)

print(result[0]['generated_text'])

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)
