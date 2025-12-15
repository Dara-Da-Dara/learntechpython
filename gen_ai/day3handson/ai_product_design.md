# AI-Assisted Product Design

This guide shows how to generate product design concepts using AI (Stable Diffusion) in Python.

---

## 1. Required Libraries and Installation

Install the necessary libraries using pip:

```bash
pip install diffusers>=0.20.0
pip install transformers>=4.40.0
pip install torch>=2.0.0
pip install pillow>=10.0.0
pip install accelerate>=0.20.0  # optional, for faster GPU inference
pip install safetensors>=0.3.0   # optional, for efficient model storage
```

---

## 2. Python Code: AI Product Design Generator

```python
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# -------------------------------
# 1. Load the Stable Diffusion model
# -------------------------------
model_id = "runwayml/stable-diffusion-v1-5"  # You can change to other models
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# -------------------------------
# 2. Provide a product design prompt
# -------------------------------
prompt = """
Modern ergonomic office chair with a sleek design,
adjustable headrest, breathable mesh material,
futuristic color scheme, minimalistic style
"""

# -------------------------------
# 3. Generate the product design image
# -------------------------------
image = pipe(prompt, guidance_scale=7.5).images[0]

# -------------------------------
# 4. Save the image
# -------------------------------
image.save("ai_product_design.png")

print("✅ Product design image saved as ai_product_design.png")
```

---

## 3. requirements.txt

```
diffusers>=0.20.0
transformers>=4.40.0
torch>=2.0.0
pillow>=10.0.0
accelerate>=0.20.0
safetensors>=0.3.0
```

---

## 4. How to Run

1. Save the Python code as `product_design.py`.
2. Save the requirements as `requirements.txt`.
3. Install all libraries:
```bash
pip install -r requirements.txt
```
4. Run the script:
```bash
python product_design.py
```
5. The generated image will be saved as `ai_product_design.png`.

---

### Notes

- Be specific in the prompt for better design results.
- Use `guidance_scale` to control adherence to the prompt (higher = more faithful).
- You can iterate with multiple prompts to generate different design ideas.

