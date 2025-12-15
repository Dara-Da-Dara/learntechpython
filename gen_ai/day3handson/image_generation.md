# Text-to-Image Generation using Diffusion Model (Hugging Face)

This guide demonstrates how to create a text-to-image generation model using a diffusion model pipeline from Hugging Face.

---

## Step 1: Install Required Libraries

```bash
pip install torch torchvision diffusers transformers accelerate safetensors
```

> Note: The `torch` installation depends on your system and whether you want GPU support.

---

## Step 2: Import Required Modules

```python
import torch
from diffusers import StableDiffusionPipeline
```

---

## Step 3: Load the Pre-trained Diffusion Model

```python
# Use a pre-trained Stable Diffusion model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use GPU if available
```

---

## Step 4: Generate an Image from Text

```python
# Your text prompt
prompt = "A futuristic cityscape at sunset, digital art"

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")
```

---

## Step 5: Optional Settings for Better Quality

```python
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]
```
* 'guidance_scale

Also called classifier-free guidance scale.

Controls how strongly the model follows your text prompt.

Higher values → the image matches your prompt more closely, but may lose creativity or become less natural.

Lower values → the image may be more random, creative, or “artistic,” but might not match the prompt exactly.

Typical range: 5–15, default often around 7.5.'

* num_inference_steps

'This is the number of steps the diffusion model takes to generate the image.

More steps → smoother, higher-quality image, but slower generation.

Fewer steps → faster, but may produce rough or blurry images.

Typical range: 20–100, depending on quality and speed requirements.'
* `guidance_scale`: Higher values make the image closer to the prompt.
* `num_inference_steps`: More steps = better quality, but slower.

---

This code will generate a **high-quality image** based on your text prompt using a **diffusion model**.
 full code 

# -----------------------------
# Step 1: Install required libraries
# -----------------------------
!pip install torch torchvision diffusers transformers accelerate safetensors --quiet

# -----------------------------
# Step 2: Import required modules
# -----------------------------
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# -----------------------------
# Step 3: Load pre-trained Stable Diffusion model
# -----------------------------
model_id = "runwayml/stable-diffusion-v1-5"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
pipe = pipe.to(device)

# -----------------------------
# Step 4: Generate image from text
# -----------------------------
prompt = "A futuristic cityscape at sunset, digital art"

# Optional: improve quality
guidance_scale = 7.5
num_inference_steps = 50

image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]

# -----------------------------
# Step 5: Display and save image
# -----------------------------
image.show()  # Display in Colab
image.save("generated_image.png")  # Save as PNG
print("Image saved as generated_image.png")

