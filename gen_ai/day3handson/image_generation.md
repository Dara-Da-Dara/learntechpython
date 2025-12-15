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
