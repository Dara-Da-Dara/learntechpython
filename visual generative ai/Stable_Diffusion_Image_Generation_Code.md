
# Stable Diffusion Image Generation – Python Code Guide

This document provides complete, production-ready Python code examples for generating images using **Stable Diffusion** with the Hugging Face **Diffusers** library.

---

## 1. Install Required Libraries

```bash
pip install diffusers transformers accelerate safetensors torch
```

For GPU (recommended):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Text-to-Image Generation (GPU)

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = "A futuristic city at sunset, ultra realistic, cinematic lighting"

image = pipe(
    prompt=prompt,
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

image.save("stable_diffusion_output.png")
```

---

## 3. CPU-Only Text-to-Image Generation

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to("cpu")

prompt = "A watercolor painting of mountains and rivers"

image = pipe(prompt).images[0]
image.save("cpu_output.png")
```

Note: CPU inference is significantly slower.

---

## 4. Image-to-Image Generation (Style Transfer)

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("input.jpg").convert("RGB")
init_image = init_image.resize((512, 512))

prompt = "Turn this image into an oil painting, artistic style"

image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,
    guidance_scale=7.5
).images[0]

image.save("img2img_output.png")
```

---

## 5. Inpainting with Stable Diffusion

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("image.png").convert("RGB")
mask = Image.open("mask.png").convert("RGB")

prompt = "Remove the object and fill the background naturally"

result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    guidance_scale=7.5
).images[0]

result.save("inpaint_output.png")
```

Mask rules:
- White regions are replaced
- Black regions are preserved

---

## 6. Outpainting Concept

Outpainting uses the same inpainting pipeline.

Steps:
1. Expand the image canvas
2. Place the original image in the center
3. Mask the extended area
4. Apply inpainting

---

## 7. Important Parameters

| Parameter | Description |
|---------|-------------|
| prompt | Text description |
| guidance_scale | Prompt adherence (7–9 recommended) |
| num_inference_steps | Image quality vs speed |
| strength | Degree of change (img2img) |
| torch_dtype=float16 | Faster GPU inference |

---

## 8. Recommended System Configuration

- GPU VRAM: 6 GB minimum (8–12 GB recommended)
- Image size: 512 × 512
- Optional: xFormers for memory optimization

---

## 9. Use Cases

- AI art generation
- Image editing tools
- Style transfer applications
- Creative automation pipelines
- Agentic AI workflows

---

## 10. Summary

Stable Diffusion enables high-quality image generation using diffusion models operating in latent space. With text-to-image, image-to-image, and inpainting support, it forms the foundation of modern generative vision systems.
