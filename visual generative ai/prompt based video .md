# Prompt-Based Video Generation – Complete Guide with Code

## Overview

Prompt-based video generation is an emerging area of Generative AI where videos are created directly from natural language descriptions. These systems leverage diffusion models, transformer architectures, and multimodal learning to generate realistic or artistic videos.

This document provides:
- Conceptual understanding
- Practical Python and JavaScript code
- Open-source and commercial tools
- Prompt engineering best practices
- Teaching and research-friendly explanations

---

## 1. Stable Video Diffusion (Open Source)

Stable Video Diffusion is an open-source model by Stability AI that supports image-to-video and prompt-guided video generation.

### Installation
### Text-to-Video Generation Code
```bash
pip install diffusers transformers accelerate torch torchvision

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
).to("cuda")

prompt = "A cinematic sunrise over mountains, clouds moving slowly, ultra realistic"

video_frames = pipe(
    prompt=prompt,
    num_frames=24,
    height=576,
    width=1024,
    guidance_scale=7.5
).frames

export_to_video(video_frames, "output_video.mp4", fps=8)

---

### 2. ModelScope Text-to-Video Generation

ModelScope by DAMO Academy provides pretrained text-to-video synthesis models.


pip install modelscope torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_pipe = pipeline(
    task=Tasks.text_to_video_synthesis,
    model="damo-vilab/text-to-video-ms-1.7b"
)

prompt = "A futuristic city with flying cars, neon lights, cyberpunk style"

result = video_pipe(prompt)
video_path = result["video"]

print("Video saved at:", video_path)


### 3. Runway ML – Commercial API-Based Video Generation

Runway ML provides high-quality video generation through paid APIs.

import fetch from "node-fetch";

const response = await fetch(
  "https://api.runwayml.com/v1/text_to_video",
  {
    method: "POST",
    headers: {
      "Authorization": "Bearer YOUR_API_KEY",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      prompt: "A drone shot of ocean waves crashing on rocks",
      duration: 5,
      resolution: "720p"
    })
  }
);

const video = await response.json();
console.log(video);
---
Pika Labs – Prompt-Based Video Generation (Conceptual API)

Pika Labs focuses on creative, stylized video generation.
import requests

url = "https://api.pika.art/generate"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

payload = {
    "prompt": "A magical forest with glowing trees and floating lights",
    "style": "cinematic",
    "duration": 4
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
---

5. Image + Prompt to Video (Industry-Standard Workflow)

Most real-world systems generate video by animating a single image using a text prompt.
from diffusers import StableVideoDiffusionPipeline
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
).to("cuda")

video_frames = pipe(
    image="input_image.png",
    prompt="Camera slowly zooms in, cinematic lighting",
    num_frames=30
).frames
---
## Platform Comparison for Prompt-Based Video Generation

| Platform               | Code Access | Open Source | Best For              |
| ---------------------- | ----------- | ----------- | --------------------- |
| Stable Video Diffusion | Yes         | Yes         | Research and training |
| ModelScope             | Yes         | Yes         | Academia              |
| Runway ML              | API         | No          | Production            |
| Pika Labs              | API         | No          | Creative media        |
| OpenAI Sora            | No          | No          | Future enterprise use |

