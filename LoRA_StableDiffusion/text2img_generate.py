from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image

# ✅ 1. Use a highly prompt-sensitive model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

# ✅ 2. Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None  # Disable for full control
).to("cuda")

# ✅ 3. Prompt and negative prompt
prompt = (
    "A Tiger drinking water in a pond  under a shaddy tree"
)

negative_prompt = (
    "blurry, distorted, cartoonish, extra limbs, unrealistic, low-res, dark, deformed"
)

# ✅ 4. Generate image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=9.5,
    num_inference_steps=60
).images[0]

# ✅ 5. Save result to a fixed file (overwrite each time)
os.makedirs("results", exist_ok=True)
output_path = "results/generated_image.png"
image.save(output_path)

print(f"✅ Image generated and saved at: {output_path}")
