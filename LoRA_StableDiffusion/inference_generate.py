import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPTokenizer, CLIPTextModel
import os
from PIL import Image

# ========= Config ========= #
base_model_id = "CompVis/stable-diffusion-v1-4"
lora_weights_path = "./lora_output"  # Trained adapter directory
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

prompts = [
    # "A futuristic Indian classroom with holographic teacher",
    # "A child interacting with a virtual blackboard",
    # "A glowing Sanskrit scroll floating in the air",
    # "A digital sage explaining Vedas through light beams",
    # "Fusion of ancient India and future technology",
    "AI robot teaching students in a village school",
    "A traditional Gurukul reimagined with VR and AR",
    "Students learning under a hologram tree",
    "A child meditating in a virtual garden",
    "Knowledge as light in a dark classroom"
]

# ========= Load Base Pipeline ========= #
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32
).to("cuda")

# ========= Inject LoRA Adapter ========= #
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights_path)
pipe.unet = pipe.unet.to(torch.float32)

# ========= Inference Loop ========= #
for i, prompt in enumerate(prompts):
    print(f"🔹 Generating image for: {prompt}")
    image = pipe(prompt).images[0]
    image.save(os.path.join(output_dir, f"image_{i+1}.png"))

print(f"\n✅ All images saved to: {output_dir}")
