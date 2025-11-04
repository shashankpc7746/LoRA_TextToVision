"""
Test Gurukul LoRA Adapter - Generate Images
Tests the trained adapter with various Gurukul-themed prompts
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from pathlib import Path
from datetime import datetime

print("="*70)
print("TESTING GURUKUL LORA ADAPTER")
print("="*70)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

output_dir = Path("adapters/gurukul_lora/test_outputs")
output_dir.mkdir(exist_ok=True, parents=True)

# Load base SDXL pipeline
print("\nLoading SDXL pipeline (this may take a moment)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# Use DDIM scheduler for faster inference
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

print("✓ Base pipeline loaded")

# Load LoRA adapter
print("\nLoading Gurukul LoRA adapter...")
adapter_path = "adapters/gurukul_lora/gurukul_lora.pt"

checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
lora_state = checkpoint['state_dict']

# Load LoRA weights into UNet
from peft import PeftModel, LoraConfig

# Get the LoRA config
lora_config = checkpoint['lora_config']

# Apply LoRA to UNet
from peft import inject_adapter_in_model
pipe.unet = inject_adapter_in_model(lora_config, pipe.unet)

# Load the trained weights
pipe.unet.load_state_dict(lora_state, strict=False)

print("✓ Gurukul LoRA adapter loaded")

# Test prompts
prompts = [
    {
        "prompt": "Traditional Indian Gurukul, ancient learning center, students sitting under a banyan tree, guru teaching, warm sunlight, spiritual atmosphere, highly detailed",
        "name": "gurukul_traditional"
    },
    {
        "prompt": "Ancient Gurukul classroom, wooden architecture, Sanskrit texts on walls, oil lamps, students with traditional attire, peaceful morning scene",
        "name": "gurukul_classroom"
    },
    {
        "prompt": "Gurukul courtyard, stone pathways, meditation area, traditional Indian architecture, lush green gardens, serene atmosphere",
        "name": "gurukul_courtyard"
    },
    {
        "prompt": "Young students learning in Gurukul, traditional education, guru and shishya, ancient Indian wisdom, detailed illustration",
        "name": "gurukul_learning"
    },
    {
        "prompt": "Gurukul at sunset, traditional Indian educational institution, beautiful architecture, golden hour lighting, peaceful scene",
        "name": "gurukul_sunset"
    }
]

# Generation settings
generator = torch.Generator(device=device).manual_seed(42)  # For reproducibility

print(f"\n{'='*70}")
print(f"Generating {len(prompts)} test images...")
print(f"{'='*70}\n")

for i, test in enumerate(prompts, 1):
    print(f"[{i}/{len(prompts)}] Generating: {test['name']}")
    print(f"    Prompt: {test['prompt'][:70]}...")
    
    # Generate image
    with torch.inference_mode():
        image = pipe(
            prompt=test['prompt'],
            negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512
        ).images[0]
    
    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test['name']}_{timestamp}.png"
    save_path = output_dir / filename
    image.save(save_path)
    
    print(f"    ✓ Saved: {save_path}")
    print()

print("="*70)
print("✅ Testing complete!")
print(f"   Generated {len(prompts)} images")
print(f"   Saved to: {output_dir}")
print("="*70)

# Also generate a comparison - base model vs LoRA
print("\n" + "="*70)
print("BONUS: Generating comparison (base model vs LoRA)")
print("="*70)

comparison_prompt = "Traditional Gurukul educational scene, ancient Indian learning center"

# First, unload LoRA (generate with base model)
print("\nGenerating with base SDXL (no LoRA)...")
pipe_base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

with torch.inference_mode():
    img_base = pipe_base(
        prompt=comparison_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(100),
        height=512,
        width=512
    ).images[0]

img_base.save(output_dir / "comparison_base_model.png")
print("✓ Base model image saved")

# Generate with LoRA
print("\nGenerating with Gurukul LoRA...")
with torch.inference_mode():
    img_lora = pipe(
        prompt=comparison_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(100),
        height=512,
        width=512
    ).images[0]

img_lora.save(output_dir / "comparison_with_lora.png")
print("✓ LoRA model image saved")

print("\n" + "="*70)
print("Compare these files to see the difference:")
print(f"  - Base model: {output_dir / 'comparison_base_model.png'}")
print(f"  - With LoRA:  {output_dir / 'comparison_with_lora.png'}")
print("="*70)
