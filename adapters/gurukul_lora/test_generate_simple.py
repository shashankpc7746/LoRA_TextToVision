"""
Test Gurukul LoRA Adapter - Generate Images (Simplified)
Uses cached models from training
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

import torch
from diffusers import DiffusionPipeline
from pathlib import Path
from datetime import datetime

print("="*70)
print("TESTING GURUKUL LORA ADAPTER")
print("="*70)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

output_dir = Path("test_outputs")
output_dir.mkdir(exist_ok=True)

# Load base SDXL pipeline with local_files_only to use cached models
print("\nLoading SDXL pipeline from cache...")
try:
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True  # Use cached files
    ).to(device)
    print("✓ Base pipeline loaded from cache")
except Exception as e:
    print(f"Error loading from cache: {e}")
    print("Trying without local_files_only...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    print("✓ Base pipeline loaded")

# Load LoRA adapter
print("\nLoading Gurukul LoRA adapter...")
adapter_path = "gurukul_lora.pt"

checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)

# Load adapter using load_lora_weights if available
try:
    # Try using the built-in LoRA loading
    pipe.load_lora_weights(".", weight_name="gurukul_lora.pt")
    print("✓ LoRA loaded using load_lora_weights")
except Exception as e:
    print(f"Note: {e}")
    print("Attempting manual LoRA integration...")
    
    # Manual loading
    from peft import inject_adapter_in_model
    lora_config = checkpoint['lora_config']
    pipe.unet = inject_adapter_in_model(lora_config, pipe.unet)
    pipe.unet.load_state_dict(checkpoint['state_dict'], strict=False)
    print("✓ LoRA loaded manually")

# Test prompts - shorter for faster generation
prompts = [
    {
        "prompt": "Traditional Indian Gurukul, ancient learning center, students under banyan tree, guru teaching, warm sunlight, highly detailed",
        "name": "gurukul_01"
    },
    {
        "prompt": "Ancient Gurukul classroom, wooden architecture, Sanskrit texts, oil lamps, peaceful morning",
        "name": "gurukul_02"
    },
    {
        "prompt": "Gurukul courtyard, meditation area, traditional Indian architecture, serene atmosphere",
        "name": "gurukul_03"
    }
]

# Generation settings - faster inference
print(f"\n{'='*70}")
print(f"Generating {len(prompts)} test images (25 steps each)...")
print(f"{'='*70}\n")

for i, test in enumerate(prompts, 1):
    print(f"[{i}/{len(prompts)}] {test['name']}")
    print(f"    Prompt: {test['prompt'][:60]}...")
    
    start = datetime.now()
    
    # Generate image
    with torch.inference_mode():
        image = pipe(
            prompt=test['prompt'],
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=25,  # Faster
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
    
    elapsed = (datetime.now() - start).total_seconds()
    
    # Save image
    filename = f"{test['name']}.png"
    save_path = output_dir / filename
    image.save(save_path)
    
    print(f"    ✓ Generated in {elapsed:.1f}s")
    print(f"    ✓ Saved: {save_path}\n")

print("="*70)
print(f"✅ Testing complete!")
print(f"   Generated {len(prompts)} images")
print(f"   Saved to: {output_dir.absolute()}")
print("="*70)
