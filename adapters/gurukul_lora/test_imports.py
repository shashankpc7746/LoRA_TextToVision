"""
Ultra-minimal LoRA training - no complex imports
"""
import sys
import os

# Suppress warnings before any imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

print("Starting imports...")

import torch
print("✓ torch")
from pathlib import Path
print("✓ pathlib")
from PIL import Image
print("✓ PIL")
import json
print("✓ json")

# Try importing diffusers components one by one
try:
    from diffusers.models import UNet2DConditionModel
    print("✓ UNet2DConditionModel")
except Exception as e:
    print(f"✗ UNet2DConditionModel: {e}")
    sys.exit(1)

try:
    from diffusers.models import AutoencoderKL  
    print("✓ AutoencoderKL")
except Exception as e:
    print(f"✗ AutoencoderKL: {e}")
    sys.exit(1)

try:
    from diffusers.schedulers import DDPMScheduler
    print("✓ DDPMScheduler")
except Exception as e:
    print(f"✗ DDPMScheduler: {e}")
    sys.exit(1)

try:
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    print("✓ transformers")
except Exception as e:
    print(f"✗ transformers: {e}")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model
    print("✓ peft")
except Exception as e:
    print(f"✗ peft: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
print("=" * 60)
print("Starting training setup...")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Clear cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared")

print("\nIf you see this, imports are working!")
print("Exiting test script...")
