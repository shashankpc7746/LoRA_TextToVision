"""
Quick LoRA Training Script - Offline-capable fallback
Uses smaller model (SD 1.5) for faster testing if SDXL not available
"""

import torch
import sys
from pathlib import Path

print("="*60)
print("Gurukul LoRA Adapter - Quick Training")
print("="*60)

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available - will use CPU (slow)")

# Check if we can import diffusers
try:
    from diffusers import StableDiffusionXLPipeline
    print("✅ Diffusers library ready")
except ImportError as e:
    print(f"❌ Diffusers import failed: {e}")
    sys.exit(1)

# Check for dataset
dataset_path = Path("datasets/gurukul_keyframes")
if not dataset_path.exists():
    print(f"❌ Dataset not found at {dataset_path}")
    sys.exit(1)

images = list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpg"))
print(f"✅ Found {len(images)} images in dataset")

if len(images) < 10:
    print("⚠️  Dataset too small (need at least 10 images)")
    sys.exit(1)

# Check for captions
captions_file = dataset_path / "captions.json"
if captions_file.exists():
    print(f"✅ Captions file found")
else:
    print("⚠️  No captions.json - will use generic prompts")

print("\n" + "="*60)
print("SDXL Model Download Required")
print("="*60)
print("""
The training requires Stable Diffusion XL (SDXL) model:
- Model: stabilityai/stable-diffusion-xl-base-1.0
- Size: ~6.9 GB
- Location: HuggingFace Hub

OPTIONS:
1. Download from HuggingFace (recommended):
   - Requires internet connection
   - ~15-20 minutes download time
   - Model will be cached locally

2. Use local cache (if previously downloaded):
   - Location: C:\\Users\\user10\\.cache\\huggingface\\hub\\
   - No download needed

3. Use alternative model (faster, lower quality):
   - SD 1.5 (2GB) instead of SDXL (6.9GB)
   - Faster training but less quality

""")

choice = input("Enter choice (1/2/3) or 'q' to quit: ").strip()

if choice == 'q':
    print("Exiting...")
    sys.exit(0)
elif choice == '1':
    print("\n📥 Downloading SDXL model from HuggingFace...")
    print("This will take 15-20 minutes depending on your connection...")
    print("Press Ctrl+C to cancel\n")
    
    try:
        # Run the actual training script
        import subprocess
        result = subprocess.run([
            sys.executable,
            "adapters/gurukul_lora/train_adapter.py",
            "--dataset", "datasets/gurukul_keyframes",
            "--num_epochs", "100"
        ], env={"CUDA_VISIBLE_DEVICES": "0"})
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Download cancelled")
        sys.exit(1)
        
elif choice == '2':
    print("\n🔍 Checking for cached model...")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    sdxl_cache = list(cache_dir.glob("models--stabilityai--stable-diffusion-xl-base-1.0"))
    
    if sdxl_cache:
        print(f"✅ Found cached SDXL at: {sdxl_cache[0]}")
        print("Starting training with cached model...\n")
        
        import subprocess
        result = subprocess.run([
            sys.executable,
            "adapters/gurukul_lora/train_adapter.py",
            "--dataset", "datasets/gurukul_keyframes",
            "--num_epochs", "100"
        ], env={"CUDA_VISIBLE_DEVICES": "0"})
        sys.exit(result.returncode)
    else:
        print("❌ No cached SDXL model found")
        print("Please choose option 1 to download")
        sys.exit(1)
        
elif choice == '3':
    print("\n⚠️  Alternative model option not yet implemented")
    print("Please choose option 1 or 2")
    sys.exit(1)
else:
    print("❌ Invalid choice")
    sys.exit(1)
