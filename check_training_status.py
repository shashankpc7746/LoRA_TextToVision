"""
Quick Training Status Checker
"""
from pathlib import Path
import json
from datetime import datetime

print("="*70)
print("TRAINING STATUS CHECK")
print("="*70)
print()

# Check adapter file
adapter_path = Path("adapters/gurukul_lora.pt")
if adapter_path.exists():
    size_mb = adapter_path.stat().st_size / (1024 * 1024)
    modified = datetime.fromtimestamp(adapter_path.stat().st_mtime)
    print(f"✅ TRAINING COMPLETE!")
    print(f"   File: {adapter_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Modified: {modified}")
else:
    print("⏳ Training not complete yet")
    print(f"   Expected: {adapter_path}")

print()

# Check for checkpoint files
checkpoint_dir = Path("adapters/gurukul_lora")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        print(f"📁 Found {len(checkpoints)} checkpoint(s):")
        for cp in sorted(checkpoints):
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"   • {cp.name} ({size_mb:.2f} MB)")
    else:
        print("📁 No checkpoints found yet")
else:
    print("📁 Checkpoint directory not created yet")

print()

# Check dataset
dataset_path = Path("datasets/gurukul_keyframes")
if dataset_path.exists():
    images = list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpg"))
    captions = dataset_path / "captions.json"
    print(f"✅ Dataset ready: {len(images)} images")
    if captions.exists():
        print(f"✅ Captions: present")
else:
    print("❌ Dataset not found")

print()

# Check if training script has been modified
train_script = Path("adapters/gurukul_lora/train_adapter.py")
if train_script.exists():
    modified = datetime.fromtimestamp(train_script.stat().st_mtime)
    print(f"📝 Training script last modified: {modified}")

print()
print("="*70)
print()

if adapter_path.exists():
    print("🎉 Training is COMPLETE! You can now use the adapter for inference.")
else:
    print("💡 To train: Run START_TRAINING.bat from Windows Explorer")
    print("   (Or run it overnight for 2-3 hours)")
