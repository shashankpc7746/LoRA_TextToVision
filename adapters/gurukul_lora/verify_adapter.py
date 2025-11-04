"""
Quick test to verify the trained LoRA adapter loads correctly
"""
import torch
from pathlib import Path

print("Testing Gurukul LoRA adapter...")
print("="*60)

adapter_path = Path("adapters/gurukul_lora/gurukul_lora.pt")

if not adapter_path.exists():
    print(f"❌ Adapter not found: {adapter_path}")
    exit(1)

print(f"✓ Adapter found: {adapter_path}")
print(f"  Size: {adapter_path.stat().st_size / 1024 / 1024:.2f} MB")

# Load the adapter
checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)

print("\n✓ Adapter loaded successfully!")
print(f"\nContents:")
print(f"  - state_dict: {len(checkpoint['state_dict'])} parameters")
print(f"  - lora_config: {checkpoint['lora_config']}")

# Verify LoRA parameters
total_params = sum(p.numel() for p in checkpoint['state_dict'].values())
print(f"\n  Total LoRA parameters: {total_params:,}")

# Check a few parameter names
print(f"\n  Sample parameter names:")
for i, name in enumerate(list(checkpoint['state_dict'].keys())[:5]):
    print(f"    {i+1}. {name}")

print("\n" + "="*60)
print("✅ Gurukul LoRA adapter is valid and ready to use!")
print("="*60)
