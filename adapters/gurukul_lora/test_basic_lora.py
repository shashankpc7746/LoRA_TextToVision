"""
Test if basic LoRA training works AT ALL with a tiny fake model
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

print("Testing basic LoRA training...")

# Create a tiny fake UNet-like model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(64, 64)
        self.to_k = nn.Linear(64, 64)
        self.to_v = nn.Linear(64, 64)
        self.to_out = nn.Sequential(nn.Linear(64, 64))
        
    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        out = self.to_out(q + k + v)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create model
model = TinyModel().to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
print(f"✅ LoRA applied, trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Try training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(5):
    # Fake data
    x = torch.randn(2, 64, device=device)
    target = torch.randn(2, 64, device=device)
    
    # Forward
    output = model(x)
    loss = nn.functional.mse_loss(output, target)
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Step {step+1}: loss = {loss.item():.4f}")
    
    if torch.isnan(loss):
        print("❌ NaN detected!")
        break
else:
    print("\n✅ Basic LoRA training works! No NaN issues.")
    print("The problem is specific to SDXL/diffusers, not LoRA itself.")
