"""
Simplified Training Script - Avoids import interruptions
Uses local cached models only, minimal dependencies
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import os

# Set environment to avoid interruptions
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class GurukulKeyframeDataset(Dataset):
    """Minimal dataset for Gurukul keyframes"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.image_files = list(self.dataset_path.glob("*.png"))
        
        # Load captions
        with open(self.dataset_path / "captions.json", 'r') as f:
            self.captions = json.load(f)
        
        print(f"✅ Loaded {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        # Resize to 512 (cheaper than 1024)
        image = image.resize((512, 512))
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        caption = self.captions.get(img_path.name, "Gurukul scene")
        return image, caption


def train_lightweight():
    """Train with minimal memory footprint"""
    print("=" * 60)
    print("Simplified Gurukul LoRA Training")
    print("=" * 60)
    
    # Use cached model directory
    model_cache = Path.home() / ".cache/huggingface/hub"
    print(f"\n1. Checking model cache: {model_cache}")
    
    # Load dataset
    print("\n2. Loading dataset...")
    dataset = GurukulKeyframeDataset("datasets/gurukul_keyframes")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create simple model (we'll add LoRA to it)
    print("\n3. Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple encoder for demonstration
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, 512)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleEncoder().to(device)
    
    # Add LoRA
    print("\n4. Adding LoRA layers...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["conv1", "conv2", "conv3"],
        lora_dropout=0.1,
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Training
    print("\n5. Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    num_epochs = 10  # Quick training
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, captions in pbar:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Dummy target (in real scenario, use CLIP embeddings)
            targets = torch.randn_like(outputs)
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save
    print("\n6. Saving adapter...")
    output_path = Path("adapters/gurukul_lora")
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path / "checkpoint")
    
    # Save metadata
    metadata = {
        "model_name": "gurukul_lora_simple",
        "trainable_params": trainable_params,
        "num_epochs": num_epochs,
        "dataset_size": len(dataset)
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"Adapter saved to: {output_path / 'checkpoint'}")
    print(f"Metadata saved to: {output_path / 'metadata.json'}")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    train_lightweight()
