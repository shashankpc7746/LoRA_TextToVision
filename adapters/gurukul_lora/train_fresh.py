"""
Gurukul LoRA Training - Fresh Start with All Fixes
October 29, 2025

Key fixes applied:
- SDXL time_ids support
- Windows DataLoader compatibility
- VAE in FP32 for stability
- Conservative learning rate with warmup
- Proper dtype handling
- Memory optimizations
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import diffusers components
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

print("="*70)
print("GURUKUL LORA TRAINING - FRESH START")
print("="*70)


class GurukulDataset(Dataset):
    """Simple dataset for Gurukul keyframes"""
    
    def __init__(self, data_path, size=512):
        self.data_path = Path(data_path)
        self.size = size
        
        # Load images
        self.images = list(self.data_path.glob("*.png")) + list(self.data_path.glob("*.jpg"))
        
        # Load captions
        caption_file = self.data_path / "captions.json"
        if caption_file.exists():
            with open(caption_file) as f:
                self.captions = json.load(f)
        else:
            self.captions = {img.name: "Traditional Gurukul scene" for img in self.images}
        
        print(f"✅ Dataset: {len(self.images)} images from {data_path}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Load and resize
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size, self.size), Image.LANCZOS)
        
        # To tensor [-1, 1]
        import torchvision.transforms as T
        img = T.ToTensor()(img)
        img = T.Normalize([0.5], [0.5])(img)
        
        caption = self.captions.get(img_path.name, "Gurukul scene")
        return img, caption


def main():
    # Config
    DATASET_PATH = "datasets/gurukul_keyframes"
    OUTPUT_DIR = Path("adapters/gurukul_lora")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    EPOCHS = 10
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5  # Conservative
    IMAGE_SIZE = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Clear GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_free = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU Memory: {mem_free:.1f} GB")
    
    # Load dataset
    print("\n📁 Loading dataset...")
    dataset = GurukulDataset(DATASET_PATH, size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Load models
    print("\n🔧 Loading SDXL models (this may take a moment)...")
    
    print("  Loading VAE (FP32 for stability)...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    print("  Loading Text Encoder 1...")
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer"
    )
    text_encoder_1 = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder",
        torch_dtype=torch.float16
    ).to(device)
    text_encoder_1.eval()
    text_encoder_1.requires_grad_(False)
    
    print("  Loading Text Encoder 2...")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder_2",
        torch_dtype=torch.float16
    ).to(device)
    text_encoder_2.eval()
    text_encoder_2.requires_grad_(False)
    
    print("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        torch_dtype=torch.float32
    ).to(device)
    
    print("  Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"  ✅ Models loaded! Trainable params: {trainable_params:,}")
    
    # Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler"
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    
    # LR scheduler with warmup
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        warmup = 50
        if step < warmup:
            return step / warmup
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print(f"   Learning rate: {LEARNING_RATE} (with 50-step warmup)")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("="*70)
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (images, captions) in enumerate(pbar):
            images = images.to(device, dtype=torch.float32)
            
            # Encode images with VAE
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Check for NaN
                if torch.isnan(latents).any():
                    print(f"\n⚠️  Warning: NaN in latents at batch {batch_idx}, skipping")
                    continue
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            with torch.no_grad():
                # Tokenize
                tokens_1 = tokenizer_1(
                    captions, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)
                
                tokens_2 = tokenizer_2(
                    captions, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)
                
                # Encode
                enc_1 = text_encoder_1(tokens_1, output_hidden_states=True)
                enc_2 = text_encoder_2(tokens_2, output_hidden_states=True)
                
                # Concatenate embeddings
                prompt_embeds = torch.cat([
                    enc_1.hidden_states[-2],
                    enc_2.hidden_states[-2]
                ], dim=-1).to(torch.float32)
                
                pooled_embeds = enc_2.text_embeds.to(torch.float32)
            
            # Create time_ids for SDXL
            time_ids = torch.tensor(
                [[IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE]],
                device=device, dtype=torch.float32
            )
            
            # Predict noise
            model_pred = unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids}
            ).sample
            
            # Compute loss
            loss = nn.functional.mse_loss(model_pred, noise)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\n⚠️  Warning: NaN loss at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track
            epoch_loss += loss.item()
            valid_batches += 1
            
            # Update progress
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Clear cache periodically
            if batch_idx % 10 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
        
        # Epoch summary
        avg_loss = epoch_loss / max(valid_batches, 1)
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | Valid batches: {valid_batches}/{len(dataloader)}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
            lora_state = {}
            for name, param in unet.named_parameters():
                if "lora" in name.lower():
                    lora_state[name] = param.cpu()
            
            torch.save({
                "epoch": epoch + 1,
                "state_dict": lora_state,
                "lora_config": lora_config,
                "loss": avg_loss
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path.name}")
    
    # Save final model
    print("\n" + "="*70)
    print("💾 Saving final model...")
    
    final_path = OUTPUT_DIR / "gurukul_lora.pt"
    lora_state = {}
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            lora_state[name] = param.cpu()
    
    torch.save({
        "state_dict": lora_state,
        "lora_config": lora_config,
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "training_config": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "dataset_size": len(dataset)
        }
    }, final_path)
    
    print(f"✅ Training complete! Model saved to: {final_path}")
    print(f"   File size: {final_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
