"""
Simplified Gurukul LoRA Training - Minimal dependencies
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TensorFlow warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import sys

# Import only what we need from diffusers
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model


class SimpleKeyframeDataset(Dataset):
    def __init__(self, dataset_path, size=512):
        self.dataset_path = Path(dataset_path)
        self.size = size
        
        # Load images
        self.image_files = list(self.dataset_path.glob("*.png")) + \
                          list(self.dataset_path.glob("*.jpg"))
        
        # Load captions
        caption_file = self.dataset_path / "captions.json"
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                self.captions = json.load(f)
        else:
            self.captions = {}
            
        print(f"✅ Loaded {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size))
        
        # Convert to tensor
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = transform(image)
        
        caption = self.captions.get(img_path.name, "Gurukul scene")
        return image, caption


def train_simple():
    print("\n" + "="*60)
    print("Simplified Gurukul LoRA Training")
    print("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared")
    
    # Load dataset
    dataset = SimpleKeyframeDataset("datasets/gurukul_keyframes", size=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    print("\nLoading SDXL components...")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float32  # Try FP32 instead of FP16 for stability
    )
    vae = vae.to(device)
    vae.eval()
    
    # Load text encoders
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer"
    )
    text_encoder_1 = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    text_encoder_1 = text_encoder_1.to(device)
    text_encoder_1.eval()
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder_2",
        torch_dtype=torch.float16
    )
    text_encoder_2 = text_encoder_2.to(device)
    text_encoder_2.eval()
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        torch_dtype=torch.float32  # Train in FP32
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet = unet.to(device)
    unet.train()
    
    print(f"✅ Models loaded")
    print(f"Trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
    
    # Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler"
    )
    
    # Optimizer with much lower learning rate
    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6)  # Even lower!
    
    print("\nStarting training...")
    print(f"Learning rate: 5e-6 (very conservative to prevent NaN)")
    num_epochs = 10
    
    # Learning rate warmup scheduler
    from torch.optim.lr_scheduler import LambdaLR
    def warmup_lambda(step):
        warmup_steps = 100
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        valid_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, captions) in enumerate(progress_bar):
            images = images.to(device, dtype=torch.float32)  # Match VAE dtype (FP32)
            
            # Check for NaN/Inf in input images
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"⚠️  Invalid values in input images at batch {batch_idx}, skipping...")
                continue
            
            # Encode image with VAE (no grad)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                
                # Check for NaN in latents
                if torch.isnan(latents).any():
                    print(f"⚠️  NaN in VAE latents at batch {batch_idx}, skipping...")
                    continue
                
                latents = latents * vae.config.scaling_factor
                
                # Check again after scaling
                if torch.isnan(latents).any():
                    print(f"⚠️  NaN after VAE scaling at batch {batch_idx}, skipping...")
                    continue
            
            # Add noise
            noise = torch.randn_like(latents)
            
            # Check for NaN in noise
            if torch.isnan(noise).any():
                print(f"⚠️  NaN in noise at batch {batch_idx}, skipping...")
                continue
                
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                     (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Check for NaN after adding noise
            if torch.isnan(noisy_latents).any():
                print(f"⚠️  NaN in noisy_latents at batch {batch_idx}, skipping...")
                continue
            
            # Encode text (no grad)
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
                encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
                encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
                
                prompt_embeds = torch.cat([
                    encoder_output_1.hidden_states[-2],
                    encoder_output_2.hidden_states[-2]
                ], dim=-1).float()
                
                pooled_prompt_embeds = encoder_output_2.text_embeds.float()
            
            # Create time_ids for SDXL
            add_time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], 
                                       device=device, dtype=torch.float32)
            
            # Predict noise
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
            
            noise_pred = unet(noisy_latents, timesteps, prompt_embeds,
                            added_cond_kwargs=added_cond_kwargs).sample
            
            # Calculate loss
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            if torch.isnan(loss):
                print(f"⚠️  NaN loss at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.5)
            optimizer.step()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            valid_batches += 1
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Clear cache every 10 batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / max(valid_batches, 1)  # Avoid division by zero
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f} (valid batches: {valid_batches}/{len(dataloader)})")
    
    # Save adapter
    output_dir = Path("adapters/gurukul_lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            lora_state_dict[name] = param.cpu()
    
    torch.save({
        "state_dict": lora_state_dict,
        "lora_config": lora_config
    }, output_dir / "gurukul_lora.pt")
    
    print(f"\n✅ Training complete! Adapter saved to {output_dir / 'gurukul_lora.pt'}")


if __name__ == "__main__":
    try:
        train_simple()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
