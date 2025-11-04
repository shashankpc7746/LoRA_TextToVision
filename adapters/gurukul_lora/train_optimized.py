"""
Gurukul LoRA Training - Optimized imports to avoid slow chains
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'  # Disable torch dynamo to skip sympy imports

print("Starting imports (this may take 1-2 minutes on first run)...")

import torch
print("✓ torch")

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
print("✓ basic libraries")

# Import with timeout warning
import sys
print("Loading diffusers (be patient, scanning model files)...", end="", flush=True)

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
print(" ✓")

print("Loading transformers...", end="", flush=True)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
print(" ✓")

print("Loading PEFT...", end="", flush=True)
from peft import LoraConfig, get_peft_model
print(" ✓")

print("\n" + "="*70)
print("ALL IMPORTS SUCCESSFUL!")
print("="*70 + "\n")


class GurukulDataset(Dataset):
    def __init__(self, data_path, size=512):
        self.data_path = Path(data_path)
        self.size = size
        self.images = list(self.data_path.glob("*.png")) + list(self.data_path.glob("*.jpg"))
        
        caption_file = self.data_path / "captions.json"
        if caption_file.exists():
            with open(caption_file) as f:
                self.captions = json.load(f)
        else:
            self.captions = {}
        
        print(f"Dataset: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB").resize((self.size, self.size))
        import torchvision.transforms as T
        img = T.Normalize([0.5], [0.5])(T.ToTensor()(img))
        caption = self.captions.get(self.images[idx].name, "Gurukul scene")
        return img, caption


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    # Dataset
    dataset = GurukulDataset("datasets/gurukul_keyframes", size=512)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Load models
    print("Loading models...")
    print("  VAE (FP32)...", end="", flush=True)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()
    vae.requires_grad_(False)
    print(" ✓")
    
    print("  Text Encoders...", end="", flush=True)
    tok1 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    te1 = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device).eval()
    te1.requires_grad_(False)
    
    tok2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    te2 = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder_2", torch_dtype=torch.float16
    ).to(device).eval()
    te2.requires_grad_(False)
    print(" ✓")
    
    print("  UNet + LoRA...", end="", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet", torch_dtype=torch.float32
    ).to(device)
    
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["to_k", "to_q", "to_v", "to_out.0"], lora_dropout=0.1, bias="none")
    unet = get_peft_model(unet, lora_cfg).train()
    print(f" ✓ ({sum(p.numel() for p in unet.parameters() if p.requires_grad):,} params)")
    
    scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    
    # Training setup
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-5)
    from torch.optim.lr_scheduler import LambdaLR
    sched = LambdaLR(opt, lambda s: min(s / 50, 1.0))
    
    print(f"\nTraining 10 epochs, LR=1e-5 with warmup")
    print("="*70 + "\n")
    
    # Train
    for epoch in range(10):
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/10")
        
        for imgs, caps in pbar:
            imgs = imgs.to(device, dtype=torch.float32)
            
            with torch.no_grad():
                lat = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
                if torch.isnan(lat).any():
                    continue
                
                # Text encode
                t1 = tok1(caps, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
                t2 = tok2(caps, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
                
                e1 = te1(t1, output_hidden_states=True)
                e2 = te2(t2, output_hidden_states=True)
                
                embeds = torch.cat([e1.hidden_states[-2], e2.hidden_states[-2]], dim=-1).float()
                pooled = e2.text_embeds.float()
            
            noise = torch.randn_like(lat)
            ts = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=device).long()
            noisy = scheduler.add_noise(lat, noise, ts)
            
            # Predict
            pred = unet(
                noisy, ts, embeds,
                added_cond_kwargs={
                    "text_embeds": pooled,
                    "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float32)
                }
            ).sample
            
            loss = nn.functional.mse_loss(pred, noise)
            
            if torch.isnan(loss):
                opt.zero_grad()
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{sched.get_last_lr()[0]:.2e}"})
        
        print(f"Epoch {epoch+1} | Avg Loss: {sum(losses)/len(losses):.4f}\n")
    
    # Save
    out = Path("adapters/gurukul_lora")
    out.mkdir(exist_ok=True)
    
    lora_state = {n: p.cpu() for n, p in unet.named_parameters() if "lora" in n.lower()}
    torch.save({"state_dict": lora_state, "lora_config": lora_cfg}, out / "gurukul_lora.pt")
    
    print(f"\n{'='*70}")
    print(f"✅ Training complete! Saved to: {out / 'gurukul_lora.pt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
