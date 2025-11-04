"""
Gurukul LoRA Adapter Training Script - Task 9 Day 1
Train indigenous adapter on curated keyframes for deterministic generation

FIXED: Added time_ids to SDXL UNet forward pass (required for SDXL architecture)
Date: October 25, 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import json
from datetime import datetime
import hashlib
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np


class GurukulKeyframeDataset(Dataset):
    """Dataset for Gurukul keyframe training (50-200 curated images)"""
    
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load image files
        self.image_files = list(self.dataset_path.glob("*.png")) + \
                          list(self.dataset_path.glob("*.jpg"))
        
        # Load captions/prompts
        self.captions = self._load_captions()
        
        print(f"Loaded {len(self.image_files)} keyframes for training")
        
    def _load_captions(self) -> Dict[str, str]:
        """Load caption/prompt for each image"""
        captions = {}
        caption_file = self.dataset_path / "captions.json"
        
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                captions = json.load(f)
        else:
            # Default Gurukul-style captions
            for img_file in self.image_files:
                captions[img_file.name] = "Traditional Indian Gurukul educational scene"
                
        return captions
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        caption = self.captions.get(img_path.name, "Gurukul scene")
        
        return image, caption


class GurukulLoRATrainer:
    """Train indigenous LoRA adapter for Gurukul-style keyframes"""
    
    def __init__(self, 
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 output_dir: str = "adapters/gurukul_lora"):
        
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LoRA configuration (task_type removed - not needed for diffusion models)
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",  # Attention
                "proj_in", "proj_out",  # Projection layers
            ],
            lora_dropout=0.1,
            bias="none"
        )
        
        # Training configuration (optimized for 8GB GPU)
        self.train_config = {
            "num_epochs": 100,
            "batch_size": 1,  # Reduced for 8GB GPU
            "learning_rate": 1e-5,  # Reduced from 1e-4 to prevent NaN loss
            "gradient_accumulation_steps": 1,  # Set to 1 for faster testing (was 8)
            "save_steps": 25,
            "warmup_steps": 10,
            "max_grad_norm": 0.5,  # Reduced from 1.0 for stability
            "use_8bit_adam": True,
            "mixed_precision": "no",  # Disabled FP16 - using FP32 for stability
            "seed": 42,  # Deterministic training
        }
        
        # Metadata for reproducibility
        self.metadata = {
            "model_name": "gurukul_lora",
            "base_model": base_model,
            "lora_config": {
                "r": self.lora_config.r,
                "alpha": self.lora_config.lora_alpha,
                "target_modules": self.lora_config.target_modules,
                "dropout": self.lora_config.lora_dropout
            },
            "training_config": self.train_config,
            "created_at": datetime.now().isoformat(),
        }
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")
        
    def _setup_pipeline(self) -> StableDiffusionXLPipeline:
        """Load SDXL pipeline and apply LoRA with memory optimizations"""
        print(f"Loading base model: {self.base_model}")
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True  # Use cached model, avoid network calls
        )
        
        # Apply LoRA configuration
        pipeline.unet = get_peft_model(pipeline.unet, self.lora_config)
        
        # Memory optimizations for 8GB GPU
        pipeline.enable_attention_slicing(1)  # Reduce memory for attention
        pipeline.enable_vae_slicing()  # Process VAE in slices
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ xFormers memory-efficient attention enabled")
            except:
                print("⚠️  xFormers not available, using default attention")
        
        # Enable gradient checkpointing for UNet
        pipeline.unet.enable_gradient_checkpointing()
        
        pipeline = pipeline.to(self.device)
        
        print("LoRA adapter applied to UNet")
        print(f"Trainable parameters: {self._count_trainable_parameters(pipeline.unet)}")
        
        return pipeline
        
    def _count_trainable_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def train(self, dataset_path: str, num_epochs: Optional[int] = None) -> str:
        """
        Train Gurukul LoRA adapter
        
        Args:
            dataset_path: Path to curated keyframe dataset (50-200 images)
            num_epochs: Number of training epochs (default from config)
            
        Returns:
            Path to saved adapter checkpoint
        """
        if num_epochs:
            self.train_config["num_epochs"] = num_epochs
            
        print(f"\n{'='*60}")
        print("Starting Gurukul LoRA Adapter Training")
        print(f"{'='*60}\n")
        
        # Clear GPU memory cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("✅ GPU memory cache cleared")
        
        # Set deterministic seed
        torch.manual_seed(self.train_config["seed"])
        np.random.seed(self.train_config["seed"])
        
        # Load dataset
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Reduced from 1024 to save memory
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        dataset = GurukulKeyframeDataset(dataset_path, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=0  # Fix: Windows multiprocessing issue, use 0 for single-threaded
        )
        
        # Setup pipeline
        pipeline = self._setup_pipeline()
        optimizer = torch.optim.AdamW(
            pipeline.unet.parameters(),
            lr=self.train_config["learning_rate"]
        )
        
        # Gradient scaler for mixed precision (prevents NaN)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.train_config["mixed_precision"] == "fp16"))
        
        # Training loop
        global_step = 0
        best_loss = float('inf')
        
        # Set VAE and text encoder to eval mode (we're only training UNet)
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        pipeline.text_encoder_2.eval()
        
        for epoch in range(self.train_config["num_epochs"]):
            epoch_loss = 0.0
            pipeline.unet.train()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.train_config['num_epochs']}")
            
            for batch_idx, (images, captions) in enumerate(progress_bar):
                images = images.to(self.device, dtype=torch.float16)  # Match VAE dtype
                
                # Encode images with VAE (FP16) - no gradients needed
                with torch.no_grad():
                    latents = pipeline.vae.encode(images).latent_dist.sample()
                    latents = latents * pipeline.vae.config.scaling_factor
                
                # Convert latents to FP32 for training
                latents = latents.float()
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device
                ).long()
                
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                
                # Encode prompts (ensure captions is a list of strings)
                caption_list = [captions] if isinstance(captions, str) else list(captions)
                
                # SDXL returns: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
                with torch.no_grad():
                    encoder_output = pipeline.encode_prompt(
                        prompt=caption_list,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False
                    )
                
                # Extract prompt embeds and pooled embeds
                if len(encoder_output) == 4:
                    prompt_embeds, _, pooled_prompt_embeds, _ = encoder_output
                else:
                    prompt_embeds = encoder_output[0]
                    pooled_prompt_embeds = encoder_output[2] if len(encoder_output) > 2 else None
                
                # Convert embeddings to FP32 for training
                prompt_embeds = prompt_embeds.float()
                pooled_prompt_embeds = pooled_prompt_embeds.float()
                
                # Predict noise (SDXL requires both text_embeds and time_ids)
                # Create time_ids: [original_size, crops_coords_top_left, target_size]
                # For 512x512 images with no cropping:
                original_size = (512, 512)
                crops_coords = (0, 0)
                target_size = (512, 512)
                
                add_time_ids = torch.tensor([
                    list(original_size) + list(crops_coords) + list(target_size)
                ], device=self.device, dtype=torch.float32).repeat(latents.shape[0], 1)
                
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
                
                # Forward pass through UNet (trainable, FP32)
                noise_pred = pipeline.unet(
                    noisy_latents, 
                    timesteps, 
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"⚠️  NaN loss detected at batch {batch_idx}. Skipping...")
                    optimizer.zero_grad()  # Clear gradients
                    continue
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.train_config["gradient_accumulation_steps"] == 0:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        pipeline.unet.parameters(),
                        self.train_config["max_grad_norm"]
                    )
                    # Step optimizer with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                epoch_loss += loss.item()
                global_step += 1
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Clear CUDA cache every 10 batches to prevent fragmentation
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Save checkpoint
                if global_step % self.train_config["save_steps"] == 0:
                    checkpoint_path = self.output_dir / f"checkpoint_{global_step}.pt"
                    self._save_checkpoint(pipeline, checkpoint_path, epoch, global_step)
                    
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = self.output_dir / "gurukul_lora_best.pt"
                self._save_checkpoint(pipeline, best_model_path, epoch, global_step)
                print(f"✅ Best model saved: {best_model_path}")
                
        # Save final adapter
        final_path = self.output_dir / "gurukul_lora.pt"
        self._save_adapter(pipeline, final_path)
        
        # Save metadata
        self._save_metadata(dataset_path, global_step)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Final adapter saved: {final_path}")
        print(f"Metadata saved: {self.output_dir / 'metadata.json'}")
        print(f"{'='*60}\n")
        
        return str(final_path)
        
    def _save_checkpoint(self, pipeline, checkpoint_path: Path, 
                        epoch: int, global_step: int):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "unet_state_dict": pipeline.unet.state_dict(),
            "lora_config": self.lora_config,
            "train_config": self.train_config,
        }
        torch.save(checkpoint, checkpoint_path)
        
    def _save_adapter(self, pipeline, output_path: Path):
        """Save trained LoRA adapter"""
        # Extract only LoRA parameters
        lora_state_dict = {}
        for name, param in pipeline.unet.named_parameters():
            if "lora" in name.lower():
                lora_state_dict[name] = param.cpu()
                
        adapter_checkpoint = {
            "state_dict": lora_state_dict,
            "lora_config": self.lora_config,
            "base_model": self.base_model,
            "metadata": self.metadata,
        }
        
        torch.save(adapter_checkpoint, output_path)
        print(f"Adapter saved: {output_path} ({self._get_file_size(output_path)})")
        
    def _save_metadata(self, dataset_path: str, total_steps: int):
        """Save training metadata to NAS"""
        # Generate deterministic seed from model state
        model_hash = self._calculate_model_hash()
        
        metadata = {
            **self.metadata,
            "dataset_path": str(dataset_path),
            "dataset_size": len(list(Path(dataset_path).glob("*.png"))) + \
                           len(list(Path(dataset_path).glob("*.jpg"))),
            "total_training_steps": total_steps,
            "model_hash": model_hash,
            
            # Deterministic generation metadata
            "deterministic_config": {
                "seed": self.train_config["seed"],
                "cfg_scale": 7.5,
                "scheduler": "DDIMScheduler",
                "num_inference_steps": 30,
                "tokenizer": "openai/clip-vit-large-patch14"
            },
            
            # KSML lineage
            "ksml_lineage": {
                "parent_models": ["SDXL", "clip-vit-large"],
                "training_dataset": "gurukul_keyframes_v1",
                "adapter_type": "LoRA",
                "rank": self.lora_config.r,
            }
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Metadata saved to NAS: {metadata_path}")
        
    def _calculate_model_hash(self) -> str:
        """Calculate hash of model for lineage tracking"""
        adapter_path = self.output_dir / "gurukul_lora.pt"
        if adapter_path.exists():
            with open(adapter_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        return "not_yet_saved"
        
    def _get_file_size(self, path: Path) -> str:
        """Get human-readable file size"""
        size_bytes = path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


def train_gurukul_adapter(dataset_path: str, 
                          num_epochs: int = 100,
                          output_dir: str = "adapters/gurukul_lora") -> str:
    """
    Convenience function to train Gurukul LoRA adapter
    
    Args:
        dataset_path: Path to curated keyframe dataset (50-200 images)
        num_epochs: Number of training epochs
        output_dir: Output directory for adapter
        
    Returns:
        Path to trained adapter
    """
    trainer = GurukulLoRATrainer(output_dir=output_dir)
    return trainer.train(dataset_path, num_epochs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Gurukul LoRA Adapter")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to curated keyframe dataset")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="adapters/gurukul_lora",
                       help="Output directory")
    
    args = parser.parse_args()
    
    adapter_path = train_gurukul_adapter(
        args.dataset,
        args.num_epochs,
        args.output_dir
    )
    
    print(f"\n✅ Training complete! Adapter saved: {adapter_path}")
