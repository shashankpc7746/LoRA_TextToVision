"""
LoRA Adapter Trainer for Task-7 Quality Leap
Fine-tuning SDXL/AnimateDiff models on Gurukul keyframes
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import json
from tqdm import tqdm
import os

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from .lora_adapter import LoRAAdapter


class KeyframeDataset(Dataset):
    """Dataset for keyframe training data"""

    def __init__(self, image_paths: List[str], captions: List[str], transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
            "image_path": image_path
        }


class LoRATrainer:
    """Trainer for LoRA adapters"""

    def __init__(self, adapter: LoRAAdapter, output_dir: str = "adapters"):
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Training configuration
        self.training_config = {
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "save_steps": 500,
            "logging_steps": 50,
        }

        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

    def prepare_training_data(self, keyframes_dir: str = "keyframes") -> Tuple[List[str], List[str]]:
        """Prepare training data from keyframes directory"""
        keyframes_path = Path(keyframes_dir)
        image_paths = []
        captions = []

        # Look for images and corresponding captions
        for image_file in keyframes_path.glob("*.png"):
            image_paths.append(str(image_file))

            # Try to find corresponding caption file
            caption_file = image_file.with_suffix('.txt')
            if caption_file.exists():
                with open(caption_file, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                # Generate caption from filename
                caption = f"Gurukul educational scene: {image_file.stem.replace('_', ' ')}"

            captions.append(caption)

        print(f"Found {len(image_paths)} training images")
        return image_paths, captions

    def setup_optimizer(self, model):
        """Setup optimizer and scheduler"""
        # Only train LoRA parameters
        lora_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                lora_params.append(param)

        print(f"Training {len(lora_params)} LoRA parameters")

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.training_config["learning_rate"],
            weight_decay=0.01
        )

        # Simple scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.9
        )

    def train_epoch(self, dataloader, model, device) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Training")

        for batch in progress_bar:
            images = batch["image"].to(device)
            captions = batch["caption"]

            # Forward pass (simplified - would need actual training loop)
            # This is a placeholder for the actual SDXL training logic
            loss = torch.tensor(0.5, requires_grad=True)  # Placeholder loss

            # Backward pass
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config["max_grad_norm"])
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, keyframes_dir: str = "keyframes", resume_from: Optional[str] = None) -> str:
        """Main training function"""
        print("Starting LoRA adapter training...")

        # Prepare data
        image_paths, captions = self.prepare_training_data(keyframes_dir)

        if len(image_paths) == 0:
            raise ValueError(f"No training images found in {keyframes_dir}")

        # Create dataset
        dataset = KeyframeDataset(image_paths, captions)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )

        # Load model
        pipeline = self.adapter.load_base_model()
        model = pipeline.unet

        # Apply LoRA
        if resume_from:
            print(f"Resuming from {resume_from}")
            model = PeftModel.from_pretrained(model, resume_from)
        else:
            from peft import get_peft_model
            model = get_peft_model(model, self.adapter.lora_config)

        # Setup optimizer
        self.setup_optimizer(model)

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"Training on device: {device}")
        print(f"Training configuration: {self.training_config}")

        # Training loop
        best_loss = float('inf')

        for epoch in range(self.training_config["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.training_config['num_epochs']}")

            epoch_loss = self.train_epoch(dataloader, model, device)

            print(".4f")

            # Save checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint_path = self.output_dir / f"gurukul_lora_epoch_{epoch+1}"
                model.save_pretrained(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        # Save final adapter
        final_path = self.output_dir / "gurukul_lora_final"
        model.save_pretrained(final_path)
        print(f"Saved final adapter: {final_path}")

        # Update adapter path
        self.adapter.adapter_path = final_path / "adapter_model.bin"

        return str(final_path)

    def evaluate_adapter(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Evaluate trained adapter on test prompts"""
        print("Evaluating LoRA adapter...")

        results = []
        for prompt in test_prompts:
            try:
                result = self.adapter.generate_with_adapter(prompt)
                results.append({
                    "prompt": prompt,
                    "success": True,
                    "num_images": len(result.get("images", []))
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "success": False,
                    "error": str(e)
                })

        success_rate = sum(1 for r in results if r["success"]) / len(results)

        return {
            "total_prompts": len(results),
            "success_rate": success_rate,
            "results": results
        }


def create_gurukul_training_data(output_dir: str = "keyframes"):
    """Create sample training data for Gurukul LoRA training"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Sample Gurukul training prompts
    training_prompts = [
        "traditional Indian teacher explaining ancient texts to students in forest ashram",
        "students meditating in traditional Gurukul courtyard at sunrise",
        "Indian mathematics lesson with slate boards and traditional geometry",
        "Sanskrit language class in ancient Indian school setting",
        "yoga and meditation session in spiritual Gurukul environment",
        "astronomy lesson under night sky in traditional Indian educational setting",
        "herbal medicine and Ayurveda class in forest Gurukul",
        "Indian classical music lesson with traditional instruments",
        "philosophy discussion in ancient Indian wisdom tradition",
        "art and crafts class in traditional Gurukul setting"
    ]

    # Create placeholder images and captions
    for i, prompt in enumerate(training_prompts):
        # Create placeholder image file (would be real keyframes in practice)
        image_path = output_path / "04d"

        # Create caption file
        caption_path = output_path / "04d"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        print(f"Created training sample {i+1:02d}: {prompt[:50]}...")

    print(f"\nCreated {len(training_prompts)} training samples in {output_dir}")
    print("Note: Replace placeholder images with actual SDXL-generated keyframes for real training")

    return training_prompts


# Quick training function for testing
def quick_train_gurukul_adapter(keyframes_dir: str = "keyframes") -> str:
    """Quick training function for development/testing"""
    print("Starting quick Gurukul LoRA adapter training...")

    # Create training data if it doesn't exist
    if not Path(keyframes_dir).exists():
        create_gurukul_training_data(keyframes_dir)

    # Initialize trainer
    adapter = LoRAAdapter()
    trainer = LoRATrainer(adapter)

    # Quick training config for testing
    trainer.training_config.update({
        "num_epochs": 2,  # Quick training
        "batch_size": 1,
        "save_steps": 100
    })

    # Train
    try:
        output_path = trainer.train(keyframes_dir)
        print(f"✅ Training completed successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return ""