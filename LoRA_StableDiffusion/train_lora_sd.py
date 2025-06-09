# train_lora_sd.py

import argparse
import json
import logging
import math
import os
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

# Setup logging
logger = get_logger(__name__)

def main():
    # --- 1. ARGUMENTS AND CONFIGURATION ---
    # In a real script, you'd use argparse here. For simplicity, we'll hardcode the values.
    # This is the base model we are fine-tuning
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    # This is the folder containing our training images and the captions file
    train_data_dir = "./LoRA_StableDiffusion/training_data" 
    # This is the trigger phrase we are teaching the model
    instance_prompt = "a photo in Gurukul Hologram style" 
    # Where to save the final LoRA adapter
    output_dir = "./LoRA_StableDiffusion/gurukul_hologram_lora"
    
    # Training parameters
    resolution = 512
    train_batch_size = 1
    num_train_epochs = 100 # We train for more epochs on a small dataset
    learning_rate = 1e-4
    
    # Initialize accelerator for distributed training / mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16", # Use fp16 for faster training and less memory
    )

    # --- 2. DATASET PREPARATION ---
    print("🔄 Loading and preprocessing dataset...")
    
    # Define image transformations
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [image_transforms(image) for image in images]
        # For now, we are not using the text captions in this simple script, 
        # but this is where you would tokenize them.
        return examples

    # Load the dataset using the 'imagefolder' builder
    # It automatically finds the images and the `metadata.jsonl` or `captions.jsonl` file
    # Note: For this to work, your captions file MUST be named "metadata.jsonl"
    # Please rename `captions.jsonl` to `metadata.jsonl`
    
    # Let's create a dummy metadata.jsonl for demonstration if it doesn't exist.
    metadata_path = os.path.join(train_data_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        print(f"⚠️ 'metadata.jsonl' not found. Creating a dummy file.")
        print("Please rename your `captions.jsonl` to `metadata.jsonl` and place it inside the `training_data` folder.")
        # Create dummy metadata if it doesn't exist
        with open(metadata_path, "w") as f:
            for i in range(1, 11): # Assuming 10 images
                if os.path.exists(os.path.join(train_data_dir, "gurukul_hologram_style", f"{i}.png")):
                    record = {"file_name": f"gurukul_hologram_style/{i}.png", "text": instance_prompt}
                    f.write(json.dumps(record) + "\n")

    train_dataset = load_dataset(
        "imagefolder",
        data_dir=train_data_dir,
        split="train"
    )

    # Preprocess the dataset
    train_dataset = train_dataset.with_transform(preprocess_train)

    # Create a dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )

    print("✅ Dataset loaded successfully!")
    print(f"   Number of training examples: {len(train_dataset)}")
    
    # --- The rest of the script (model loading, training loop) will go here ---
    print("\n--- Script ended after data loading. Next step is to load models and train. ---")


if __name__ == "__main__":
    main()

