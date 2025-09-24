"""
LoRA Adapter for Task-7 Quality Leap
Fine-tuned LoRA adapters for SDXL/AnimateDiff models
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model, PeftModel


class LoRAAdapter:
    """LoRA adapter for SDXL/AnimateDiff fine-tuning"""

    def __init__(self, base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.base_model_path = base_model_path
        self.adapter_path = Path("adapters/gurukul_lora.pt")
        self.adapter_path.parent.mkdir(exist_ok=True)

        # LoRA configuration for SDXL
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",  # Attention layers
                "proj_in", "proj_out",  # Feed-forward layers
                "conv1", "conv2",  # Convolutional layers
            ],
            lora_dropout=0.1,
            bias="none",
        )

        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.is_loaded = False

    def load_base_model(self) -> StableDiffusionXLPipeline:
        """Load base SDXL model"""
        if self.pipeline is None:
            print("Loading SDXL base model...")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
        return self.pipeline

    def apply_lora_adapter(self) -> StableDiffusionXLPipeline:
        """Apply LoRA adapter to base model"""
        pipeline = self.load_base_model()

        if self.adapter_path.exists():
            print(f"Loading existing LoRA adapter: {self.adapter_path}")
            # Load the LoRA weights
            state_dict = torch.load(self.adapter_path, map_location="cpu")

            # Apply LoRA to the model
            pipeline.unet = PeftModel.from_pretrained(
                pipeline.unet, self.adapter_path.parent / "gurukul_lora"
            )
        else:
            print("No LoRA adapter found, using base model")
            # Apply LoRA configuration to model
            pipeline.unet = get_peft_model(pipeline.unet, self.lora_config)

        self.is_loaded = True
        return pipeline

    def save_adapter(self, output_path: Optional[str] = None):
        """Save the trained LoRA adapter"""
        if output_path is None:
            output_path = str(self.adapter_path)

        if self.pipeline is not None and hasattr(self.pipeline.unet, 'save_pretrained'):
            print(f"Saving LoRA adapter to {output_path}")
            self.pipeline.unet.save_pretrained(output_path)
        else:
            print("Warning: No trained adapter to save")

    def generate_with_adapter(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using LoRA adapter"""
        if not self.is_loaded:
            self.apply_lora_adapter()

        # Default generation parameters
        defaults = {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024,
        }
        defaults.update(kwargs)

        print(f"Generating with LoRA adapter: {prompt[:50]}...")

        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                **defaults
            )

        return {
            "images": result.images,
            "nsfw_content_detected": result.nsfw_content_detected if hasattr(result, 'nsfw_content_detected') else [],
            "prompt": prompt,
            "parameters": defaults
        }


class GurukulLoRA:
    """Specialized LoRA adapter for Gurukul-themed content"""

    def __init__(self):
        self.adapter = LoRAAdapter()
        self.gurukul_prompts = [
            "traditional Indian classroom with students and teacher",
            "ancient Gurukul school in forest setting",
            "Indian students learning mathematics with slate",
            "teacher explaining Sanskrit texts to students",
            "meditation session in traditional Indian school",
            "students practicing yoga in Gurukul courtyard",
            "Indian classical music lesson in traditional setting",
            "astronomy lesson under night sky in ancient India",
            "herbal medicine class in traditional Gurukul",
            "Indian philosophy discussion in forest ashram"
        ]

    def is_trained(self) -> bool:
        """Check if Gurukul LoRA adapter is trained"""
        return (self.adapter.adapter_path.parent / "gurukul_lora").exists()

    def load_gurukul_adapter(self) -> StableDiffusionXLPipeline:
        """Load Gurukul-specific LoRA adapter"""
        return self.adapter.apply_lora_adapter()

    def generate_gurukul_content(self, custom_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Generate Gurukul-themed content with LoRA adapter"""
        if custom_prompt:
            prompt = f"{custom_prompt}, traditional Indian Gurukul setting, educational, spiritual, ancient wisdom"
        else:
            # Use random Gurukul prompt
            import random
            base_prompt = random.choice(self.gurukul_prompts)
            prompt = f"{base_prompt}, highly detailed, traditional Indian art style, educational atmosphere"

        return self.adapter.generate_with_adapter(prompt, **kwargs)


# Global instances
_lora_adapter = None
_gurukul_lora = None


def get_lora_adapter() -> LoRAAdapter:
    """Get global LoRA adapter instance"""
    global _lora_adapter
    if _lora_adapter is None:
        _lora_adapter = LoRAAdapter()
    return _lora_adapter


def get_gurukul_lora() -> GurukulLoRA:
    """Get global Gurukul LoRA instance"""
    global _gurukul_lora
    if _gurukul_lora is None:
        _gurukul_lora = GurukulLoRA()
    return _gurukul_lora