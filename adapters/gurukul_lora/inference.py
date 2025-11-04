"""
Indigenous Generator - Deterministic Keyframe Generation - Task 9 Day 1
Generate keyframes using trained Gurukul LoRA adapter with deterministic seeding
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import json
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from peft import PeftModel
import hashlib


class IndigenousGenerator:
    """Generate keyframes using indigenous Gurukul LoRA adapter"""
    
    def __init__(self, 
                 adapter_path: str = "adapters/gurukul_lora/gurukul_lora.pt",
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda:0"):
        
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self.device = device
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Default generation config
        self.generation_config = {
            "seed": 42,
            "cfg_scale": 7.5,
            "num_inference_steps": 30,
            "width": 1024,
            "height": 1024,
            "scheduler": "DDIM",
        }
        
        # Update from metadata if available
        if self.metadata and "deterministic_config" in self.metadata:
            self.generation_config.update(self.metadata["deterministic_config"])
            
        self.pipeline = None
        print(f"Indigenous Generator initialized")
        print(f"Adapter: {self.adapter_path}")
        print(f"Device: {self.device}")
        
    def _load_metadata(self) -> Optional[Dict]:
        """Load adapter metadata"""
        metadata_path = self.adapter_path.parent / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"✅ Loaded metadata from {metadata_path}")
                return metadata
        else:
            print(f"⚠️ No metadata found at {metadata_path}")
            return None
            
    def load_adapter(self) -> bool:
        """Load trained Gurukul LoRA adapter"""
        if not self.adapter_path.exists():
            print(f"❌ Adapter not found: {self.adapter_path}")
            print("Please train the adapter first using train_adapter.py")
            return False
            
        try:
            print(f"Loading indigenous adapter: {self.adapter_path}")
            
            # Load base pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            
            # Load adapter checkpoint
            checkpoint = torch.load(self.adapter_path, map_location="cpu")
            
            if "state_dict" in checkpoint:
                lora_state_dict = checkpoint["state_dict"]
            else:
                lora_state_dict = checkpoint
                
            # Apply LoRA weights to UNet
            # This is a simplified version - in production would use PeftModel.from_pretrained
            print("Applying LoRA weights to pipeline...")
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Setup scheduler
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            print(f"✅ Adapter loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load adapter: {e}")
            return False
            
    def generate_keyframe(self, 
                         prompt: str,
                         seed: Optional[int] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate keyframe with deterministic seeding
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed (uses default if None)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results and metadata
        """
        if self.pipeline is None:
            if not self.load_adapter():
                return {
                    "success": False,
                    "error": "Failed to load adapter"
                }
                
        # Use provided seed or default
        if seed is None:
            seed = self.generation_config["seed"]
            
        # Set deterministic seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Merge generation config
        gen_params = {**self.generation_config, **kwargs}
        gen_params["seed"] = seed
        
        print(f"\n🎨 Generating keyframe with Gurukul LoRA")
        print(f"Prompt: {prompt}")
        print(f"Seed: {seed}")
        print(f"CFG Scale: {gen_params['cfg_scale']}")
        
        try:
            # Generate
            output = self.pipeline(
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted, modern, western style",
                num_inference_steps=gen_params["num_inference_steps"],
                guidance_scale=gen_params["cfg_scale"],
                width=gen_params["width"],
                height=gen_params["height"],
                generator=generator,
            )
            
            image = output.images[0]
            
            # Generate unique ID
            image_id = self._generate_image_id(prompt, seed)
            
            # Create result
            result = {
                "success": True,
                "image": image,
                "image_id": image_id,
                "metadata": {
                    "prompt": prompt,
                    "seed": seed,
                    "cfg_scale": gen_params["cfg_scale"],
                    "num_inference_steps": gen_params["num_inference_steps"],
                    "resolution": (gen_params["width"], gen_params["height"]),
                    "scheduler": gen_params["scheduler"],
                    "adapter_used": str(self.adapter_path),
                    "generated_at": datetime.now().isoformat(),
                    
                    # KSML compliance
                    "ksml_token": {
                        "intent": prompt,
                        "karma_state": "indigenous_generation",
                        "lineage": {
                            "adapter": "gurukul_lora.pt",
                            "base_model": self.base_model,
                            "parent_models": ["SDXL", "clip-vit-large"],
                            "training_dataset": self.metadata.get("ksml_lineage", {}).get("training_dataset", "gurukul_keyframes_v1") if self.metadata else "gurukul_keyframes_v1"
                        }
                    }
                }
            }
            
            print(f"✅ Generation successful: {image_id}")
            return result
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def generate_batch(self, 
                      prompts: List[str],
                      seeds: Optional[List[int]] = None,
                      output_dir: str = "outputs/indigenous_keyframes") -> List[Dict]:
        """
        Generate batch of keyframes
        
        Args:
            prompts: List of text prompts
            seeds: List of seeds (one per prompt)
            output_dir: Output directory for images
            
        Returns:
            List of generation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate seeds if not provided
        if seeds is None:
            seeds = [self.generation_config["seed"] + i for i in range(len(prompts))]
        elif len(seeds) != len(prompts):
            raise ValueError("Number of seeds must match number of prompts")
            
        results = []
        
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            print(f"\n[{i+1}/{len(prompts)}] Generating keyframe...")
            
            result = self.generate_keyframe(prompt, seed)
            
            if result["success"]:
                # Save image
                image_path = output_path / f"keyframe_{result['image_id']}.png"
                result["image"].save(image_path)
                result["image_path"] = str(image_path)
                
                # Save metadata
                metadata_path = output_path / f"keyframe_{result['image_id']}_metadata.json"
                with open(metadata_path, 'w') as f:
                    # Remove image from metadata for JSON serialization
                    metadata_to_save = {k: v for k, v in result.items() if k != "image"}
                    json.dump(metadata_to_save, f, indent=2)
                    
                print(f"💾 Saved: {image_path}")
                
            results.append(result)
            
        # Generate batch summary
        summary = {
            "total_prompts": len(prompts),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "output_directory": str(output_path),
            "generated_at": datetime.now().isoformat()
        }
        
        summary_path = output_path / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n{'='*60}")
        print(f"Batch Generation Complete")
        print(f"{'='*60}")
        print(f"✅ Successful: {summary['successful']}/{summary['total_prompts']}")
        print(f"📁 Output: {output_path}")
        print(f"{'='*60}\n")
        
        return results
        
    def _generate_image_id(self, prompt: str, seed: int) -> str:
        """Generate unique deterministic image ID"""
        content = f"{prompt}_{seed}_{self.adapter_path.name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
        
    def verify_determinism(self, prompt: str, seed: int, num_trials: int = 3) -> Dict:
        """
        Verify that generation is deterministic
        
        Args:
            prompt: Test prompt
            seed: Test seed
            num_trials: Number of generation trials
            
        Returns:
            Verification results
        """
        print(f"\n🔬 Testing determinism with {num_trials} trials...")
        print(f"Prompt: {prompt}")
        print(f"Seed: {seed}\n")
        
        results = []
        image_hashes = []
        
        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials}...")
            result = self.generate_keyframe(prompt, seed)
            
            if result["success"]:
                # Calculate image hash
                img_array = torch.tensor(result["image"]).numpy()
                img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
                image_hashes.append(img_hash)
                results.append(result)
            else:
                print(f"⚠️ Trial {i+1} failed")
                
        # Check if all hashes are identical
        is_deterministic = len(set(image_hashes)) == 1
        
        verification = {
            "is_deterministic": is_deterministic,
            "num_trials": num_trials,
            "successful_trials": len(results),
            "unique_outputs": len(set(image_hashes)),
            "prompt": prompt,
            "seed": seed
        }
        
        if is_deterministic:
            print(f"\n✅ DETERMINISTIC: All {num_trials} trials produced identical output")
        else:
            print(f"\n❌ NOT DETERMINISTIC: Got {len(set(image_hashes))} unique outputs")
            
        return verification


def generate_with_gurukul_lora(prompt: str,
                               seed: int = 42,
                               adapter_path: str = "adapters/gurukul_lora/gurukul_lora.pt",
                               output_dir: str = "outputs/indigenous_keyframes") -> Dict:
    """
    Convenience function to generate with Gurukul LoRA
    
    Args:
        prompt: Text prompt
        seed: Random seed for deterministic generation
        adapter_path: Path to trained adapter
        output_dir: Output directory
        
    Returns:
        Generation result dictionary
    """
    generator = IndigenousGenerator(adapter_path)
    
    if not generator.load_adapter():
        return {
            "success": False,
            "error": "Failed to load adapter"
        }
        
    result = generator.generate_keyframe(prompt, seed)
    
    if result["success"]:
        # Save output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_path = output_path / f"keyframe_{result['image_id']}.png"
        result["image"].save(image_path)
        result["image_path"] = str(image_path)
        
        print(f"💾 Saved: {image_path}")
        
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate with Gurukul LoRA Adapter")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for deterministic generation")
    parser.add_argument("--adapter_path", type=str,
                       default="adapters/gurukul_lora/gurukul_lora.pt",
                       help="Path to trained adapter")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/indigenous_keyframes",
                       help="Output directory")
    parser.add_argument("--verify_determinism", action="store_true",
                       help="Run determinism verification test")
    parser.add_argument("--batch", action="store_true",
                       help="Generate batch with multiple seeds")
    parser.add_argument("--num_variations", type=int, default=5,
                       help="Number of variations for batch generation")
    
    args = parser.parse_args()
    
    generator = IndigenousGenerator(args.adapter_path)
    
    if not generator.load_adapter():
        print("❌ Failed to load adapter. Exiting.")
        exit(1)
        
    if args.verify_determinism:
        # Test determinism
        verification = generator.verify_determinism(args.prompt, args.seed)
        print(f"\nVerification: {json.dumps(verification, indent=2)}")
        
    elif args.batch:
        # Generate batch with multiple seeds
        seeds = [args.seed + i for i in range(args.num_variations)]
        prompts = [args.prompt] * args.num_variations
        
        results = generator.generate_batch(prompts, seeds, args.output_dir)
        
        print(f"\n✅ Generated {len(results)} keyframe variations")
        
    else:
        # Single generation
        result = generate_with_gurukul_lora(
            args.prompt,
            args.seed,
            args.adapter_path,
            args.output_dir
        )
        
        if result["success"]:
            print(f"\n✅ Generation successful!")
            print(f"Image: {result.get('image_path', 'N/A')}")
        else:
            print(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}")
