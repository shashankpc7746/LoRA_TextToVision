"""
Keyframe Generator for Task-7 Quality Leap
Generates high-quality keyframes for AnimateDiff interpolation
"""

import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .lora_adapter import get_gurukul_lora


class KeyframeGenerator:
    """Generates high-quality keyframes for video animation"""

    def __init__(self, output_dir: str = "keyframes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Keyframe generation settings
        self.keyframe_config = {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "num_keyframes": 6,
            "batch_size": 2,  # Generate multiple keyframes in parallel
        }

        self.gurukul_lora = get_gurukul_lora()

    def generate_keyframe_prompts(self, base_prompt: str, num_keyframes: int = 6) -> List[str]:
        """Generate varied prompts for keyframes"""
        # Create variations for smooth animation transitions
        variations = [
            f"{base_prompt}, wide shot, establishing scene",
            f"{base_prompt}, medium shot, main action begins",
            f"{base_prompt}, close-up, emotional moment",
            f"{base_prompt}, dynamic angle, action intensifies",
            f"{base_prompt}, wide shot, climax approaching",
            f"{base_prompt}, medium shot, resolution",
        ]

        # Extend if more keyframes needed
        while len(variations) < num_keyframes:
            variations.append(f"{base_prompt}, creative variation {len(variations) + 1}")

        return variations[:num_keyframes]

    async def generate_single_keyframe(self, prompt: str, index: int,
                                     output_path: Path) -> Dict[str, Any]:
        """Generate a single keyframe asynchronously"""
        try:
            print(f"Generating keyframe {index + 1}: {prompt[:50]}...")

            # Generate with Gurukul LoRA
            result = self.gurukul_lora.generate_gurukul_content(
                custom_prompt=prompt,
                width=self.keyframe_config["width"],
                height=self.keyframe_config["height"],
                num_inference_steps=self.keyframe_config["num_inference_steps"],
                guidance_scale=self.keyframe_config["guidance_scale"]
            )

            if result["images"]:
                # Save the first (and typically only) image
                image = result["images"][0]
                image_path = output_path / "04d"

                # Save image
                image.save(image_path, "PNG")

                # Save metadata
                metadata = {
                    "prompt": prompt,
                    "index": index,
                    "timestamp": datetime.now().isoformat(),
                    "parameters": result.get("parameters", {}),
                    "success": True
                }

                metadata_path = output_path / "04d"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                return {
                    "success": True,
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "prompt": prompt,
                    "index": index
                }
            else:
                return {
                    "success": False,
                    "error": "No images generated",
                    "prompt": prompt,
                    "index": index
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
                "index": index
            }

    async def generate_keyframes_async(self, base_prompt: str,
                                      num_keyframes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate multiple keyframes asynchronously"""
        if num_keyframes is None:
            num_keyframes = self.keyframe_config["num_keyframes"]

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"keyframes_{timestamp}"
        output_path.mkdir(exist_ok=True)

        # Generate varied prompts
        prompts = self.generate_keyframe_prompts(base_prompt, num_keyframes)

        print(f"Generating {num_keyframes} keyframes for: {base_prompt[:50]}...")
        print(f"Output directory: {output_path}")

        # Generate keyframes concurrently (but not too many at once to avoid GPU memory issues)
        batch_size = min(self.keyframe_config["batch_size"], num_keyframes)
        results = []

        for i in range(0, num_keyframes, batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_tasks = []

            for j, prompt in enumerate(batch_prompts):
                task = self.generate_single_keyframe(prompt, i + j, output_path)
                batch_tasks.append(task)

            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

            # Small delay between batches
            if i + batch_size < num_keyframes:
                await asyncio.sleep(1)

        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nKeyframe generation complete:")
        print(f"  Total: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        print(f"  Output: {output_path}")

        return results

    def generate_keyframes_sync(self, base_prompt: str,
                               num_keyframes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Synchronous wrapper for keyframe generation"""
        async def run():
            return await self.generate_keyframes_async(base_prompt, num_keyframes)

        return asyncio.run(run())

    def load_keyframes_from_directory(self, keyframes_dir: str) -> List[Dict[str, Any]]:
        """Load existing keyframes from directory"""
        keyframes_path = Path(keyframes_dir)
        keyframes = []

        if not keyframes_path.exists():
            print(f"Keyframes directory not found: {keyframes_dir}")
            return keyframes

        # Find all PNG files and their metadata
        for image_file in sorted(keyframes_path.glob("*.png")):
            metadata_file = image_file.with_suffix('.json')

            keyframe_data = {
                "image_path": str(image_file),
                "index": len(keyframes)
            }

            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        keyframe_data.update(metadata)
                except Exception as e:
                    print(f"Warning: Could not load metadata for {image_file}: {e}")

            keyframes.append(keyframe_data)

        print(f"Loaded {len(keyframes)} keyframes from {keyframes_dir}")
        return keyframes

    def validate_keyframes(self, keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate keyframe quality and consistency"""
        if not keyframes:
            return {"valid": False, "error": "No keyframes provided"}

        validation_results = {
            "total_keyframes": len(keyframes),
            "valid_images": 0,
            "invalid_images": 0,
            "average_resolution": None,
            "consistent_resolution": True,
            "issues": []
        }

        resolutions = []

        for kf in keyframes:
            image_path = kf.get("image_path")
            if not image_path or not Path(image_path).exists():
                validation_results["invalid_images"] += 1
                validation_results["issues"].append(f"Missing image: {image_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolutions.append((width, height))
                    validation_results["valid_images"] += 1
            except Exception as e:
                validation_results["invalid_images"] += 1
                validation_results["issues"].append(f"Invalid image {image_path}: {e}")

        # Check resolution consistency
        if resolutions:
            first_res = resolutions[0]
            for res in resolutions[1:]:
                if res != first_res:
                    validation_results["consistent_resolution"] = False
                    break

            avg_width = sum(w for w, h in resolutions) / len(resolutions)
            avg_height = sum(h for w, h in resolutions) / len(resolutions)
            validation_results["average_resolution"] = (int(avg_width), int(avg_height))

        validation_results["valid"] = validation_results["invalid_images"] == 0

        return validation_results


# Global keyframe generator instance
_keyframe_generator = None


def get_keyframe_generator() -> KeyframeGenerator:
    """Get global keyframe generator instance"""
    global _keyframe_generator
    if _keyframe_generator is None:
        _keyframe_generator = KeyframeGenerator()
    return _keyframe_generator


def generate_keyframes(prompt: str, num_keyframes: int = 6,
                      output_dir: str = "keyframes") -> List[Dict[str, Any]]:
    """Convenience function for keyframe generation"""
    generator = KeyframeGenerator(output_dir)
    return generator.generate_keyframes_sync(prompt, num_keyframes)


def quick_test_keyframes():
    """Quick test function for keyframe generation"""
    print("Testing keyframe generation...")

    test_prompt = "traditional Indian classroom with teacher and students"
    generator = get_keyframe_generator()

    try:
        results = generator.generate_keyframes_sync(test_prompt, num_keyframes=2)
        successful = sum(1 for r in results if r["success"])

        print(f"Test completed: {successful}/{len(results)} keyframes generated")

        if successful > 0:
            print("✅ Keyframe generation working!")
            return True
        else:
            print("❌ Keyframe generation failed")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    # Quick test when run directly
    quick_test_keyframes()