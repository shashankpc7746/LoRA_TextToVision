"""
Dataset Curator for Gurukul LoRA Training - Task 9 Day 1
Manages and validates the curated keyframe dataset (50-200 images)
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
from datetime import datetime


class GurukulDatasetCurator:
    """Curate and validate keyframe dataset for indigenous adapter training"""
    
    def __init__(self, dataset_path: str = "datasets/gurukul_keyframes"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        self.min_images = 50
        self.max_images = 200
        self.target_resolution = (1024, 1024)
        
        # Gurukul-themed prompts for educational content
        self.gurukul_prompts = [
            "Ancient Indian Gurukul classroom with teacher and students",
            "Traditional Vedic learning under banyan tree",
            "Students practicing yoga and meditation in ashram",
            "Teacher explaining mathematics on wooden board",
            "Traditional Indian music lesson with instruments",
            "Ayurveda herb garden teaching session",
            "Sanskrit calligraphy practice in classroom",
            "Traditional Indian dance training",
            "Astronomy lesson under night sky",
            "Philosophy discussion in forest ashram",
            "Traditional painting and art class",
            "Meditation and mindfulness practice",
            "Ancient Indian martial arts training",
            "Agricultural education in village setting",
            "Traditional storytelling session",
            "Gurukul morning prayer ritual",
            "Students learning archery in open field",
            "Traditional textile weaving lesson",
            "Cooking traditional Indian food class",
            "Nature walk and ecological education"
        ]
        
    def validate_dataset(self, verbose: bool = True) -> Dict[str, any]:
        """
        Validate existing dataset
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": False,
            "num_images": 0,
            "valid_images": 0,
            "invalid_images": [],
            "has_captions": False,
            "resolution_issues": [],
            "recommendations": []
        }
        
        # Check images
        image_files = list(self.dataset_path.glob("*.png")) + \
                     list(self.dataset_path.glob("*.jpg"))
        
        validation["num_images"] = len(image_files)
        
        if len(image_files) < self.min_images:
            validation["recommendations"].append(
                f"Need at least {self.min_images} images (found {len(image_files)})"
            )
        elif len(image_files) > self.max_images:
            validation["recommendations"].append(
                f"Too many images ({len(image_files)}), consider using {self.max_images}"
            )
            
        # Validate each image
        for img_file in image_files:
            try:
                img = Image.open(img_file)
                
                # Check resolution
                if img.size != self.target_resolution:
                    validation["resolution_issues"].append({
                        "file": img_file.name,
                        "size": img.size,
                        "expected": self.target_resolution
                    })
                else:
                    validation["valid_images"] += 1
                    
                img.close()
                
            except Exception as e:
                validation["invalid_images"].append({
                    "file": img_file.name,
                    "error": str(e)
                })
                
        # Check for captions
        caption_file = self.dataset_path / "captions.json"
        validation["has_captions"] = caption_file.exists()
        
        if not validation["has_captions"]:
            validation["recommendations"].append(
                "No captions.json found - will use default prompts"
            )
            
        # Overall validation
        validation["valid"] = (
            self.min_images <= validation["num_images"] <= self.max_images and
            len(validation["invalid_images"]) == 0 and
            validation["valid_images"] >= self.min_images
        )
        
        if verbose:
            self._print_validation_report(validation)
            
        return validation
        
    def _print_validation_report(self, validation: Dict):
        """Print validation report"""
        print(f"\n{'='*60}")
        print("Dataset Validation Report")
        print(f"{'='*60}\n")
        
        print(f"📁 Dataset Path: {self.dataset_path}")
        print(f"📊 Total Images: {validation['num_images']}")
        print(f"✅ Valid Images: {validation['valid_images']}")
        print(f"❌ Invalid Images: {len(validation['invalid_images'])}")
        print(f"📝 Has Captions: {'Yes' if validation['has_captions'] else 'No'}")
        
        if validation['resolution_issues']:
            print(f"\n⚠️ Resolution Issues: {len(validation['resolution_issues'])}")
            for issue in validation['resolution_issues'][:5]:  # Show first 5
                print(f"   - {issue['file']}: {issue['size']} (expected {issue['expected']})")
                
        if validation['invalid_images']:
            print(f"\n❌ Invalid Images:")
            for invalid in validation['invalid_images']:
                print(f"   - {invalid['file']}: {invalid['error']}")
                
        if validation['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in validation['recommendations']:
                print(f"   - {rec}")
                
        status = "✅ VALID" if validation['valid'] else "❌ INVALID"
        print(f"\n{'='*60}")
        print(f"Status: {status}")
        print(f"{'='*60}\n")
        
    def create_placeholder_dataset(self, num_images: int = 100) -> bool:
        """
        Create placeholder dataset for testing
        
        Args:
            num_images: Number of placeholder images to create
            
        Returns:
            True if successful
        """
        print(f"Creating placeholder dataset with {num_images} images...")
        
        try:
            # Create placeholder images
            for i in range(num_images):
                # Create a simple colored placeholder
                img = Image.new('RGB', self.target_resolution, 
                              color=(200, 180, 150))  # Beige color
                
                # Add text overlay (would be actual keyframe in production)
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                
                text = f"Placeholder Keyframe {i+1}\nGurukul Training Data"
                
                # Simple text (would use actual keyframes in production)
                try:
                    # Try to use a font
                    font = ImageFont.truetype("arial.ttf", 40)
                except:
                    font = ImageFont.load_default()
                    
                # Center text
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (self.target_resolution[0] - text_width) // 2
                y = (self.target_resolution[1] - text_height) // 2
                
                draw.text((x, y), text, fill=(50, 50, 50), font=font)
                
                # Save
                img_path = self.dataset_path / f"keyframe_{i+1:04d}.png"
                img.save(img_path)
                
            # Create captions
            self.generate_captions(num_images)
            
            print(f"✅ Created {num_images} placeholder images")
            print(f"📁 Location: {self.dataset_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create placeholder dataset: {e}")
            return False
            
    def generate_captions(self, num_images: Optional[int] = None) -> bool:
        """
        Generate captions.json for dataset
        
        Args:
            num_images: Number of images (auto-detect if None)
            
        Returns:
            True if successful
        """
        if num_images is None:
            image_files = list(self.dataset_path.glob("*.png")) + \
                         list(self.dataset_path.glob("*.jpg"))
            num_images = len(image_files)
            
        captions = {}
        
        # Distribute Gurukul prompts across images
        for i in range(num_images):
            img_name = f"keyframe_{i+1:04d}.png"
            # Rotate through available prompts
            prompt_idx = i % len(self.gurukul_prompts)
            captions[img_name] = self.gurukul_prompts[prompt_idx]
            
        # Save captions
        caption_file = self.dataset_path / "captions.json"
        with open(caption_file, 'w') as f:
            json.dump(captions, f, indent=2)
            
        print(f"✅ Generated captions for {num_images} images")
        return True
        
    def resize_images(self, target_size: Optional[Tuple[int, int]] = None) -> int:
        """
        Resize all images to target resolution
        
        Args:
            target_size: Target resolution (use default if None)
            
        Returns:
            Number of images resized
        """
        if target_size is None:
            target_size = self.target_resolution
            
        image_files = list(self.dataset_path.glob("*.png")) + \
                     list(self.dataset_path.glob("*.jpg"))
        
        resized_count = 0
        
        print(f"Resizing images to {target_size}...")
        
        for img_file in image_files:
            try:
                img = Image.open(img_file)
                
                if img.size != target_size:
                    # Resize with high-quality resampling
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    img_resized.save(img_file)
                    resized_count += 1
                    
                img.close()
                
            except Exception as e:
                print(f"⚠️ Failed to resize {img_file.name}: {e}")
                
        print(f"✅ Resized {resized_count} images")
        return resized_count
        
    def export_dataset_info(self) -> Dict:
        """Export dataset information for metadata"""
        validation = self.validate_dataset(verbose=False)
        
        info = {
            "dataset_name": "gurukul_keyframes_v1",
            "dataset_path": str(self.dataset_path),
            "num_images": validation["num_images"],
            "valid_images": validation["valid_images"],
            "resolution": self.target_resolution,
            "created_at": datetime.now().isoformat(),
            "prompts_used": self.gurukul_prompts,
            "validation_status": "valid" if validation["valid"] else "invalid"
        }
        
        return info


def prepare_training_dataset(dataset_path: str = "datasets/gurukul_keyframes",
                            create_placeholder: bool = False,
                            num_images: int = 100) -> Dict:
    """
    Convenience function to prepare training dataset
    
    Args:
        dataset_path: Path to dataset directory
        create_placeholder: Create placeholder images if no dataset exists
        num_images: Number of placeholder images to create
        
    Returns:
        Validation results dictionary
    """
    curator = GurukulDatasetCurator(dataset_path)
    
    # Check if dataset exists
    existing_images = list(Path(dataset_path).glob("*.png")) + \
                     list(Path(dataset_path).glob("*.jpg"))
    
    if len(existing_images) == 0 and create_placeholder:
        print("No existing dataset found. Creating placeholder dataset...")
        curator.create_placeholder_dataset(num_images)
    elif len(existing_images) > 0:
        print(f"Found {len(existing_images)} existing images")
        
        # Resize if needed
        curator.resize_images()
        
        # Generate captions if missing
        caption_file = Path(dataset_path) / "captions.json"
        if not caption_file.exists():
            curator.generate_captions()
            
    # Validate
    validation = curator.validate_dataset()
    
    return validation


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curate Gurukul Keyframe Dataset")
    parser.add_argument("--dataset_path", type=str, 
                       default="datasets/gurukul_keyframes",
                       help="Path to dataset directory")
    parser.add_argument("--create_placeholder", action="store_true",
                       help="Create placeholder dataset if none exists")
    parser.add_argument("--num_images", type=int, default=100,
                       help="Number of placeholder images to create")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate existing dataset")
    
    args = parser.parse_args()
    
    curator = GurukulDatasetCurator(args.dataset_path)
    
    if args.validate_only:
        validation = curator.validate_dataset()
    else:
        validation = prepare_training_dataset(
            args.dataset_path,
            args.create_placeholder,
            args.num_images
        )
        
    if validation["valid"]:
        print("\n✅ Dataset is ready for training!")
    else:
        print("\n⚠️ Dataset needs attention before training")
