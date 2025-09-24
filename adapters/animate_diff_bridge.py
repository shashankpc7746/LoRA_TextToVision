"""
AnimateDiff Bridge for Task-7 Quality Leap
Integrates keyframes with AnimateDiff for smooth video generation
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import json
from datetime import datetime
import subprocess
import shutil

from .keyframe_generator import get_keyframe_generator


class AnimateDiffBridge:
    """Bridge between keyframes and AnimateDiff animation"""

    def __init__(self, animate_diff_path: str = "AnimateDiff"):
        self.animate_diff_path = Path(animate_diff_path)
        self.keyframe_generator = get_keyframe_generator()

        # Animation settings
        self.animation_config = {
            "fps": 12,
            "width": 512,
            "height": 512,
            "num_frames": 24,  # 2 seconds at 12fps
            "motion_scale": 1.0,
            "controlnet_strength": 0.8,
        }

    def animate_between_keyframes(self, keyframes_dir: str,
                                 output_video_path: str,
                                 **kwargs) -> Dict[str, Any]:
        """Animate between keyframes using AnimateDiff"""

        keyframes_path = Path(keyframes_dir)
        if not keyframes_path.exists():
            return {
                "success": False,
                "error": f"Keyframes directory not found: {keyframes_dir}"
            }

        # Load keyframes
        keyframes = self.keyframe_generator.load_keyframes_from_directory(keyframes_dir)

        if len(keyframes) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 keyframes, found {len(keyframes)}"
            }

        print(f"Animating between {len(keyframes)} keyframes...")

        try:
            # For now, create a simple video from keyframes (placeholder for full AnimateDiff integration)
            # In production, this would call the actual AnimateDiff pipeline

            from moviepy.editor import ImageSequenceClip

            # Extract image paths in order
            image_paths = []
            for kf in sorted(keyframes, key=lambda x: x.get("index", 0)):
                img_path = kf.get("image_path")
                if img_path and Path(img_path).exists():
                    image_paths.append(img_path)

            if not image_paths:
                return {
                    "success": False,
                    "error": "No valid keyframe images found"
                }

            # Create simple video from keyframes
            clip = ImageSequenceClip(image_paths, fps=self.animation_config["fps"])

            # Resize if needed
            if clip.size != (self.animation_config["width"], self.animation_config["height"]):
                clip = clip.resize(width=self.animation_config["width"],
                                 height=self.animation_config["height"])

            # Write video
            output_path = Path(output_video_path)
            output_path.parent.mkdir(exist_ok=True)

            clip.write_videofile(
                str(output_path),
                fps=self.animation_config["fps"],
                codec="libx264",
                audio=False,
                verbose=False,
                logger=None
            )

            # Clean up
            clip.close()

            return {
                "success": True,
                "output_path": str(output_path),
                "num_keyframes": len(keyframes),
                "duration_seconds": len(image_paths) / self.animation_config["fps"],
                "fps": self.animation_config["fps"],
                "resolution": f"{self.animation_config['width']}x{self.animation_config['height']}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Animation failed: {str(e)}"
            }

    def create_preview_clip(self, keyframes_dir: str,
                           output_dir: str = "previews") -> Dict[str, Any]:
        """Create a quick preview clip from keyframes"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"preview_{timestamp}.mp4"
        video_path = output_path / video_filename

        result = self.animate_between_keyframes(
            keyframes_dir,
            str(video_path)
        )

        if result["success"]:
            # Create additional metadata
            metadata = {
                "type": "preview_clip",
                "keyframes_dir": keyframes_dir,
                "created": datetime.now().isoformat(),
                "animation_config": self.animation_config,
                "result": result
            }

            metadata_path = video_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            result["metadata_path"] = str(metadata_path)

        return result

    def validate_animation_quality(self, video_path: str) -> Dict[str, Any]:
        """Validate the quality of generated animation"""

        video_file = Path(video_path)
        if not video_file.exists():
            return {"valid": False, "error": "Video file not found"}

        validation = {
            "file_exists": True,
            "file_size_mb": video_file.stat().st_size / (1024 * 1024),
            "valid": True,
            "issues": []
        }

        # Basic validation - could be extended with VMAF, frame analysis, etc.
        try:
            # Check if video is readable
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(video_path))
            validation["duration"] = clip.duration
            validation["fps"] = clip.fps
            validation["resolution"] = clip.size
            clip.close()

        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Video validation failed: {e}")

        return validation


# Global bridge instance
_animate_diff_bridge = None


def get_animate_diff_bridge() -> AnimateDiffBridge:
    """Get global AnimateDiff bridge instance"""
    global _animate_diff_bridge
    if _animate_diff_bridge is None:
        _animate_diff_bridge = AnimateDiffBridge()
    return _animate_diff_bridge


def create_keyframe_animation(keyframes_dir: str,
                            output_video: str = "output.mp4") -> Dict[str, Any]:
    """Convenience function for keyframe animation"""
    bridge = get_animate_diff_bridge()
    return bridge.animate_between_keyframes(keyframes_dir, output_video)


def create_preview_from_prompt(prompt: str,
                             num_keyframes: int = 6,
                             output_dir: str = "previews") -> Dict[str, Any]:
    """Create preview animation directly from prompt"""

    # Generate keyframes
    keyframe_gen = get_keyframe_generator()
    keyframes_result = keyframe_gen.generate_keyframes_sync(prompt, num_keyframes)

    if not keyframes_result or not any(kf["success"] for kf in keyframes_result):
        return {
            "success": False,
            "error": "Keyframe generation failed"
        }

    # Find successful keyframes directory
    successful_keyframes = [kf for kf in keyframes_result if kf["success"]]
    if not successful_keyframes:
        return {
            "success": False,
            "error": "No successful keyframes generated"
        }

    # Extract directory from first successful keyframe
    first_keyframe_path = Path(successful_keyframes[0]["image_path"])
    keyframes_dir = str(first_keyframe_path.parent)

    # Create animation
    bridge = get_animate_diff_bridge()
    result = bridge.create_preview_clip(keyframes_dir, output_dir)

    # Add keyframe generation info
    result["keyframes_generated"] = len(successful_keyframes)
    result["keyframes_dir"] = keyframes_dir

    return result


def quick_test_animation():
    """Quick test of the animation pipeline"""
    print("Testing keyframe animation pipeline...")

    test_prompt = "traditional Indian classroom scene"

    try:
        result = create_preview_from_prompt(test_prompt, num_keyframes=3)

        if result["success"]:
            print("✅ Animation pipeline working!")
            print(f"   Video: {result['output_path']}")
            print(f"   Duration: {result.get('duration_seconds', 'N/A')}s")
            return True
        else:
            print(f"❌ Animation failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    # Quick test when run directly
    quick_test_animation()