"""
Interpolation Pipeline for Task-7 Quality Leap
Complete pipeline: keyframes → RIFE → stabilization → video
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .rife_interpolator import get_rife_interpolator, get_frame_cache
from adapters.keyframe_generator import get_keyframe_generator


class StabilizationEngine:
    """Temporal stabilization and flicker reduction"""

    def __init__(self):
        self.stabilization_config = {
            "temporal_window": 5,  # Frames for median filtering
            "histogram_bins": 256,
            "color_correction_strength": 0.3,
            "flicker_threshold": 0.05
        }

    def stabilize_sequence(self, frame_paths: List[str],
                          output_dir: str) -> List[str]:
        """Apply temporal stabilization to frame sequence"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        stabilized_frames = []

        print(f"Stabilizing {len(frame_paths)} frames...")

        # Load all frames into memory for temporal processing
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)

        if len(frames) < 3:
            print("Not enough frames for stabilization, returning original")
            return frame_paths

        # Apply temporal median filtering
        stabilized_frames_data = self._apply_temporal_filtering(frames)

        # Apply color histogram normalization
        stabilized_frames_data = self._normalize_color_histograms(stabilized_frames_data)

        # Save stabilized frames
        for i, frame in enumerate(stabilized_frames_data):
            output_file = output_path / "04d"
            cv2.imwrite(str(output_file), frame)
            stabilized_frames.append(str(output_file))

        print(f"Stabilization complete: {len(stabilized_frames)} frames")
        return stabilized_frames

    def _apply_temporal_filtering(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply temporal median filtering to reduce flicker"""

        window_size = self.stabilization_config["temporal_window"]
        stabilized = []

        for i in range(len(frames)):
            # Get temporal window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(frames), i + window_size // 2 + 1)

            window_frames = frames[start_idx:end_idx]

            # Apply median filtering across temporal window
            median_frame = np.median(window_frames, axis=0).astype(np.uint8)
            stabilized.append(median_frame)

        return stabilized

    def _normalize_color_histograms(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize color histograms across frames"""

        # Calculate reference histogram from middle frame
        ref_frame = frames[len(frames) // 2]
        ref_hist = self._calculate_histogram(ref_frame)

        normalized_frames = []

        for frame in frames:
            frame_hist = self._calculate_histogram(frame)

            # Simple color correction based on histogram matching
            # This is a simplified version - production would use more sophisticated methods
            correction_factor = self.stabilization_config["color_correction_strength"]

            # Apply gentle correction
            corrected_frame = cv2.convertScaleAbs(
                frame, alpha=1.0, beta=0
            )  # Placeholder for actual histogram matching

            normalized_frames.append(corrected_frame)

        return normalized_frames

    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate color histogram for a frame"""
        hist = []
        for channel in range(3):  # BGR channels
            channel_hist = cv2.calcHist(
                [frame], [channel], None,
                [self.stabilization_config["histogram_bins"]], [0, 256]
            )
            hist.append(channel_hist.flatten())

        return np.concatenate(hist)


class InterpolationPipeline:
    """Complete interpolation pipeline with stabilization"""

    def __init__(self):
        self.interpolator = get_rife_interpolator()
        self.frame_cache = get_frame_cache()
        self.stabilizer = StabilizationEngine()
        self.keyframe_generator = get_keyframe_generator()

        # Pipeline configuration
        self.pipeline_config = {
            "input_fps": 12,  # Keyframe rate
            "target_fps": 24,  # Interpolated rate
            "stabilization_enabled": True,
            "cache_enabled": True,
            "gpu_device": "cuda:1",  # RTX 3060
        }

    async def process_keyframes_to_video(self, keyframes_dir: str,
                                       output_video: str,
                                       **kwargs) -> Dict[str, Any]:
        """Complete pipeline: keyframes → interpolation → stabilization → video"""

        # Update config with kwargs
        config = self.pipeline_config.copy()
        config.update(kwargs)

        print("Starting interpolation pipeline...")
        print(f"Input: {keyframes_dir}")
        print(f"Target FPS: {config['input_fps']} → {config['target_fps']}")

        try:
            # Step 1: Load and validate keyframes
            keyframes = self.keyframe_generator.load_keyframes_from_directory(keyframes_dir)

            if len(keyframes) < 2:
                return {
                    "success": False,
                    "error": f"Need at least 2 keyframes, found {len(keyframes)}"
                }

            # Step 2: Interpolate frames
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                interp_result = self.interpolator.interpolate_video_sequence(
                    keyframes_dir, temp_dir, config["target_fps"]
                )

                if not interp_result["success"]:
                    return interp_result

                frame_paths = interp_result["frame_paths"]

                # Step 3: Apply stabilization if enabled
                if config["stabilization_enabled"]:
                    stabilized_dir = Path(temp_dir) / "stabilized"
                    frame_paths = self.stabilizer.stabilize_sequence(
                        frame_paths, str(stabilized_dir)
                    )

                # Step 4: Create final video
                from moviepy.editor import ImageSequenceClip

                if not frame_paths:
                    return {
                        "success": False,
                        "error": "No frames available for video creation"
                    }

                clip = ImageSequenceClip(frame_paths, fps=config["target_fps"])

                # Video export settings
                output_path = Path(output_video)
                output_path.parent.mkdir(exist_ok=True)

                clip.write_videofile(
                    str(output_path),
                    fps=config["target_fps"],
                    codec="libx264",
                    audio=False,
                    verbose=False,
                    logger=None
                )

                clip.close()

                # Calculate metrics
                duration = len(frame_paths) / config["target_fps"]
                file_size_mb = output_path.stat().st_size / (1024 * 1024)

                return {
                    "success": True,
                    "output_path": str(output_path),
                    "duration_seconds": duration,
                    "fps": config["target_fps"],
                    "num_frames": len(frame_paths),
                    "keyframes_used": len(keyframes),
                    "file_size_mb": file_size_mb,
                    "stabilization_applied": config["stabilization_enabled"],
                    "interpolation_factor": config["target_fps"] // config["input_fps"]
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Pipeline failed: {str(e)}"
            }

    def validate_interpolation_quality(self, video_path: str) -> Dict[str, Any]:
        """Validate the quality of interpolated video"""

        video_file = Path(video_path)
        if not video_file.exists():
            return {"valid": False, "error": "Video file not found"}

        validation = {
            "file_exists": True,
            "file_size_mb": video_file.stat().st_size / (1024 * 1024),
            "valid": True,
            "issues": [],
            "quality_metrics": {}
        }

        try:
            # Basic video validation
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(video_path))

            validation["quality_metrics"].update({
                "duration": clip.duration,
                "fps": clip.fps,
                "resolution": clip.size,
                "aspect_ratio": clip.aspect_ratio
            })

            # Check for smooth motion (basic frame difference analysis)
            # This is a simplified version - production would use more sophisticated metrics
            frame_differences = []
            prev_frame = None

            # Sample a few frames for analysis
            sample_times = np.linspace(0, clip.duration, min(10, int(clip.duration)))

            for t in sample_times:
                frame = clip.get_frame(t)
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)))
                    frame_differences.append(diff)
                prev_frame = frame

            if frame_differences:
                avg_difference = np.mean(frame_differences)
                validation["quality_metrics"]["avg_frame_difference"] = avg_difference

                # Flag potential issues
                if avg_difference < 1.0:  # Too similar frames
                    validation["issues"].append("Frames too similar - possible interpolation issues")
                elif avg_difference > 50.0:  # Too different frames
                    validation["issues"].append("Frames too different - possible stabilization issues")

            clip.close()

        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Video validation failed: {e}")

        validation["valid"] = len(validation["issues"]) == 0
        return validation


# Global pipeline instance
_interpolation_pipeline = None


def get_interpolation_pipeline() -> InterpolationPipeline:
    """Get global interpolation pipeline instance"""
    global _interpolation_pipeline
    if _interpolation_pipeline is None:
        _interpolation_pipeline = InterpolationPipeline()
    return _interpolation_pipeline


def interpolate_video_from_keyframes(keyframes_dir: str,
                                   output_video: str = "interpolated_video.mp4",
                                   **kwargs) -> Dict[str, Any]:
    """Convenience function for complete interpolation pipeline"""

    pipeline = get_interpolation_pipeline()

    # Run async pipeline
    async def run_pipeline():
        return await pipeline.process_keyframes_to_video(keyframes_dir, output_video, **kwargs)

    return asyncio.run(run_pipeline())


def quick_test_pipeline():
    """Quick test of interpolation pipeline"""
    print("Testing interpolation pipeline...")

    try:
        pipeline = get_interpolation_pipeline()
        interpolator = pipeline.interpolator
        stabilizer = pipeline.stabilizer

        print("✅ Pipeline components initialized")
        print(f"   Interpolator device: {interpolator.device}")
        print(f"   Target FPS: {pipeline.pipeline_config['target_fps']}")
        print(f"   Stabilization: {pipeline.pipeline_config['stabilization_enabled']}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    quick_test_pipeline()