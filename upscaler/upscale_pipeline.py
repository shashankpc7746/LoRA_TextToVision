"""
Upscale Pipeline for Task-7 Quality Leap
Complete pipeline: video → denoise → upscaler → cinematic polish → final output
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

from .esrgan_upscaler import get_esrgan_upscaler, get_tile_processor


class DenoiseEngine:
    """Advanced denoising and quality enhancement"""

    def __init__(self):
        self.denoise_config = {
            "temporal_radius": 3,  # Temporal window for denoising
            "spatial_sigma": 15,   # Spatial denoising strength
            "luminance_sigma": 10, # Luminance denoising
            "color_sigma": 20,     # Color denoising
            "quality_enhancement": True,
            "sharpen_strength": 0.3
        }

    def denoise_frame_sequence(self, frame_paths: List[str],
                             output_dir: str) -> List[str]:
        """Apply temporal and spatial denoising to frame sequence"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        denoised_frames = []

        print(f"Denoising {len(frame_paths)} frames...")

        # Load all frames for temporal processing
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)

        if len(frames) < 3:
            print("Not enough frames for temporal denoising, applying spatial only")
            # Apply spatial denoising only
            for i, frame in enumerate(frames):
                denoised = self._apply_spatial_denoise(frame)
                output_file = output_path / "04d"
                cv2.imwrite(str(output_file), denoised)
                denoised_frames.append(str(output_file))
            return denoised_frames

        # Apply temporal denoising
        denoised_frames_data = self._apply_temporal_denoise(frames)

        # Apply additional quality enhancement
        if self.denoise_config["quality_enhancement"]:
            denoised_frames_data = self._enhance_quality(denoised_frames_data)

        # Save denoised frames
        for i, frame in enumerate(denoised_frames_data):
            output_file = output_path / "04d"
            cv2.imwrite(str(output_file), frame)
            denoised_frames.append(str(output_file))

        print(f"Denoising complete: {len(denoised_frames)} frames")
        return denoised_frames

    def _apply_temporal_denoise(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply temporal denoising across frame sequence"""

        radius = self.denoise_config["temporal_radius"]
        denoised = []

        for i in range(len(frames)):
            # Get temporal window
            start_idx = max(0, i - radius)
            end_idx = min(len(frames), i + radius + 1)
            window_frames = frames[start_idx:end_idx]

            # Apply temporal median filtering
            temporal_denoised = np.median(window_frames, axis=0).astype(np.uint8)

            # Apply additional spatial denoising
            spatial_denoised = self._apply_spatial_denoise(temporal_denoised)

            denoised.append(spatial_denoised)

        return denoised

    def _apply_spatial_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply spatial denoising to single frame"""

        # Convert to float for processing
        frame_float = frame.astype(np.float32)

        # Apply bilateral filter for edge-preserving denoising
        denoised = cv2.bilateralFilter(
            frame_float,
            d=9,  # Diameter of pixel neighborhood
            sigmaColor=self.denoise_config["color_sigma"],
            sigmaSpace=self.denoise_config["spatial_sigma"]
        )

        # Apply luminance denoising
        lab = cv2.cvtColor(denoised.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Denoise luminance channel
        l_denoised = cv2.bilateralFilter(
            l.astype(np.float32),
            d=5,
            sigmaColor=self.denoise_config["luminance_sigma"],
            sigmaSpace=self.denoise_config["spatial_sigma"]
        )

        # Recombine channels
        lab_denoised = cv2.merge([l_denoised.astype(np.uint8), a, b])
        result = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)

        return result

    def _enhance_quality(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply quality enhancement (sharpening, contrast)"""

        enhanced = []

        for frame in frames:
            # Apply unsharp masking for sharpening
            gaussian = cv2.GaussianBlur(frame, (0, 0), 1.0)
            sharpened = cv2.addWeighted(frame, 1.0 + self.denoise_config["sharpen_strength"],
                                      gaussian, -self.denoise_config["sharpen_strength"], 0)

            # Apply subtle contrast enhancement
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Enhance luminance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            # Recombine
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            enhanced.append(result)

        return enhanced


class CinematicPolisher:
    """Cinematic post-processing and color grading"""

    def __init__(self):
        self.polish_config = {
            "color_grading": {
                "contrast": 1.1,
                "brightness": 5,
                "saturation": 1.05,
                "temperature": 2500,  # Warm tone
                "tint": 10
            },
            "film_grain": {
                "intensity": 0.02,
                "size": 1
            },
            "vignette": {
                "intensity": 0.1,
                "feather": 0.3
            },
            "bloom": {
                "intensity": 0.1,
                "threshold": 0.8
            }
        }

    def apply_cinematic_polish(self, frame_paths: List[str],
                             output_dir: str) -> List[str]:
        """Apply complete cinematic polish to frame sequence"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        polished_frames = []

        print(f"Applying cinematic polish to {len(frame_paths)} frames...")

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # Apply color grading
            polished = self._apply_color_grading(frame)

            # Apply film grain
            polished = self._apply_film_grain(polished)

            # Apply vignette
            polished = self._apply_vignette(polished)

            # Apply bloom effect
            polished = self._apply_bloom(polished)

            # Save polished frame
            output_file = output_path / "04d"
            cv2.imwrite(str(output_file), polished)
            polished_frames.append(str(output_file))

        print(f"Cinematic polish complete: {len(polished_frames)} frames")
        return polished_frames

    def _apply_color_grading(self, frame: np.ndarray) -> np.ndarray:
        """Apply professional color grading"""

        # Convert to LAB color space for better color control
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab.astype(np.float32))

        # Adjust contrast and brightness
        l = cv2.convertScaleAbs(l, alpha=self.polish_config["color_grading"]["contrast"],
                               beta=self.polish_config["color_grading"]["brightness"])

        # Adjust saturation (a and b channels)
        a = a * self.polish_config["color_grading"]["saturation"]
        b = b * self.polish_config["color_grading"]["saturation"]

        # Apply color temperature/tint adjustments
        temp_adjust = self.polish_config["color_grading"]["temperature"] / 6500.0  # Normalize to daylight
        tint_adjust = self.polish_config["color_grading"]["tint"] / 100.0

        # Warm/cool temperature adjustment
        if temp_adjust > 1.0:  # Warm
            b = b * (1 + (temp_adjust - 1.0) * 0.2)  # Increase blue-yellow axis
        elif temp_adjust < 1.0:  # Cool
            a = a * (1 + (1.0 - temp_adjust) * 0.2)  # Increase green-magenta axis

        # Green-magenta tint adjustment
        a = a + tint_adjust * 10

        # Clamp values
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

        # Recombine
        lab_corrected = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
        result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

        return result

    def _apply_film_grain(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle film grain effect"""

        intensity = self.polish_config["film_grain"]["intensity"]
        size = self.polish_config["film_grain"]["size"]

        # Generate noise
        noise = np.random.normal(0, intensity * 255, frame.shape).astype(np.float32)

        # Apply gaussian blur to noise for film-like grain
        noise = cv2.GaussianBlur(noise, (size * 2 + 1, size * 2 + 1), 0)

        # Add noise to frame
        result = frame.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle vignette effect"""

        intensity = self.polish_config["vignette"]["intensity"]
        feather = self.polish_config["vignette"]["feather"]

        height, width = frame.shape[:2]

        # Create vignette mask
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Radial distance from center
        radius = np.sqrt(xx**2 + yy**2)

        # Create vignette (darker at edges)
        vignette = 1 - intensity * np.clip(radius - feather, 0, 1) / (1 - feather)
        vignette = np.clip(vignette, 0, 1)

        # Apply vignette
        result = frame.astype(np.float32) * vignette[:, :, np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _apply_bloom(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle bloom/glow effect"""

        intensity = self.polish_config["bloom"]["intensity"]
        threshold = self.polish_config["bloom"]["threshold"]

        if intensity <= 0:
            return frame

        # Convert to float
        frame_float = frame.astype(np.float32) / 255.0

        # Extract bright areas
        bright_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        bright_mask = np.where(bright_mask > threshold, bright_mask, 0)

        # Blur bright areas for glow
        glow = cv2.GaussianBlur(bright_mask, (21, 21), 0)

        # Add glow back to original
        result = frame_float + glow[:, :, np.newaxis] * intensity
        result = np.clip(result, 0, 1)

        return (result * 255).astype(np.uint8)


class UpscalePipeline:
    """Complete upscale pipeline with denoising and cinematic polish"""

    def __init__(self):
        self.esrgan = get_esrgan_upscaler()
        self.tile_processor = get_tile_processor()
        self.denoise_engine = DenoiseEngine()
        self.cinematic_polisher = CinematicPolisher()

        self.pipeline_config = {
            "target_resolution": (1920, 1080),  # 1080p
            "apply_denoising": True,
            "apply_cinematic_polish": True,
            "tile_processing": True,
            "max_tile_size": 512
        }

    async def process_video_upscale(self, video_path: str,
                                  output_video: str,
                                  **kwargs) -> Dict[str, Any]:
        """Complete upscale pipeline: video → denoise → upscale → polish → final"""

        # Update config with kwargs
        config = self.pipeline_config.copy()
        config.update(kwargs)

        print("Starting upscale pipeline...")
        print(f"Input: {video_path}")
        print(f"Target resolution: {config['target_resolution']}")

        try:
            # Step 1: Extract frames from video
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = Path(temp_dir) / "frames"
                frames_dir.mkdir()

                frame_paths = self._extract_video_frames(video_path, str(frames_dir))

                if not frame_paths:
                    return {
                        "success": False,
                        "error": "Could not extract frames from video"
                    }

                # Step 2: Apply denoising if enabled
                processed_frames = frame_paths
                if config["apply_denoising"]:
                    denoised_dir = Path(temp_dir) / "denoised"
                    processed_frames = self.denoise_engine.denoise_frame_sequence(
                        processed_frames, str(denoised_dir)
                    )

                # Step 3: Upscale frames
                upscaled_dir = Path(temp_dir) / "upscaled"
                if config["tile_processing"]:
                    # Use tile processing for large images
                    processed_frames = self._upscale_with_tiles(
                        processed_frames, str(upscaled_dir), config["target_resolution"]
                    )
                else:
                    processed_frames = self.esrgan.upscale_video_frames(
                        processed_frames, str(upscaled_dir), config["target_resolution"]
                    )

                # Step 4: Apply cinematic polish if enabled
                if config["apply_cinematic_polish"]:
                    polished_dir = Path(temp_dir) / "polished"
                    processed_frames = self.cinematic_polisher.apply_cinematic_polish(
                        processed_frames, str(polished_dir)
                    )

                # Step 5: Create final video
                output_path = Path(output_video)
                output_path.parent.mkdir(exist_ok=True)

                # Get original video FPS
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                # Create video from processed frames
                from moviepy.editor import ImageSequenceClip

                clip = ImageSequenceClip(processed_frames, fps=fps)

                clip.write_videofile(
                    str(output_path),
                    fps=fps,
                    codec="libx264",
                    audio=False,  # Audio will be added separately
                    verbose=False,
                    logger=None
                )

                clip.close()

                # Calculate metrics
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                duration = len(processed_frames) / fps

                return {
                    "success": True,
                    "output_path": str(output_path),
                    "original_resolution": "auto-detected",
                    "final_resolution": config["target_resolution"],
                    "duration_seconds": duration,
                    "fps": fps,
                    "file_size_mb": file_size_mb,
                    "frames_processed": len(processed_frames),
                    "denoising_applied": config["apply_denoising"],
                    "cinematic_polish_applied": config["apply_cinematic_polish"],
                    "tile_processing_used": config["tile_processing"]
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Upscale pipeline failed: {str(e)}"
            }

    def _extract_video_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from video"""

        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_file = output_path / "04d"
            cv2.imwrite(str(frame_file), frame)
            frame_paths.append(str(frame_file))
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from video")
        return frame_paths

    def _upscale_with_tiles(self, frame_paths: List[str], output_dir: str,
                          target_resolution: Tuple[int, int]) -> List[str]:
        """Upscale frames using tile processing for memory efficiency"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        upscaled_frames = []

        print(f"Upscaling {len(frame_paths)} frames with tile processing...")

        for i, frame_path in enumerate(frame_paths):
            # Load image
            img = cv2.imread(frame_path)
            if img is None:
                continue

            # Split into tiles
            tiles = self.tile_processor.split_image_into_tiles(img)

            # Process each tile
            processed_tiles = []
            for tile_info in tiles:
                tile = tile_info["tile"]

                # Upscale tile (placeholder - would use ESRGAN)
                # For now, simple resize
                tile_resized = cv2.resize(tile, target_resolution[::-1],
                                        interpolation=cv2.INTER_LANCZOS4)

                tile_info["processed_tile"] = tile_resized
                processed_tiles.append(tile_info)

            # Merge tiles back
            result = self.tile_processor.merge_tiles(processed_tiles,
                                                   (target_resolution[1], target_resolution[0], 3))

            # Save result
            output_file = output_path / "04d"
            cv2.imwrite(str(output_file), result)
            upscaled_frames.append(str(output_file))

        return upscaled_frames

    def validate_upscale_quality(self, video_path: str) -> Dict[str, Any]:
        """Validate upscale quality and metrics"""

        video_file = Path(video_path)
        if not video_file.exists():
            return {"valid": False, "error": "Video file not found"}

        validation = {
            "file_exists": True,
            "file_size_mb": video_file.stat().st_size / (1024 * 1024),
            "valid": True,
            "quality_metrics": {}
        }

        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(video_path))

            validation["quality_metrics"].update({
                "duration": clip.duration,
                "fps": clip.fps,
                "resolution": clip.size,
                "aspect_ratio": clip.aspect_ratio
            })

            # Check resolution is 1080p
            width, height = clip.size
            is_1080p = width >= 1920 and height >= 1080
            validation["quality_metrics"]["is_1080p"] = is_1080p

            # Estimate quality score based on resolution and file size
            expected_size_mb = (width * height * clip.fps * clip.duration) / (1024 * 1024 * 8)  # Rough estimate
            size_ratio = validation["file_size_mb"] / expected_size_mb if expected_size_mb > 0 else 1

            validation["quality_metrics"]["compression_efficiency"] = size_ratio
            validation["quality_metrics"]["quality_score"] = min(1.0, size_ratio * (1 if is_1080p else 0.5))

            clip.close()

        except Exception as e:
            validation["valid"] = False
            validation["error"] = f"Quality validation failed: {e}"

        return validation


# Global pipeline instance
_upscale_pipeline = None


def get_upscale_pipeline() -> UpscalePipeline:
    """Get global upscale pipeline instance"""
    global _upscale_pipeline
    if _upscale_pipeline is None:
        _upscale_pipeline = UpscalePipeline()
    return _upscale_pipeline


def upscale_video_to_1080p(video_path: str, output_video: str = "upscaled_1080p.mp4",
                          **kwargs) -> Dict[str, Any]:
    """Convenience function for 1080p video upscaling"""

    pipeline = get_upscale_pipeline()

    # Run async pipeline
    async def run_pipeline():
        return await pipeline.process_video_upscale(video_path, output_video, **kwargs)

    return asyncio.run(run_pipeline())


def quick_test_upscale_pipeline():
    """Quick test of upscale pipeline components"""
    print("Testing upscale pipeline...")

    try:
        pipeline = get_upscale_pipeline()
        esrgan = pipeline.esrgan
        denoiser = pipeline.denoise_engine
        polisher = pipeline.cinematic_polisher

        print("✅ Upscale pipeline components initialized")
        print(f"   ESRGAN device: {esrgan.device}")
        print(f"   Target resolution: {pipeline.pipeline_config['target_resolution']}")
        print(f"   Denoising: {pipeline.pipeline_config['apply_denoising']}")
        print(f"   Cinematic polish: {pipeline.pipeline_config['apply_cinematic_polish']}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    quick_test_upscale_pipeline()