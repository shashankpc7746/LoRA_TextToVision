"""
Compression Engine for Task 4 Day 2
CRF-based FFmpeg compression with quality presets
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import tempfile


@dataclass
class CompressionPreset:
    """Compression preset configuration"""
    name: str
    codec: str
    crf: int
    preset: str
    target_vmaf: float
    max_bitrate: str
    description: str


class CompressionEngine:
    """FFmpeg-based compression engine with CRF and quality presets"""

    def __init__(self):
        self.presets = self._get_default_presets()
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable"""
        # Try common locations
        common_paths = [
            "ffmpeg",  # In PATH
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",  # Windows
        ]

        for path in common_paths:
            try:
                result = subprocess.run(
                    [path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue

        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in PATH.")

    def _get_default_presets(self) -> Dict[str, CompressionPreset]:
        """Get default compression presets"""
        return {
            "mobile_fast": CompressionPreset(
                name="mobile_fast",
                codec="libx264",
                crf=24,
                preset="veryfast",
                target_vmaf=70.0,
                max_bitrate="1M",
                description="Fast encoding for mobile devices"
            ),
            "mobile_quality": CompressionPreset(
                name="mobile_quality",
                codec="libx264",
                crf=22,
                preset="fast",
                target_vmaf=75.0,
                max_bitrate="2M",
                description="Balanced quality for mobile"
            ),
            "desktop_standard": CompressionPreset(
                name="desktop_standard",
                codec="libx264",
                crf=20,
                preset="fast",
                target_vmaf=80.0,
                max_bitrate="5M",
                description="Standard desktop quality"
            ),
            "desktop_hd": CompressionPreset(
                name="desktop_hd",
                codec="libx264",
                crf=18,
                preset="slow",
                target_vmaf=85.0,
                max_bitrate="10M",
                description="High quality for desktop"
            ),
            "broadcast": CompressionPreset(
                name="broadcast",
                codec="libx264",
                crf=16,
                preset="slow",
                target_vmaf=90.0,
                max_bitrate="20M",
                description="Broadcast quality"
            ),
            "archive_av1": CompressionPreset(
                name="archive_av1",
                codec="libsvtav1",
                crf=35,
                preset="6",
                target_vmaf=85.0,
                max_bitrate="5M",
                description="AV1 for efficient archiving"
            )
        }

    def compress_video(
        self,
        input_path: str,
        output_path: str,
        preset_name: str = "desktop_standard",
        audio_bitrate: str = "128k",
        two_pass: bool = False
    ) -> Dict[str, Any]:
        """
        Compress video using specified preset

        Args:
            input_path: Path to input video
            output_path: Path to output video
            preset_name: Compression preset to use
            audio_bitrate: Audio bitrate (e.g., "128k")
            two_pass: Whether to use two-pass encoding

        Returns:
            Dict with compression results and metadata
        """
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset = self.presets[preset_name]

        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-i", input_path,
        ]

        # Video encoding options
        if preset.codec == "libx264":
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(preset.crf),
                "-preset", preset.preset,
                "-maxrate", preset.max_bitrate,
                "-bufsize", f"{int(preset.max_bitrate.rstrip('M')) * 2}M",
            ])
        elif preset.codec == "libsvtav1":
            cmd.extend([
                "-c:v", "libsvtav1",
                "-crf", str(preset.crf),
                "-preset", preset.preset,
            ])

        # Audio options
        cmd.extend([
            "-c:a", "aac",
            "-b:a", audio_bitrate,
        ])

        # Output
        cmd.append(output_path)

        # Execute compression
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            # Get output file info
            output_info = self._get_video_info(output_path)

            return {
                "success": True,
                "preset_used": preset_name,
                "input_path": input_path,
                "output_path": output_path,
                "compression_ratio": output_info.get("size_mb", 0) / self._get_file_size_mb(input_path),
                "output_info": output_info,
                "ffmpeg_command": " ".join(cmd)
            }

        except subprocess.TimeoutExpired:
            raise RuntimeError("Compression timed out")
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "preset_used": preset_name,
                "input_path": input_path,
                "output_path": output_path
            }

    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0.0

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information using ffprobe"""
        try:
            cmd = [
                self.ffmpeg_path.replace("ffmpeg", "ffprobe"),
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                data = json.loads(result.stdout)

                # Extract relevant info
                video_stream = None
                audio_stream = None

                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                    elif stream.get("codec_type") == "audio":
                        audio_stream = stream

                format_info = data.get("format", {})

                return {
                    "duration": float(format_info.get("duration", 0)),
                    "size_mb": float(format_info.get("size", 0)) / (1024 * 1024),
                    "bitrate": int(format_info.get("bit_rate", 0)),
                    "video_codec": video_stream.get("codec_name") if video_stream else None,
                    "video_width": video_stream.get("width") if video_stream else None,
                    "video_height": video_stream.get("height") if video_stream else None,
                    "video_fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
                    "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
                    "audio_channels": audio_stream.get("channels") if audio_stream else None,
                }

        except Exception as e:
            print(f"Warning: Failed to get video info: {e}")

        return {}

    def get_optimal_preset(self, device_class: str, quality_target: str = "balanced") -> str:
        """Get optimal preset for device class and quality target"""
        device_presets = {
            "mobile": ["mobile_fast", "mobile_quality"],
            "desktop": ["desktop_standard", "desktop_hd"],
            "laptop": ["desktop_standard", "mobile_quality"],
            "broadcast": ["broadcast", "desktop_hd"]
        }

        quality_map = {
            "fast": 0,
            "balanced": 1,
            "quality": 1,
            "high": 2
        }

        presets = device_presets.get(device_class, ["desktop_standard"])
        quality_idx = min(quality_map.get(quality_target, 1), len(presets) - 1)

        return presets[quality_idx]

    def batch_compress(
        self,
        input_paths: List[str],
        output_dir: str,
        preset_name: str = "auto",
        device_class: str = "desktop"
    ) -> List[Dict[str, Any]]:
        """Batch compress multiple videos"""
        results = []

        for input_path in input_paths:
            if preset_name == "auto":
                actual_preset = self.get_optimal_preset(device_class)
            else:
                actual_preset = preset_name

            input_name = Path(input_path).stem
            output_path = os.path.join(output_dir, f"{input_name}_compressed.mp4")

            result = self.compress_video(input_path, output_path, actual_preset)
            results.append(result)

        return results

    def add_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
        preset_name: str = "desktop_standard"
    ) -> Dict[str, Any]:
        """Add subtitles to video with compression"""
        preset = self.presets[preset_name]

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", video_path,
            "-i", subtitle_path,
            "-c:v", preset.codec,
            "-crf", str(preset.crf),
            "-preset", preset.preset,
            "-c:a", "copy",  # Copy audio
            "-c:s", "mov_text",  # Subtitle codec
            "-metadata:s:s:0", "language=eng",
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            return {
                "success": result.returncode == 0,
                "command": " ".join(cmd),
                "stderr": result.stderr if result.returncode != 0 else ""
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global compression engine instance
_compression_engine = None

def get_compression_engine() -> CompressionEngine:
    """Get global compression engine instance"""
    global _compression_engine
    if _compression_engine is None:
        _compression_engine = CompressionEngine()
    return _compression_engine