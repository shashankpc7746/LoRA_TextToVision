"""
BGM Manager for Task-6 Production Hardening
Background music mixing with ffmpeg for video generation pipeline
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import os


class BGMConfig:
    """Configuration for background music mixing"""
    def __init__(self):
        self.default_volume_bgm: float = 0.25  # 25% volume for background music
        self.fade_duration: float = 2.0  # 2 second fade transitions
        self.audio_format: str = "aac"
        self.audio_bitrate: str = "128k"
        self.sample_rate: int = 44100


class BGMManager:
    """Manages background music mixing for video generation"""

    def __init__(self, bgm_directory: str = "assets/bgm"):
        self.bgm_dir = Path(bgm_directory)
        self.config = BGMConfig()
        self.default_bgm_path = self.bgm_dir / "default_bed.mp3"

        # Ensure BGM directory exists
        self.bgm_dir.mkdir(parents=True, exist_ok=True)

    def mix_bgm(self, voice_path: str, bgm_path: Optional[str] = None,
                output_path: str = None, volume_bgm: Optional[float] = None) -> Dict[str, Any]:
        """
        Mix voice (speech) with background music using ffmpeg.

        Args:
            voice_path: Path to input speech audio
            bgm_path: Path to background music (optional, uses default if None)
            output_path: Path to output merged audio (optional, auto-generated)
            volume_bgm: Relative volume for BGM [0.0-1.0] (optional, uses config default)

        Returns:
            Dict with mixing results and metadata
        """
        import time
        start_time = time.time()

        try:
            # Use default BGM if not specified
            if bgm_path is None:
                bgm_path = str(self.default_bgm_path)
                if not Path(bgm_path).exists():
                    return {
                        "success": False,
                        "error": f"Default BGM file not found: {bgm_path}",
                        "processing_time": time.time() - start_time
                    }

            # Auto-generate output path if not specified
            if output_path is None:
                voice_stem = Path(voice_path).stem
                output_path = str(Path(voice_path).with_stem(f"{voice_stem}_with_bgm"))

            # Use config default volume if not specified
            volume_bgm = volume_bgm if volume_bgm is not None else self.config.default_volume_bgm

            # Build ffmpeg command for audio mixing
            voice = str(voice_path)
            bgm = str(bgm_path)
            outp = str(output_path)

            # FFmpeg filter complex for mixing with volume control
            # [1:a]volume=0.25[bgm];[0:a][bgm]amix=inputs=2:duration=shortest:dropout_transition=2
            cmd = [
                "ffmpeg", "-y",  # Overwrite output files
                "-i", voice,     # Input 1: voice
                "-i", bgm,       # Input 2: background music
                "-filter_complex",
                f"[1:a]volume={volume_bgm}[bgm];[0:a][bgm]amix=inputs=2:duration=shortest:dropout_transition={self.config.fade_duration}",
                "-c:a", self.config.audio_format,
                "-b:a", self.config.audio_bitrate,
                "-ar", str(self.config.sample_rate),
                outp
            ]

            # Execute ffmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            processing_time = time.time() - start_time

            if result.returncode == 0 and Path(output_path).exists():
                # Get file sizes for metadata
                voice_size = Path(voice_path).stat().st_size
                bgm_size = Path(bgm_path).stat().st_size
                output_size = Path(output_path).stat().st_size

                return {
                    "success": True,
                    "output_path": output_path,
                    "processing_time": processing_time,
                    "voice_file": voice_path,
                    "bgm_file": bgm_path,
                    "volume_bgm": volume_bgm,
                    "file_sizes": {
                        "voice_mb": voice_size / (1024 * 1024),
                        "bgm_mb": bgm_size / (1024 * 1024),
                        "output_mb": output_size / (1024 * 1024)
                    },
                    "audio_format": self.config.audio_format,
                    "audio_bitrate": self.config.audio_bitrate,
                    "sample_rate": self.config.sample_rate
                }
            else:
                return {
                    "success": False,
                    "error": f"FFmpeg failed: {result.stderr}",
                    "ffmpeg_command": " ".join(cmd),
                    "return_code": result.returncode,
                    "processing_time": processing_time
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "BGM mixing timed out",
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"BGM mixing failed: {str(e)}",
                "processing_time": time.time() - start_time
            }

    def get_available_bgm(self) -> Dict[str, Any]:
        """Get list of available background music files"""
        bgm_files = []
        if self.bgm_dir.exists():
            for file_path in self.bgm_dir.glob("*.mp3"):
                stat = file_path.stat()
                bgm_files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": stat.st_mtime
                })

        return {
            "bgm_directory": str(self.bgm_dir),
            "default_bgm": str(self.default_bgm_path),
            "available_files": bgm_files,
            "total_files": len(bgm_files)
        }

    def validate_bgm_file(self, bgm_path: str) -> Dict[str, Any]:
        """Validate a background music file"""
        try:
            path = Path(bgm_path)
            if not path.exists():
                return {"valid": False, "error": "File does not exist"}

            # Check file extension
            if path.suffix.lower() not in ['.mp3', '.wav', '.flac', '.aac']:
                return {"valid": False, "error": "Unsupported audio format"}

            # Get basic file info using ffprobe
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)

                # Extract audio stream info
                audio_stream = None
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        audio_stream = stream
                        break

                if audio_stream:
                    return {
                        "valid": True,
                        "duration": float(info.get("format", {}).get("duration", 0)),
                        "size_mb": path.stat().st_size / (1024 * 1024),
                        "codec": audio_stream.get("codec_name"),
                        "channels": audio_stream.get("channels"),
                        "sample_rate": audio_stream.get("sample_rate"),
                        "bitrate": audio_stream.get("bit_rate")
                    }
                else:
                    return {"valid": False, "error": "No audio stream found"}
            else:
                return {"valid": False, "error": f"FFprobe failed: {result.stderr}"}

        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}

    def update_config(self, **kwargs):
        """Update BGM configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# Global BGM manager instance
_bgm_manager = None

def get_bgm_manager() -> BGMManager:
    """Get global BGM manager instance"""
    global _bgm_manager
    if _bgm_manager is None:
        _bgm_manager = BGMManager()
    return _bgm_manager