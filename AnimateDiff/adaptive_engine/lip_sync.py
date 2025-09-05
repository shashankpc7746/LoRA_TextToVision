"""
Lip-Sync Integration for Task 4 Day 3
Small model lip-sync with fallback to Wav2Lip
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class LipSyncConfig:
    """Lip-sync configuration"""
    model_size: str = "small"  # small, medium, large
    enhance_audio: bool = True
    face_detection_threshold: float = 0.5
    batch_size: int = 1
    use_gpu: bool = True
    fallback_enabled: bool = True


@dataclass
class LipSyncResult:
    """Lip-sync processing result"""
    success: bool
    output_path: Optional[str]
    processing_time: float
    confidence_score: float
    model_used: str
    error_message: Optional[str] = None


class LipSyncManager:
    """Manages lip-sync processing with small model and fallback"""

    def __init__(self, sadtalker_path: str = "SadTalker", wav2lip_path: str = "Wav2Lip"):
        self.sadtalker_path = Path(sadtalker_path)
        self.wav2lip_path = Path(wav2lip_path)
        self.config = LipSyncConfig()

        # Check available models
        self.sadtalker_available = self._check_sadtalker()
        self.wav2lip_available = self._check_wav2lip()

    def _check_sadtalker(self) -> bool:
        """Check if SadTalker is available"""
        try:
            # Check if SadTalker directory exists and has required files
            required_files = ["inference.py", "src"]
            return all((self.sadtalker_path / file).exists() for file in required_files)
        except:
            return False

    def _check_wav2lip(self) -> bool:
        """Check if Wav2Lip is available"""
        try:
            # Check if Wav2Lip directory exists
            return self.wav2lip_path.exists()
        except:
            return False

    def process_lip_sync(self, video_path: str, audio_path: str,
                        output_path: Optional[str] = None) -> LipSyncResult:
        """
        Process lip-sync for video and audio

        Args:
            video_path: Path to input video
            audio_path: Path to input audio
            output_path: Path to output video (optional)

        Returns:
            LipSyncResult with processing details
        """
        import time
        start_time = time.time()

        if not output_path:
            output_path = str(Path(video_path).with_stem(f"{Path(video_path).stem}_lipsync"))

        try:
            # Try SadTalker small model first
            if self.sadtalker_available:
                result = self._process_sadtalker(video_path, audio_path, output_path)
                if result.success:
                    processing_time = time.time() - start_time
                    result.processing_time = processing_time
                    return result

            # Fallback to Wav2Lip if available
            if self.wav2lip_available and self.config.fallback_enabled:
                print("[LipSync] SadTalker failed, trying Wav2Lip fallback...")
                result = self._process_wav2lip(video_path, audio_path, output_path)
                if result.success:
                    processing_time = time.time() - start_time
                    result.processing_time = processing_time
                    return result

            # Return failure result
            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                model_used="none",
                error_message="No lip-sync models available or all failed"
            )

        except Exception as e:
            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                model_used="error",
                error_message=str(e)
            )

    def _process_sadtalker(self, video_path: str, audio_path: str, output_path: str) -> LipSyncResult:
        """Process with SadTalker small model"""
        try:
            # SadTalker command for small model
            cmd = [
                "python",
                str(self.sadtalker_path / "inference.py"),
                "--driven_audio", audio_path,
                "--source_image", video_path,  # Will extract first frame
                "--result_dir", str(Path(output_path).parent),
                "--still", "false",  # Video input
                "--model_type", "small",  # Small model for speed
                "--batch_size", str(self.config.batch_size),
                "--face_model_resolution", "256",
                "--use_enhancer", "false",  # Skip enhancement for speed
                "--use_eye_blink", "true"
            ]

            # Run SadTalker
            result = subprocess.run(
                cmd,
                cwd=self.sadtalker_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                # Find output file (SadTalker creates files with specific naming)
                output_dir = Path(output_path).parent
                output_files = list(output_dir.glob("*_enhanced.mp4"))
                if output_files:
                    final_output = str(output_files[0])
                    return LipSyncResult(
                        success=True,
                        output_path=final_output,
                        processing_time=0.0,  # Will be set by caller
                        confidence_score=0.85,  # Estimated confidence
                        model_used="sadtalker_small"
                    )

            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=0.0,
                confidence_score=0.0,
                model_used="sadtalker_small",
                error_message=f"SadTalker failed: {result.stderr}"
            )

        except subprocess.TimeoutExpired:
            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=0.0,
                confidence_score=0.0,
                model_used="sadtalker_small",
                error_message="SadTalker timed out"
            )
        except Exception as e:
            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=0.0,
                confidence_score=0.0,
                model_used="sadtalker_small",
                error_message=f"SadTalker error: {str(e)}"
            )

    def _process_wav2lip(self, video_path: str, audio_path: str, output_path: str) -> LipSyncResult:
        """Process with Wav2Lip fallback"""
        try:
            # Wav2Lip command
            cmd = [
                "python",
                "inference.py",
                "--checkpoint_path", "wav2lip.pth",  # Assuming model file exists
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--batch_size", str(self.config.batch_size),
                "--face_det_batch_size", str(self.config.batch_size)
            ]

            # Run Wav2Lip
            result = subprocess.run(
                cmd,
                cwd=self.wav2lip_path,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            if result.returncode == 0 and Path(output_path).exists():
                return LipSyncResult(
                    success=True,
                    output_path=output_path,
                    processing_time=0.0,  # Will be set by caller
                    confidence_score=0.75,  # Lower confidence for fallback
                    model_used="wav2lip"
                )

            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=0.0,
                confidence_score=0.0,
                model_used="wav2lip",
                error_message=f"Wav2Lip failed: {result.stderr}"
            )

        except subprocess.TimeoutExpired:
            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=0.0,
                confidence_score=0.0,
                model_used="wav2lip",
                error_message="Wav2Lip timed out"
            )
        except Exception as e:
            return LipSyncResult(
                success=False,
                output_path=None,
                processing_time=0.0,
                confidence_score=0.0,
                model_used="wav2lip",
                error_message=f"Wav2Lip error: {str(e)}"
            )

    def extract_audio_from_video(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video file"""
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # WAV format
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                audio_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            return result.returncode == 0 and Path(audio_path).exists()

        except Exception as e:
            print(f"[LipSync] Audio extraction failed: {e}")
            return False

    def merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Merge audio and video files"""
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",  # Copy video codec
                "-c:a", "aac",   # Convert audio to AAC
                "-strict", "experimental",
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            return result.returncode == 0 and Path(output_path).exists()

        except Exception as e:
            print(f"[LipSync] Audio-video merge failed: {e}")
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of available lip-sync models"""
        return {
            "sadtalker_available": self.sadtalker_available,
            "sadtalker_path": str(self.sadtalker_path),
            "wav2lip_available": self.wav2lip_available,
            "wav2lip_path": str(self.wav2lip_path),
            "config": self.config.__dict__
        }

    def update_config(self, **kwargs):
        """Update lip-sync configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# Global lip-sync instance
_lip_sync = None

def get_lip_sync() -> LipSyncManager:
    """Get global lip-sync instance"""
    global _lip_sync
    if _lip_sync is None:
        _lip_sync = LipSyncManager()
    return _lip_sync