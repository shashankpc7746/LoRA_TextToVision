"""
Quality Assessor for Task 4 Day 2
VMAF-based quality assessment and validation
"""

import os
import subprocess
import json
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class QualityMetrics:
    """Quality assessment results"""
    vmaf_score: float
    psnr_score: float
    ssim_score: float
    bitrate_kbps: float
    compression_ratio: float
    encoding_time_seconds: float
    file_size_mb: float


class QualityAssessor:
    """VMAF-based quality assessment engine"""

    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self.ffmpeg_path.replace("ffmpeg", "ffprobe")

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

    def assess_quality(
        self,
        original_path: str,
        compressed_path: str,
        sample_rate: float = 0.1
    ) -> QualityMetrics:
        """
        Assess video quality using VMAF, PSNR, and SSIM

        Args:
            original_path: Path to original video
            compressed_path: Path to compressed video
            sample_rate: Fraction of frames to sample (0.1 = 10%)

        Returns:
            QualityMetrics object with assessment results
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames from both videos
            original_frames = self._extract_frames(original_path, temp_dir, "original", sample_rate)
            compressed_frames = self._extract_frames(compressed_path, temp_dir, "compressed", sample_rate)

            if not original_frames or not compressed_frames:
                raise RuntimeError("Failed to extract frames from videos")

            # Calculate metrics
            vmaf_score = self._calculate_vmaf(original_frames, compressed_frames)
            psnr_score = self._calculate_psnr(original_frames, compressed_frames)
            ssim_score = self._calculate_ssim(original_frames, compressed_frames)

            # Get file info
            file_info = self._get_file_info(compressed_path)

            return QualityMetrics(
                vmaf_score=vmaf_score,
                psnr_score=psnr_score,
                ssim_score=ssim_score,
                bitrate_kbps=file_info["bitrate_kbps"],
                compression_ratio=file_info["compression_ratio"],
                encoding_time_seconds=0.0,  # Would need to be passed from compression
                file_size_mb=file_info["file_size_mb"]
            )

    def _extract_frames(self, video_path: str, output_dir: str, prefix: str, sample_rate: float) -> List[str]:
        """Extract frames from video for quality assessment"""
        frames_dir = os.path.join(output_dir, f"{prefix}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Get video info to determine frame count
        video_info = self._get_video_info(video_path)
        duration = video_info.get("duration", 0)
        fps = video_info.get("fps", 30)

        if duration == 0 or fps == 0:
            return []

        total_frames = int(duration * fps)
        sample_count = max(1, int(total_frames * sample_rate))

        # Extract frames at regular intervals
        frame_paths = []
        for i in range(sample_count):
            timestamp = (i / (sample_count - 1)) * duration if sample_count > 1 else 0

            frame_path = os.path.join(frames_dir, "06d")
            cmd = [
                self.ffmpeg_path,
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",  # High quality
                frame_path
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    frame_paths.append(frame_path)
            except:
                continue

        return frame_paths

    def _calculate_vmaf(self, original_frames: List[str], compressed_frames: List[str]) -> float:
        """Calculate VMAF score between frame pairs"""
        if len(original_frames) != len(compressed_frames):
            return 0.0

        vmaf_scores = []

        for orig, comp in zip(original_frames, compressed_frames):
            if not os.path.exists(orig) or not os.path.exists(comp):
                continue

            # Use FFmpeg's libvmaf filter
            cmd = [
                self.ffmpeg_path,
                "-i", comp,
                "-i", orig,
                "-lavfi", "[0:v][1:v]libvmaf=log_fmt=json:log_path=/dev/stdout",
                "-f", "null", "-"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    # Parse VMAF output from stderr
                    output = result.stderr
                    if '"vmaf":' in output:
                        # Extract VMAF score
                        start = output.find('"vmaf":') + 7
                        end = output.find(',', start)
                        if end == -1:
                            end = output.find('}', start)
                        vmaf_str = output[start:end].strip()
                        try:
                            vmaf_scores.append(float(vmaf_str))
                        except ValueError:
                            pass
            except:
                continue

        return np.mean(vmaf_scores) if vmaf_scores else 0.0

    def _calculate_psnr(self, original_frames: List[str], compressed_frames: List[str]) -> float:
        """Calculate PSNR between frame pairs"""
        if len(original_frames) != len(compressed_frames):
            return 0.0

        psnr_scores = []

        for orig, comp in zip(original_frames, compressed_frames):
            if not os.path.exists(orig) or not os.path.exists(comp):
                continue

            cmd = [
                self.ffmpeg_path,
                "-i", comp,
                "-i", orig,
                "-lavfi", "psnr=stats_file=-",
                "-f", "null", "-"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    # Parse PSNR from output
                    output = result.stderr
                    if 'PSNR' in output:
                        lines = output.split('\n')
                        for line in lines:
                            if 'PSNR' in line and 'avg:' in line:
                                try:
                                    psnr_str = line.split('avg:')[1].split()[0]
                                    psnr_scores.append(float(psnr_str))
                                    break
                                except (ValueError, IndexError):
                                    pass
            except:
                continue

        return np.mean(psnr_scores) if psnr_scores else 0.0

    def _calculate_ssim(self, original_frames: List[str], compressed_frames: List[str]) -> float:
        """Calculate SSIM between frame pairs"""
        if len(original_frames) != len(compressed_frames):
            return 0.0

        ssim_scores = []

        for orig, comp in zip(original_frames, compressed_frames):
            if not os.path.exists(orig) or not os.path.exists(comp):
                continue

            cmd = [
                self.ffmpeg_path,
                "-i", comp,
                "-i", orig,
                "-lavfi", "ssim=stats_file=-",
                "-f", "null", "-"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    # Parse SSIM from output
                    output = result.stderr
                    if 'SSIM' in output:
                        lines = output.split('\n')
                        for line in lines:
                            if 'SSIM' in line and 'All:' in line:
                                try:
                                    ssim_str = line.split('All:')[1].split()[0]
                                    ssim_scores.append(float(ssim_str))
                                    break
                                except (ValueError, IndexError):
                                    pass
            except:
                continue

        return np.mean(ssim_scores) if ssim_scores else 0.0

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                format_info = data.get("format", {})
                streams = data.get("streams", [])

                # Find video stream
                video_stream = None
                for stream in streams:
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                        break

                return {
                    "duration": float(format_info.get("duration", 0)),
                    "fps": eval(video_stream.get("r_frame_rate", "30/1")) if video_stream else 30,
                    "width": video_stream.get("width", 0) if video_stream else 0,
                    "height": video_stream.get("height", 0) if video_stream else 0,
                }

        except Exception as e:
            print(f"Warning: Failed to get video info: {e}")

        return {"duration": 0, "fps": 30, "width": 0, "height": 0}

    def _get_file_info(self, video_path: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            stat = os.stat(video_path)
            file_size_mb = stat.st_size / (1024 * 1024)

            # Get bitrate from ffprobe
            info = self._get_video_info(video_path)
            duration = info.get("duration", 0)

            # Estimate original size (rough approximation)
            # This would need the original file size to be passed in
            compression_ratio = 1.0  # Placeholder

            return {
                "file_size_mb": file_size_mb,
                "bitrate_kbps": (file_size_mb * 1024 * 1024 * 8) / (duration * 1000) if duration > 0 else 0,
                "compression_ratio": compression_ratio
            }

        except Exception as e:
            print(f"Warning: Failed to get file info: {e}")
            return {
                "file_size_mb": 0,
                "bitrate_kbps": 0,
                "compression_ratio": 1.0
            }

    def meets_quality_threshold(self, metrics: QualityMetrics, threshold: float = 70.0) -> bool:
        """Check if quality meets threshold"""
        return metrics.vmaf_score >= threshold

    def get_quality_recommendation(self, metrics: QualityMetrics) -> str:
        """Get quality recommendation based on metrics"""
        if metrics.vmaf_score >= 90:
            return "Excellent quality - no changes needed"
        elif metrics.vmaf_score >= 80:
            return "Good quality - minor adjustments possible"
        elif metrics.vmaf_score >= 70:
            return "Acceptable quality - consider slight improvements"
        elif metrics.vmaf_score >= 60:
            return "Poor quality - recommend re-encoding with higher bitrate"
        else:
            return "Very poor quality - significant improvements needed"


# Global quality assessor instance
_quality_assessor = None

def get_quality_assessor() -> QualityAssessor:
    """Get global quality assessor instance"""
    global _quality_assessor
    if _quality_assessor is None:
        _quality_assessor = QualityAssessor()
    return _quality_assessor