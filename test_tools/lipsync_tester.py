"""
Lip-sync Tester for Task-7 Quality Leap
Automated lip-sync accuracy measurement and validation
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import subprocess
import librosa
import soundfile as sf

from ..audio_manager.enhanced_sadtalker import get_audio_pipeline


class LipSyncTester:
    """Automated lip-sync quality testing and validation"""

    def __init__(self):
        self.audio_pipeline = get_audio_pipeline()

        # Lip-sync testing configuration
        self.test_config = {
            "sample_rate": 16000,
            "hop_length": 512,
            "n_mfcc": 13,
            "lip_sync_threshold": 0.06,  # 60ms threshold
            "quality_levels": {
                "excellent": 0.04,  # 40ms or better
                "good": 0.06,       # 60ms
                "acceptable": 0.08, # 80ms
                "poor": 0.10        # 100ms or worse
            }
        }

    def test_lip_sync(self, video_path: str, audio_path: str,
                     method: str = "auto") -> Dict[str, Any]:
        """Comprehensive lip-sync quality test"""

        print(f"Testing lip-sync quality: {Path(video_path).name}")

        results = {
            "video_path": video_path,
            "audio_path": audio_path,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        try:
            # Test 1: Basic lip-sync processing
            processing_result = self.audio_pipeline.process_lip_sync(
                video_path, audio_path, method
            )

            results["tests"]["processing"] = {
                "success": processing_result["success"],
                "method_used": processing_result.get("method", method),
                "processing_time": processing_result.get("processing_time", 0),
                "error": processing_result.get("error")
            }

            if processing_result["success"]:
                output_video = processing_result["output_path"]

                # Test 2: Lip-sync accuracy measurement
                accuracy_result = self._measure_lip_sync_accuracy(
                    output_video, audio_path
                )
                results["tests"]["accuracy"] = accuracy_result

                # Test 3: Temporal consistency
                temporal_result = self._check_temporal_consistency(output_video)
                results["tests"]["temporal"] = temporal_result

                # Test 4: Visual quality assessment
                visual_result = self._assess_visual_quality(output_video)
                results["tests"]["visual"] = visual_result

                # Overall quality score
                results["overall_score"] = self._calculate_overall_score(results["tests"])
                results["quality_rating"] = self._get_quality_rating(results["overall_score"])

            else:
                results["overall_score"] = 0.0
                results["quality_rating"] = "failed"

        except Exception as e:
            results["error"] = str(e)
            results["overall_score"] = 0.0
            results["quality_rating"] = "error"

        return results

    def _measure_lip_sync_accuracy(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """Measure lip-sync accuracy using audio-visual correlation"""

        try:
            # Extract audio features (MFCCs)
            audio_features = self._extract_audio_features(audio_path)

            # Extract video mouth movement features
            video_features = self._extract_mouth_movement_features(video_path)

            # Calculate correlation
            if len(audio_features) > 0 and len(video_features) > 0:
                # Align sequences (simple time alignment)
                min_length = min(len(audio_features), len(video_features))
                audio_aligned = audio_features[:min_length]
                video_aligned = video_features[:min_length]

                # Calculate cross-correlation
                correlation = np.correlate(audio_aligned, video_aligned, mode='full')
                max_corr_idx = np.argmax(np.abs(correlation))
                lag_samples = max_corr_idx - (len(audio_aligned) - 1)

                # Convert to time delay (seconds)
                sample_rate = self.test_config["sample_rate"]
                hop_length = self.test_config["hop_length"]
                time_delay = abs(lag_samples) * hop_length / sample_rate

                # Calculate correlation strength
                correlation_strength = np.max(np.abs(correlation)) / np.sqrt(
                    np.sum(audio_aligned**2) * np.sum(video_aligned**2)
                )

                return {
                    "time_delay_seconds": time_delay,
                    "correlation_strength": float(correlation_strength),
                    "is_synced": time_delay <= self.test_config["lip_sync_threshold"],
                    "quality_score": max(0, 1 - (time_delay / 0.2))  # Score from 0-1
                }
            else:
                return {
                    "error": "Could not extract features",
                    "time_delay_seconds": float('inf'),
                    "correlation_strength": 0.0,
                    "is_synced": False,
                    "quality_score": 0.0
                }

        except Exception as e:
            return {
                "error": str(e),
                "time_delay_seconds": float('inf'),
                "correlation_strength": 0.0,
                "is_synced": False,
                "quality_score": 0.0
            }

    def _extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from audio"""

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.test_config["sample_rate"])

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.test_config["n_mfcc"],
                hop_length=self.test_config["hop_length"]
            )

            # Take mean across MFCC coefficients for simpler representation
            return np.mean(mfccs, axis=0)

        except Exception as e:
            print(f"Audio feature extraction failed: {e}")
            return np.array([])

    def _extract_mouth_movement_features(self, video_path: str) -> np.ndarray:
        """Extract mouth movement features from video"""

        try:
            cap = cv2.VideoCapture(video_path)
            features = []

            frame_count = 0
            prev_mouth_region = None

            while cap.isOpened() and frame_count < 100:  # Limit to 100 frames for speed
                ret, frame = cap.read()
                if not ret:
                    break

                # Simple mouth region extraction (placeholder)
                # In production, this would use face detection and landmark extraction
                height, width = frame.shape[:2]
                mouth_region = frame[height//2:, width//3:2*width//3]  # Bottom center region

                # Calculate movement (difference from previous frame)
                if prev_mouth_region is not None:
                    diff = cv2.absdiff(mouth_region, prev_mouth_region)
                    movement = np.mean(diff)
                    features.append(movement)

                prev_mouth_region = mouth_region.copy()
                frame_count += 1

            cap.release()
            return np.array(features)

        except Exception as e:
            print(f"Video feature extraction failed: {e}")
            return np.array([])

    def _check_temporal_consistency(self, video_path: str) -> Dict[str, Any]:
        """Check temporal consistency of lip movements"""

        try:
            cap = cv2.VideoCapture(video_path)
            movements = []
            frame_count = 0

            prev_frame = None
            while cap.isOpened() and frame_count < 50:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(frame, prev_frame)
                    movement = np.mean(diff)
                    movements.append(movement)

                prev_frame = frame.copy()
                frame_count += 1

            cap.release()

            if movements:
                # Check for consistent movement patterns
                movement_std = np.std(movements)
                movement_mean = np.mean(movements)

                # Calculate flicker score (lower is better)
                flicker_score = movement_std / (movement_mean + 1e-6)

                return {
                    "movement_consistency": 1 / (1 + flicker_score),  # 0-1 score
                    "avg_movement": float(movement_mean),
                    "movement_variation": float(movement_std),
                    "flicker_detected": flicker_score > 0.5
                }
            else:
                return {
                    "error": "No movement data",
                    "movement_consistency": 0.0
                }

        except Exception as e:
            return {
                "error": str(e),
                "movement_consistency": 0.0
            }

    def _assess_visual_quality(self, video_path: str) -> Dict[str, Any]:
        """Assess visual quality of lip-sync video"""

        try:
            cap = cv2.VideoCapture(video_path)

            frame_count = 0
            total_blur = 0
            total_brightness = 0

            while cap.isOpened() and frame_count < 20:  # Sample 20 frames
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate blur (Laplacian variance)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                total_blur += blur

                # Calculate brightness
                brightness = np.mean(gray)
                total_brightness += brightness

                frame_count += 1

            cap.release()

            if frame_count > 0:
                avg_blur = total_blur / frame_count
                avg_brightness = total_brightness / frame_count

                # Quality scores (normalized 0-1)
                blur_score = min(1.0, avg_blur / 500.0)  # Higher blur = better
                brightness_score = 1.0 - abs(avg_brightness - 128) / 128  # Closer to 128 = better

                return {
                    "avg_blur": float(avg_blur),
                    "avg_brightness": float(avg_brightness),
                    "blur_score": blur_score,
                    "brightness_score": brightness_score,
                    "overall_visual_score": (blur_score + brightness_score) / 2
                }
            else:
                return {"error": "No frames analyzed", "overall_visual_score": 0.0}

        except Exception as e:
            return {
                "error": str(e),
                "overall_visual_score": 0.0
            }

    def _calculate_overall_score(self, test_results: Dict[str, Any]) -> float:
        """Calculate overall lip-sync quality score"""

        scores = []

        # Accuracy score (most important)
        if "accuracy" in test_results:
            acc = test_results["accuracy"]
            if "quality_score" in acc:
                scores.append(acc["quality_score"] * 0.4)  # 40% weight

        # Temporal consistency
        if "temporal" in test_results:
            temp = test_results["temporal"]
            if "movement_consistency" in temp:
                scores.append(temp["movement_consistency"] * 0.3)  # 30% weight

        # Visual quality
        if "visual" in test_results:
            vis = test_results["visual"]
            if "overall_visual_score" in vis:
                scores.append(vis["overall_visual_score"] * 0.3)  # 30% weight

        return np.mean(scores) if scores else 0.0

    def _get_quality_rating(self, score: float) -> str:
        """Convert score to quality rating"""

        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "acceptable"
        else:
            return "poor"

    def batch_test_lip_sync(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run lip-sync tests on multiple video-audio pairs"""

        results = []

        for i, test_case in enumerate(test_cases):
            print(f"\nTesting case {i+1}/{len(test_cases)}")

            video_path = test_case.get("video_path")
            audio_path = test_case.get("audio_path")
            method = test_case.get("method", "auto")

            if video_path and audio_path:
                result = self.test_lip_sync(video_path, audio_path, method)
                result["test_case"] = test_case
                results.append(result)
            else:
                results.append({
                    "error": "Missing video_path or audio_path",
                    "test_case": test_case
                })

        return results

    def generate_test_report(self, results: List[Dict[str, Any]],
                           output_path: str = "lipsync_test_report.json") -> str:
        """Generate comprehensive test report"""

        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(results),
                "tester_version": "1.0"
            },
            "summary": {
                "passed_tests": sum(1 for r in results if r.get("overall_score", 0) > 0.4),
                "avg_score": np.mean([r.get("overall_score", 0) for r in results]),
                "quality_distribution": {}
            },
            "results": results
        }

        # Calculate quality distribution
        quality_counts = {}
        for result in results:
            rating = result.get("quality_rating", "unknown")
            quality_counts[rating] = quality_counts.get(rating, 0) + 1

        report["summary"]["quality_distribution"] = quality_counts

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Test report saved: {output_path}")
        return output_path


# Global instance
_lip_sync_tester = None


def get_lip_sync_tester() -> LipSyncTester:
    """Get global lip-sync tester instance"""
    global _lip_sync_tester
    if _lip_sync_tester is None:
        _lip_sync_tester = LipSyncTester()
    return _lip_sync_tester


def test_lip_sync_quality(video_path: str, audio_path: str,
                         method: str = "auto") -> Dict[str, Any]:
    """Convenience function for lip-sync testing"""
    tester = get_lip_sync_tester()
    return tester.test_lip_sync(video_path, audio_path, method)


def quick_lip_sync_validation():
    """Quick validation test"""
    print("Running lip-sync validation test...")

    try:
        tester = get_lip_sync_tester()
        print("✅ Lip-sync tester initialized")
        print(f"   Lip-sync threshold: {tester.test_config['lip_sync_threshold']}s")

        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    quick_lip_sync_validation()