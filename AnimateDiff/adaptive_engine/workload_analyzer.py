#!/usr/bin/env python3
"""
Workload Analyzer Module - Task-4 Day-1
Analyzes task complexity to determine optimal processing tier and quality settings
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class TaskAnalysis:
    """Task complexity analysis result"""
    complexity: str  # "simple", "medium", "complex"
    confidence: float  # 0.0 to 1.0
    factors: Dict[str, Any]
    recommended_tier: str
    estimated_vram_gb: float
    estimated_time_sec: int
    reasoning: List[str]


class WorkloadAnalyzer:
    """Analyzes video generation task complexity"""

    def __init__(self):
        # Complexity scoring thresholds
        self.complexity_thresholds = {
            "simple": {"max_score": 30, "tier": "local"},
            "medium": {"max_score": 60, "tier": "local"},
            "complex": {"min_score": 61, "tier": "office_gpu"}
        }

        # Factor weights for complexity scoring
        self.factor_weights = {
            "text_length": 0.15,
            "scene_count": 0.20,
            "style_complexity": 0.15,
            "resolution_requirement": 0.20,
            "animation_complexity": 0.15,
            "quality_requirement": 0.15
        }

    def analyze_task(self, prompt: str, style: str = "realistic",
                    target_quality: str = "balanced",
                    additional_params: Optional[Dict[str, Any]] = None) -> TaskAnalysis:
        """
        Analyze task complexity based on prompt, style, and requirements

        Args:
            prompt: The text prompt for video generation
            style: Visual style (realistic, anime, artistic)
            target_quality: Desired quality level
            additional_params: Additional parameters

        Returns:
            TaskAnalysis with complexity assessment
        """

        factors = {}
        reasoning = []

        # Analyze text complexity
        text_score, text_factors = self._analyze_text_complexity(prompt)
        factors.update(text_factors)

        # Analyze style complexity
        style_score, style_factors = self._analyze_style_complexity(style)
        factors.update(style_factors)

        # Analyze quality requirements
        quality_score, quality_factors = self._analyze_quality_requirements(target_quality)
        factors.update(quality_factors)

        # Analyze additional parameters
        if additional_params:
            param_score, param_factors = self._analyze_additional_params(additional_params)
            factors.update(param_factors)
        else:
            param_score = 0

        # Calculate total complexity score
        total_score = (
            text_score * self.factor_weights["text_length"] +
            style_score * self.factor_weights["style_complexity"] +
            quality_score * self.factor_weights["quality_requirement"] +
            param_score * 0.1  # Additional params have lower weight
        )

        # Determine complexity level
        complexity = self._determine_complexity(total_score)

        # Calculate resource requirements
        estimated_vram, estimated_time = self._estimate_resources(complexity, factors)

        # Determine recommended tier
        recommended_tier = self._recommend_tier(complexity, estimated_vram)

        # Build reasoning
        reasoning.append(f"Total complexity score: {total_score:.1f}/100")
        reasoning.append(f"Primary factors: {', '.join(factors.get('primary_factors', []))}")

        if estimated_vram > 6.0:
            reasoning.append("High VRAM requirement suggests GPU acceleration needed")
        if estimated_time > 300:
            reasoning.append("Long processing time may benefit from distributed processing")

        return TaskAnalysis(
            complexity=complexity,
            confidence=min(0.95, 0.7 + (total_score / 200)),  # Higher score = higher confidence
            factors=factors,
            recommended_tier=recommended_tier,
            estimated_vram_gb=estimated_vram,
            estimated_time_sec=estimated_time,
            reasoning=reasoning
        )

    def _analyze_text_complexity(self, prompt: str) -> Tuple[float, Dict[str, Any]]:
        """Analyze text prompt complexity"""
        factors = {}

        # Text length analysis
        word_count = len(prompt.split())
        char_count = len(prompt)

        if word_count < 10:
            text_score = 10
            factors["text_density"] = "low"
        elif word_count < 25:
            text_score = 25
            factors["text_density"] = "medium"
        else:
            text_score = 50
            factors["text_density"] = "high"

        factors["word_count"] = word_count
        factors["char_count"] = char_count

        # Scene count estimation (based on sentence structure)
        sentences = re.split(r'[.!?]+', prompt)
        sentences = [s.strip() for s in sentences if s.strip()]
        scene_count = max(1, len(sentences))

        if scene_count == 1:
            scene_score = 10
        elif scene_count <= 3:
            scene_score = 25
        else:
            scene_score = 40

        factors["scene_count"] = scene_count
        factors["scene_complexity"] = "low" if scene_count <= 2 else "high"

        # Animation complexity (keywords indicating motion)
        animation_keywords = [
            "walk", "run", "move", "dance", "fly", "swim", "jump", "fight",
            "battle", "chase", "transform", "grow", "shrink", "spin", "rotate"
        ]

        animation_matches = sum(1 for keyword in animation_keywords if keyword.lower() in prompt.lower())
        animation_score = min(30, animation_matches * 8)

        factors["animation_keywords"] = animation_matches
        factors["animation_complexity"] = "low" if animation_matches <= 2 else "high"

        total_text_score = (
            text_score * 0.4 +
            scene_score * 0.4 +
            animation_score * 0.2
        )

        factors["primary_factors"] = []
        if text_score >= 40:
            factors["primary_factors"].append("long text")
        if scene_score >= 30:
            factors["primary_factors"].append("multiple scenes")
        if animation_score >= 20:
            factors["primary_factors"].append("complex animation")

        return total_text_score, factors

    def _analyze_style_complexity(self, style: str) -> Tuple[float, Dict[str, Any]]:
        """Analyze visual style complexity"""
        factors = {}

        style_complexity_map = {
            "realistic": {"score": 20, "complexity": "medium"},
            "anime": {"score": 30, "complexity": "medium"},
            "artistic": {"score": 35, "complexity": "high"},
            "cartoon": {"score": 25, "complexity": "medium"},
            "fantasy": {"score": 40, "complexity": "high"},
            "photorealistic": {"score": 45, "complexity": "high"}
        }

        style_info = style_complexity_map.get(style.lower(), {"score": 25, "complexity": "medium"})
        factors["style_type"] = style
        factors["style_complexity"] = style_info["complexity"]

        return style_info["score"], factors

    def _analyze_quality_requirements(self, target_quality: str) -> Tuple[float, Dict[str, Any]]:
        """Analyze quality requirements"""
        factors = {}

        quality_complexity_map = {
            "ultra_fast": {"score": 10, "resolution": "360p", "complexity": "low"},
            "fast": {"score": 20, "resolution": "480p", "complexity": "low"},
            "balanced": {"score": 30, "resolution": "512p", "complexity": "medium"},
            "quality": {"score": 45, "resolution": "640p", "complexity": "high"},
            "ultra_quality": {"score": 60, "resolution": "720p", "complexity": "high"}
        }

        quality_info = quality_complexity_map.get(target_quality, {"score": 30, "resolution": "512p", "complexity": "medium"})
        factors["target_quality"] = target_quality
        factors["target_resolution"] = quality_info["resolution"]
        factors["quality_complexity"] = quality_info["complexity"]

        return quality_info["score"], factors

    def _analyze_additional_params(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Analyze additional parameters"""
        factors = {}
        param_score = 0

        # Frame count
        num_frames = params.get("num_frames", 24)
        if num_frames > 32:
            param_score += 15
            factors["high_frame_count"] = True
        elif num_frames < 16:
            param_score -= 5

        # Custom resolution
        if "resolution" in params:
            resolution = params["resolution"]
            if "720" in resolution or "1080" in resolution:
                param_score += 20
                factors["high_resolution"] = True

        # Advanced settings
        if params.get("advanced_settings", False):
            param_score += 10
            factors["advanced_settings"] = True

        factors["additional_params_score"] = param_score
        return param_score, factors

    def _determine_complexity(self, total_score: float) -> str:
        """Determine complexity level from score"""
        if total_score <= self.complexity_thresholds["simple"]["max_score"]:
            return "simple"
        elif total_score <= self.complexity_thresholds["medium"]["max_score"]:
            return "medium"
        else:
            return "complex"

    def _estimate_resources(self, complexity: str, factors: Dict[str, Any]) -> Tuple[float, int]:
        """Estimate VRAM and time requirements"""
        base_requirements = {
            "simple": {"vram": 3.0, "time": 90},
            "medium": {"vram": 5.0, "time": 180},
            "complex": {"vram": 7.0, "time": 300}
        }

        requirements = base_requirements[complexity]

        # Adjust based on factors
        if factors.get("quality_complexity") == "high":
            requirements["vram"] += 1.0
            requirements["time"] += 60

        if factors.get("animation_complexity") == "high":
            requirements["vram"] += 0.5
            requirements["time"] += 30

        if factors.get("scene_count", 1) > 3:
            requirements["time"] += 60

        return requirements["vram"], requirements["time"]

    def _recommend_tier(self, complexity: str, estimated_vram: float) -> str:
        """Recommend processing tier based on complexity"""
        if complexity == "simple":
            return "local"
        elif complexity == "medium":
            return "local" if estimated_vram <= 6.0 else "office_gpu"
        else:  # complex
            return "office_gpu" if estimated_vram <= 12.0 else "yotta"


# Global instance for easy access
workload_analyzer = WorkloadAnalyzer()


def analyze_generation_task(prompt: str, style: str = "realistic",
                           target_quality: str = "balanced",
                           additional_params: Optional[Dict[str, Any]] = None) -> TaskAnalysis:
    """Convenience function to analyze generation tasks"""
    return workload_analyzer.analyze_task(prompt, style, target_quality, additional_params)


if __name__ == "__main__":
    # Test the workload analyzer
    print("[INFO] Testing Workload Analyzer...")

    test_prompts = [
        ("A cat sitting on a mat", "realistic", "fast"),  # Simple
        ("A young wizard casting a spell in an enchanted forest with magical effects and floating particles",
         "fantasy", "balanced"),  # Medium
        ("An epic battle scene with multiple characters, complex animations, dramatic lighting, and detailed backgrounds showing a fantasy world with castles, dragons, and magical effects throughout the entire scene",
         "fantasy", "ultra_quality"),  # Complex
    ]

    for prompt, style, quality in test_prompts:
        print(f"\n[TEST] Analyzing: {prompt[:50]}...")
        print(f"  Style: {style}, Quality: {quality}")

        analysis = workload_analyzer.analyze_task(prompt, style, quality)

        print(f"  Complexity: {analysis.complexity} (confidence: {analysis.confidence:.2f})")
        print(f"  Recommended Tier: {analysis.recommended_tier}")
        print(f"  Estimated VRAM: {analysis.estimated_vram_gb}GB")
        print(f"  Estimated Time: {analysis.estimated_time_sec}s")
        print(f"  Reasoning: {'; '.join(analysis.reasoning)}")

        # Show key factors
        factors = analysis.factors
        print(f"  Key Factors:")
        print(f"    - Words: {factors.get('word_count', 'N/A')}")
        print(f"    - Scenes: {factors.get('scene_count', 'N/A')}")
        print(f"    - Animation keywords: {factors.get('animation_keywords', 'N/A')}")
        print(f"    - Style complexity: {factors.get('style_complexity', 'N/A')}")