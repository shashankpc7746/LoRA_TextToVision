#!/usr/bin/env python3
"""
Budget Planner Module - Task-4 Day-1
Dynamically adjusts video generation quality based on device capabilities and cost constraints
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class QualitySettings:
    """Quality settings for video generation"""
    resolution: str = "512x512"  # width x height
    num_frames: int = 24
    fps: int = 12
    steps: int = 25
    guidance_scale: float = 15.0
    style: str = "realistic"
    estimated_vram_gb: float = 4.0
    estimated_time_sec: int = 180
    estimated_cost_usd: float = 0.02


@dataclass
class BudgetConstraints:
    """Budget and performance constraints"""
    max_latency_ms: int = 30000  # 30 seconds
    max_cost_usd: float = 0.05   # $0.05 per request
    target_quality: str = "balanced"  # fast/balanced/quality
    allow_yotta_fallback: bool = True


class BudgetPlanner:
    """Dynamic quality adjustment based on device capabilities and constraints"""

    def __init__(self, constraints: Optional[BudgetConstraints] = None):
        self.constraints = constraints or BudgetConstraints()

        # Quality presets for different scenarios
        self.quality_presets = {
            "ultra_fast": QualitySettings(
                resolution="360x360",
                num_frames=16,
                fps=8,
                steps=15,
                guidance_scale=12.0,
                estimated_vram_gb=2.0,
                estimated_time_sec=60,
                estimated_cost_usd=0.008
            ),
            "fast": QualitySettings(
                resolution="480x480",
                num_frames=20,
                fps=10,
                steps=20,
                guidance_scale=13.0,
                estimated_vram_gb=3.0,
                estimated_time_sec=90,
                estimated_cost_usd=0.015
            ),
            "balanced": QualitySettings(
                resolution="512x512",
                num_frames=24,
                fps=12,
                steps=25,
                guidance_scale=15.0,
                estimated_vram_gb=4.0,
                estimated_time_sec=180,
                estimated_cost_usd=0.025
            ),
            "quality": QualitySettings(
                resolution="512x512",
                num_frames=32,
                fps=12,
                steps=35,
                guidance_scale=16.0,
                estimated_vram_gb=6.0,
                estimated_time_sec=300,
                estimated_cost_usd=0.045
            ),
            "ultra_quality": QualitySettings(
                resolution="640x640",
                num_frames=32,
                fps=12,
                steps=50,
                guidance_scale=18.0,
                estimated_vram_gb=8.0,
                estimated_time_sec=600,
                estimated_cost_usd=0.08
            )
        }

    def plan_quality(self, device_capabilities: Dict[str, Any],
                    task_complexity: str = "medium",
                    user_preferences: Optional[Dict[str, Any]] = None) -> QualitySettings:
        """
        Plan optimal quality settings based on device capabilities and constraints

        Args:
            device_capabilities: Device capabilities from device_probe
            task_complexity: "simple", "medium", "complex"
            user_preferences: Optional user quality preferences

        Returns:
            Optimal QualitySettings for the given constraints
        """

        # Start with user's target quality or default to balanced
        target_quality = user_preferences.get("target_quality", "balanced") if user_preferences else "balanced"

        # Adjust based on device capabilities
        available_vram = device_capabilities.get("gpu_memory_gb", 4.0)
        can_handle_heavy = device_capabilities.get("can_handle_heavy_load", False)
        thermal_status = device_capabilities.get("thermal_status", "normal")
        battery_level = device_capabilities.get("battery_level")

        # Adjust for device limitations
        if not can_handle_heavy:
            # Device can't handle heavy loads, reduce quality
            if target_quality in ["ultra_quality", "quality"]:
                target_quality = "balanced"
            elif target_quality == "balanced":
                target_quality = "fast"

        # Adjust for VRAM constraints
        if available_vram < 6.0 and target_quality == "ultra_quality":
            target_quality = "quality"
        if available_vram < 4.0 and target_quality in ["quality", "balanced"]:
            target_quality = "fast"
        if available_vram < 3.0:
            target_quality = "ultra_fast"

        # Adjust for thermal/battery constraints
        if thermal_status == "hot" or (battery_level and battery_level < 30):
            # Reduce quality to prevent overheating/low battery
            quality_hierarchy = ["ultra_fast", "fast", "balanced", "quality", "ultra_quality"]
            current_index = quality_hierarchy.index(target_quality)
            target_quality = quality_hierarchy[max(0, current_index - 1)]

        # Adjust for task complexity
        if task_complexity == "simple" and target_quality in ["quality", "ultra_quality"]:
            # Simple tasks don't need ultra quality
            target_quality = "balanced"
        elif task_complexity == "complex" and target_quality in ["ultra_fast", "fast"]:
            # Complex tasks need better quality
            target_quality = "balanced"

        # Get base settings
        settings = self.quality_presets[target_quality].__dict__.copy()

        # Apply task-specific adjustments
        settings = self._apply_task_adjustments(settings, task_complexity)

        # Apply user preferences
        if user_preferences:
            settings = self._apply_user_preferences(settings, user_preferences)

        # Final validation against budget constraints
        settings = self._validate_budget_constraints(settings)

        return QualitySettings(**settings)

    def _apply_task_adjustments(self, settings: Dict[str, Any], complexity: str) -> Dict[str, Any]:
        """Apply task-specific quality adjustments"""
        if complexity == "simple":
            # Simple tasks can use faster settings
            settings["steps"] = max(15, settings["steps"] - 5)
            settings["guidance_scale"] = max(10.0, settings["guidance_scale"] - 2.0)
            settings["estimated_time_sec"] = int(settings["estimated_time_sec"] * 0.7)
            settings["estimated_cost_usd"] = settings["estimated_cost_usd"] * 0.7

        elif complexity == "complex":
            # Complex tasks need more quality
            settings["steps"] = min(50, settings["steps"] + 5)
            settings["guidance_scale"] = min(20.0, settings["guidance_scale"] + 1.0)
            settings["estimated_time_sec"] = int(settings["estimated_time_sec"] * 1.3)
            settings["estimated_cost_usd"] = settings["estimated_cost_usd"] * 1.3

        return settings

    def _apply_user_preferences(self, settings: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user-specific preferences"""
        # Style preference
        if "style" in preferences:
            settings["style"] = preferences["style"]

        # Speed vs quality preference
        if "priority" in preferences:
            if preferences["priority"] == "speed":
                settings["steps"] = max(15, settings["steps"] - 10)
                settings["num_frames"] = max(16, settings["num_frames"] - 8)
                settings["estimated_time_sec"] = int(settings["estimated_time_sec"] * 0.6)
            elif preferences["priority"] == "quality":
                settings["steps"] = min(50, settings["steps"] + 10)
                settings["guidance_scale"] = min(20.0, settings["guidance_scale"] + 2.0)
                settings["estimated_time_sec"] = int(settings["estimated_time_sec"] * 1.5)

        # Resolution preference
        if "resolution" in preferences:
            settings["resolution"] = preferences["resolution"]

        return settings

    def _validate_budget_constraints(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust settings against budget constraints"""
        # Check latency constraint
        if settings["estimated_time_sec"] * 1000 > self.constraints.max_latency_ms:
            # Reduce quality to meet latency requirements
            settings["steps"] = max(15, settings["steps"] - 10)
            settings["num_frames"] = max(16, settings["num_frames"] - 8)
            settings["estimated_time_sec"] = int(settings["estimated_time_sec"] * 0.7)

        # Check cost constraint
        if settings["estimated_cost_usd"] > self.constraints.max_cost_usd:
            # Reduce quality to meet cost requirements
            settings["steps"] = max(15, settings["steps"] - 5)
            settings["guidance_scale"] = max(10.0, settings["guidance_scale"] - 1.0)
            settings["estimated_cost_usd"] = settings["estimated_cost_usd"] * 0.8

        return settings

    def estimate_cost_and_time(self, settings: QualitySettings) -> Dict[str, Any]:
        """Estimate cost and time for given settings"""
        return {
            "estimated_time_sec": settings.estimated_time_sec,
            "estimated_cost_usd": settings.estimated_cost_usd,
            "estimated_vram_gb": settings.estimated_vram_gb,
            "within_budget": settings.estimated_cost_usd <= self.constraints.max_cost_usd,
            "within_latency": settings.estimated_time_sec * 1000 <= self.constraints.max_latency_ms
        }

    def get_available_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all available quality presets"""
        return {name: preset.__dict__ for name, preset in self.quality_presets.items()}


# Global instance for easy access
budget_planner = BudgetPlanner()


def plan_video_quality(device_capabilities: Dict[str, Any],
                      task_complexity: str = "medium",
                      user_preferences: Optional[Dict[str, Any]] = None) -> QualitySettings:
    """Convenience function to plan video quality"""
    return budget_planner.plan_quality(device_capabilities, task_complexity, user_preferences)


if __name__ == "__main__":
    # Test the budget planner
    from device_probe import get_device_capabilities

    print("[INFO] Testing Budget Planner...")

    # Get device capabilities
    device_caps = get_device_capabilities()
    print(f"[INFO] Device: {device_caps['gpu_name']} ({device_caps['gpu_memory_gb']}GB VRAM)")

    # Test different scenarios
    test_scenarios = [
        ("simple", {"target_quality": "balanced"}),
        ("medium", {"target_quality": "quality"}),
        ("complex", {"target_quality": "ultra_quality"}),
        ("medium", {"priority": "speed"}),
        ("medium", {"priority": "quality"})
    ]

    for complexity, prefs in test_scenarios:
        print(f"\n[TEST] Scenario: {complexity} task with prefs {prefs}")
        settings = budget_planner.plan_quality(device_caps, complexity, prefs)

        print(f"  Resolution: {settings.resolution}")
        print(f"  Frames: {settings.num_frames}, Steps: {settings.steps}")
        print(f"  Estimated time: {settings.estimated_time_sec}s")
        print(f"  Estimated cost: ${settings.estimated_cost_usd:.3f}")
        print(f"  Estimated VRAM: {settings.estimated_vram_gb}GB")

        # Check constraints
        estimate = budget_planner.estimate_cost_and_time(settings)
        print(f"  Within budget: {estimate['within_budget']}")
        print(f"  Within latency: {estimate['within_latency']}")