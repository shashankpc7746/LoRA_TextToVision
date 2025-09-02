#!/usr/bin/env python3
"""
Tier Router Module - Task-4 Day-1
Intelligently routes video generation tasks between local GPU, office GPU pool, and Yotta cloud
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import requests
import json


@dataclass
class RoutingDecision:
    """Routing decision result"""
    tier: str  # "local", "office_gpu", "yotta"
    reason: str
    estimated_cost: float
    estimated_latency: int
    confidence: float  # 0.0 to 1.0
    fallback_options: list


class TierRouter:
    """Smart routing logic for video generation tasks"""

    def __init__(self):
        # Tier configurations
        self.tiers = {
            "local": {
                "name": "Local GPU",
                "cost_per_minute": 0.0,  # Free (already paid for)
                "base_latency_ms": 5000,  # 5 seconds base
                "max_concurrent": 1,
                "supported_resolutions": ["360p", "480p", "512p", "640p"],
                "max_vram_gb": 8.0,  # Will be updated from device probe
                "availability": 1.0  # Always available locally
            },
            "office_gpu": {
                "name": "Office GPU Pool",
                "cost_per_minute": 0.02,  # Low cost for office GPUs
                "base_latency_ms": 8000,  # 8 seconds base
                "max_concurrent": 4,  # Multiple GPUs available
                "supported_resolutions": ["360p", "480p", "512p", "640p", "720p"],
                "max_vram_gb": 24.0,  # RTX 4090 level
                "availability": 0.9,  # 90% uptime during office hours
                "queue_endpoint": "http://192.168.0.100:8001/gpu-queue/status"  # BHIV GPU pool
            },
            "yotta": {
                "name": "Yotta Cloud GPU",
                "cost_per_minute": 0.15,  # Higher cost for cloud
                "base_latency_ms": 12000,  # 12 seconds base + network
                "max_concurrent": 100,  # Virtually unlimited
                "supported_resolutions": ["360p", "480p", "512p", "640p", "720p", "1080p"],
                "max_vram_gb": 80.0,  # A100/H100 level
                "availability": 0.99,  # 99% uptime
                "api_endpoint": "https://api.yotta.ai/v1/generate"
            }
        }

        self.current_load = {
            "local": 0,
            "office_gpu": 0,
            "yotta": 0
        }

    def route_task(self, device_capabilities: Dict[str, Any],
                  quality_settings: Dict[str, Any],
                  task_complexity: str = "medium",
                  user_preferences: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route task to optimal tier based on requirements and constraints

        Args:
            device_capabilities: Local device capabilities
            quality_settings: Planned quality settings
            task_complexity: Task complexity level
            user_preferences: User routing preferences

        Returns:
            RoutingDecision with optimal tier and reasoning
        """

        # Extract key parameters
        required_vram = quality_settings.get("estimated_vram_gb", 4.0)
        estimated_time = quality_settings.get("estimated_time_sec", 180)
        target_cost = quality_settings.get("estimated_cost_usd", 0.02)

        local_vram = device_capabilities.get("gpu_memory_gb", 4.0)
        can_handle_heavy = device_capabilities.get("can_handle_heavy_load", False)
        thermal_status = device_capabilities.get("thermal_status", "normal")
        battery_level = device_capabilities.get("battery_level")

        # Check user preferences
        prefer_local = user_preferences.get("prefer_local", True) if user_preferences else True
        max_cost = user_preferences.get("max_cost_usd", 0.10) if user_preferences else 0.10
        max_latency = user_preferences.get("max_latency_sec", 300) if user_preferences else 300

        # Decision logic
        decision = self._make_routing_decision(
            required_vram, estimated_time, target_cost,
            local_vram, can_handle_heavy, thermal_status, battery_level,
            task_complexity, prefer_local, max_cost, max_latency
        )

        return decision

    def _make_routing_decision(self, required_vram: float, estimated_time: int, target_cost: float,
                              local_vram: float, can_handle_heavy: bool, thermal_status: str,
                              battery_level: Optional[float], task_complexity: str,
                              prefer_local: bool, max_cost: float, max_latency: int) -> RoutingDecision:

        """Core routing decision logic"""

        # Tier 1: Try Local GPU First (if preferred and capable)
        if prefer_local and self._can_use_local(required_vram, local_vram, can_handle_heavy,
                                               thermal_status, battery_level):

            local_cost = self._calculate_cost("local", estimated_time)
            local_latency = self._calculate_latency("local", estimated_time)

            if local_cost <= max_cost and local_latency <= max_latency * 1000:
                return RoutingDecision(
                    tier="local",
                    reason="Local GPU available and within constraints",
                    estimated_cost=local_cost,
                    estimated_latency=local_latency,
                    confidence=0.95,
                    fallback_options=["office_gpu", "yotta"]
                )

        # Tier 2: Check Office GPU Pool
        if self._can_use_office_gpu(required_vram, estimated_time):
            office_cost = self._calculate_cost("office_gpu", estimated_time)
            office_latency = self._calculate_latency("office_gpu", estimated_time)

            if office_cost <= max_cost and office_latency <= max_latency * 1000:
                return RoutingDecision(
                    tier="office_gpu",
                    reason="Office GPU pool available for better performance",
                    estimated_cost=office_cost,
                    estimated_latency=office_latency,
                    confidence=0.90,
                    fallback_options=["yotta", "local"]
                )

        # Tier 3: Yotta Cloud (fallback)
        yotta_cost = self._calculate_cost("yotta", estimated_time)
        yotta_latency = self._calculate_latency("yotta", estimated_time)

        if yotta_cost <= max_cost and yotta_latency <= max_latency * 1000:
            return RoutingDecision(
                tier="yotta",
                reason="Using Yotta cloud for heavy/complex tasks",
                estimated_cost=yotta_cost,
                estimated_latency=yotta_latency,
                confidence=0.85,
                fallback_options=["office_gpu", "local"]
            )

        # Emergency fallback to local (even if suboptimal)
        local_cost = self._calculate_cost("local", estimated_time)
        local_latency = self._calculate_latency("local", estimated_time)

        return RoutingDecision(
            tier="local",
            reason="Emergency fallback to local GPU (constraints exceeded)",
            estimated_cost=local_cost,
            estimated_latency=local_latency,
            confidence=0.60,
            fallback_options=[]
        )

    def _can_use_local(self, required_vram: float, local_vram: float, can_handle_heavy: bool,
                      thermal_status: str, battery_level: Optional[float]) -> bool:
        """Check if local GPU can handle the task"""

        # VRAM check
        if required_vram > local_vram * 0.8:  # Use 80% of available VRAM
            return False

        # Heavy load capability
        if not can_handle_heavy and required_vram > 4.0:
            return False

        # Thermal constraints
        if thermal_status == "hot":
            return False

        # Battery constraints (for laptops)
        if battery_level and battery_level < 30:
            return False

        return True

    def _can_use_office_gpu(self, required_vram: float, estimated_time: int) -> bool:
        """Check if office GPU pool is suitable"""

        # Check VRAM availability
        office_config = self.tiers["office_gpu"]
        if required_vram > office_config["max_vram_gb"]:
            return False

        # Check queue status (simplified - in real implementation, check actual queue)
        try:
            # This would check the actual BHIV GPU queue
            # For now, assume office GPUs are available during reasonable hours
            current_hour = time.localtime().tm_hour
            if 9 <= current_hour <= 18:  # Office hours
                return True
            else:
                return False
        except:
            return False

    def _calculate_cost(self, tier: str, estimated_time_sec: int) -> float:
        """Calculate cost for given tier and time"""
        tier_config = self.tiers[tier]
        cost_per_minute = tier_config["cost_per_minute"]
        time_minutes = estimated_time_sec / 60.0
        return cost_per_minute * time_minutes

    def _calculate_latency(self, tier: str, estimated_time_sec: int) -> int:
        """Calculate total latency including base latency"""
        tier_config = self.tiers[tier]
        base_latency = tier_config["base_latency_ms"]
        processing_time = estimated_time_sec * 1000  # Convert to ms
        return base_latency + processing_time

    def update_load_status(self, tier: str, load_change: int):
        """Update current load status for a tier"""
        if tier in self.current_load:
            self.current_load[tier] = max(0, self.current_load[tier] + load_change)

    def get_tier_status(self) -> Dict[str, Any]:
        """Get current status of all tiers"""
        status = {}
        for tier_name, tier_config in self.tiers.items():
            status[tier_name] = {
                "name": tier_config["name"],
                "current_load": self.current_load[tier_name],
                "max_concurrent": tier_config["max_concurrent"],
                "availability": tier_config["availability"],
                "utilization": self.current_load[tier_name] / tier_config["max_concurrent"]
            }
        return status

    def check_office_gpu_queue(self) -> Dict[str, Any]:
        """Check actual office GPU queue status"""
        try:
            # In real implementation, this would query the BHIV GPU pool
            response = requests.get("http://192.168.0.100:8001/gpu-queue/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass

        # Fallback to simulated status
        return {
            "available_gpus": 3,
            "queue_length": 2,
            "estimated_wait_time": 30,  # seconds
            "status": "operational"
        }


# Global instance for easy access
tier_router = TierRouter()


def route_generation_task(device_capabilities: Dict[str, Any],
                         quality_settings: Dict[str, Any],
                         task_complexity: str = "medium",
                         user_preferences: Optional[Dict[str, Any]] = None) -> RoutingDecision:
    """Convenience function to route generation tasks"""
    return tier_router.route_task(device_capabilities, quality_settings,
                                 task_complexity, user_preferences)


if __name__ == "__main__":
    # Test the tier router
    from device_probe import get_device_capabilities
    from budget_planner import plan_video_quality

    print("[INFO] Testing Tier Router...")

    # Get device capabilities
    device_caps = get_device_capabilities()
    print(f"[INFO] Device: {device_caps['gpu_name']} ({device_caps['gpu_memory_gb']}GB VRAM)")

    # Test different scenarios
    test_scenarios = [
        ("simple", {"prefer_local": True}),
        ("medium", {"prefer_local": True}),
        ("complex", {"prefer_local": True}),
        ("complex", {"prefer_local": False, "max_cost_usd": 0.05}),
    ]

    for complexity, prefs in test_scenarios:
        print(f"\n[TEST] Scenario: {complexity} task with prefs {prefs}")

        # Plan quality first
        quality_settings = plan_video_quality(device_caps, complexity, prefs)

        # Route the task
        decision = tier_router.route_task(device_caps, quality_settings.__dict__, complexity, prefs)

        print(f"  Selected Tier: {decision.tier}")
        print(f"  Reason: {decision.reason}")
        print(f"  Estimated Cost: ${decision.estimated_cost:.3f}")
        print(f"  Estimated Latency: {decision.estimated_latency}ms")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Fallback Options: {decision.fallback_options}")

    # Show tier status
    print("\n[INFO] Current Tier Status:")
    status = tier_router.get_tier_status()
    for tier, info in status.items():
        print(f"  {tier}: {info['current_load']}/{info['max_concurrent']} load, {info['availability']:.1%} available")