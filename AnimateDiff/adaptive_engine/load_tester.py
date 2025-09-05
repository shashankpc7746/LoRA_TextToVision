"""
Load Testing and Scaling System for Task 4 Day 4
Simulates 50 concurrent users with graceful degradation and Yotta fallback
"""

import asyncio
import time
import random
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import statistics


@dataclass
class LoadTestResult:
    """Result of a load test run"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_rps: float
    error_rate: float
    degradation_events: int
    yotta_fallbacks: int
    duration_seconds: float


@dataclass
class SimulatedUser:
    """Simulated user for load testing"""
    user_id: str
    request_count: int = 0
    success_count: int = 0
    total_response_time: float = 0.0
    last_request_time: float = 0.0
    tier_used: str = "unknown"


class LoadTester:
    """Load testing system for concurrent user simulation"""

    def __init__(self, max_concurrent_users: int = 50, test_duration_seconds: int = 300):
        self.max_concurrent_users = max_concurrent_users
        self.test_duration_seconds = test_duration_seconds
        self.active_users: Dict[str, SimulatedUser] = {}
        self.response_times: List[float] = []
        self.errors: List[str] = []
        self.degradation_events = 0
        self.yotta_fallbacks = 0

        # Load test configuration
        self.request_interval_min = 1.0  # Minimum seconds between requests
        self.request_interval_max = 5.0  # Maximum seconds between requests
        self.failure_probability = 0.05  # 5% chance of simulated failure

        # Graceful degradation thresholds
        self.high_load_threshold = 30  # Users
        self.critical_load_threshold = 45  # Users
        self.yotta_fallback_threshold = 48  # Users (lower threshold for testing)

    async def run_load_test(self) -> LoadTestResult:
        """Run the complete load test"""
        print(f"[Load Test] Starting test with {self.max_concurrent_users} concurrent users")
        print(f"[Load Test] Test duration: {self.test_duration_seconds} seconds")

        start_time = time.time()
        tasks = []

        # Create user simulation tasks
        for user_id in range(self.max_concurrent_users):
            user = SimulatedUser(user_id=f"user_{user_id:03d}")
            self.active_users[user.user_id] = user
            tasks.append(self._simulate_user(user))

        # Run all user simulations concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate results
        end_time = time.time()
        duration = end_time - start_time

        total_requests = sum(user.request_count for user in self.active_users.values())
        successful_requests = sum(user.success_count for user in self.active_users.values())
        failed_requests = total_requests - successful_requests

        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            median_response_time = statistics.median(self.response_times)
            p95_response_time = statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0.0

        throughput_rps = total_requests / duration if duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0

        result = LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            degradation_events=self.degradation_events,
            yotta_fallbacks=self.yotta_fallbacks,
            duration_seconds=duration
        )

        print(f"[Load Test] Test completed in {duration:.2f} seconds")
        print(f"[Load Test] Results: {successful_requests}/{total_requests} successful ({error_rate:.1f}% error rate)")
        print(f"[Load Test] Throughput: {throughput_rps:.2f} RPS")
        print(f"[Load Test] Degradation events: {self.degradation_events}")
        print(f"[Load Test] Yotta fallbacks: {self.yotta_fallbacks}")

        return result

    async def _simulate_user(self, user: SimulatedUser):
        """Simulate a single user's behavior"""
        test_end_time = time.time() + self.test_duration_seconds

        while time.time() < test_end_time:
            # Check if we should degrade service
            current_load = len([u for u in self.active_users.values() if time.time() - u.last_request_time < 10])
            tier = self._determine_tier(current_load)

            # Simulate request
            request_start = time.time()
            success = await self._simulate_request(user, tier)
            request_end = time.time()

            # Record metrics
            response_time = request_end - request_start
            self.response_times.append(response_time)

            user.request_count += 1
            user.total_response_time += response_time
            user.last_request_time = request_start
            user.tier_used = tier

            if success:
                user.success_count += 1
            else:
                self.errors.append(f"User {user.user_id} request failed")

            # Wait before next request
            wait_time = random.uniform(self.request_interval_min, self.request_interval_max)
            await asyncio.sleep(wait_time)

    def _determine_tier(self, current_load: int) -> str:
        """Determine which tier to use based on current load"""
        # Integrate with degradation manager
        from .load_tester import get_degradation_manager
        degradation_manager = get_degradation_manager()

        # Assess current load level
        degradation_level = degradation_manager.assess_load(current_load, 0)  # Queue length not tracked here

        if current_load >= self.yotta_fallback_threshold:
            self.yotta_fallbacks += 1
            return "yotta"
        elif current_load >= self.critical_load_threshold:
            self.degradation_events += 1
            return "office_gpu"
        elif current_load >= self.high_load_threshold:
            self.degradation_events += 1
            return "office_gpu"
        else:
            return "local"

    async def _simulate_request(self, user: SimulatedUser, tier: str) -> bool:
        """Simulate a video generation request"""
        # Simulate different response times based on tier
        tier_delays = {
            "local": (2.0, 5.0),      # 2-5 seconds
            "office_gpu": (3.0, 8.0), # 3-8 seconds
            "yotta": (5.0, 15.0)      # 5-15 seconds
        }

        min_delay, max_delay = tier_delays.get(tier, (2.0, 5.0))
        delay = random.uniform(min_delay, max_delay)

        # Simulate occasional failures
        if random.random() < self.failure_probability:
            await asyncio.sleep(delay * 0.5)  # Faster failure
            return False

        await asyncio.sleep(delay)
        return True

    def get_load_stats(self) -> Dict[str, Any]:
        """Get current load testing statistics"""
        total_requests = sum(user.request_count for user in self.active_users.values())
        total_success = sum(user.success_count for user in self.active_users.values())

        tier_usage = {}
        for user in self.active_users.values():
            tier = user.tier_used
            if tier not in tier_usage:
                tier_usage[tier] = 0
            tier_usage[tier] += 1

        return {
            "active_users": len(self.active_users),
            "total_requests": total_requests,
            "successful_requests": total_success,
            "error_rate": ((total_requests - total_success) / total_requests * 100) if total_requests > 0 else 0,
            "tier_usage": tier_usage,
            "degradation_events": self.degradation_events,
            "yotta_fallbacks": self.yotta_fallbacks,
            "average_response_time": statistics.mean(self.response_times) if self.response_times else 0
        }


class GracefulDegradationManager:
    """Manages graceful degradation under high load"""

    def __init__(self):
        self.degradation_levels = {
            "normal": {
                "quality": "high",
                "max_concurrent": 20,
                "features": ["full_lipsync", "high_quality", "multiple_styles"]
            },
            "moderate": {
                "quality": "medium",
                "max_concurrent": 35,
                "features": ["basic_lipsync", "medium_quality", "limited_styles"]
            },
            "high": {
                "quality": "low",
                "max_concurrent": 45,
                "features": ["no_lipsync", "low_quality", "single_style"]
            },
            "critical": {
                "quality": "minimal",
                "max_concurrent": 50,
                "features": ["static_images", "text_only"]
            }
        }

        self.current_level = "normal"
        self.load_history = []

    def assess_load(self, current_users: int, queue_length: int) -> str:
        """Assess current load and determine degradation level"""
        # Record load history
        self.load_history.append((time.time(), current_users, queue_length))
        if len(self.load_history) > 100:  # Keep last 100 measurements
            self.load_history.pop(0)

        # Determine degradation level
        if current_users >= 45 or queue_length >= 20:
            new_level = "critical"
        elif current_users >= 35 or queue_length >= 10:
            new_level = "high"
        elif current_users >= 25 or queue_length >= 5:
            new_level = "moderate"
        else:
            new_level = "normal"

        # Update level if changed
        if new_level != self.current_level:
            print(f"[Degradation] Level changed: {self.current_level} -> {new_level}")
            print(f"[Degradation] Current users: {current_users}, Queue: {queue_length}")
            self.current_level = new_level

        return self.current_level

    def get_degradation_config(self) -> Dict[str, Any]:
        """Get current degradation configuration"""
        return self.degradation_levels[self.current_level]

    def should_use_yotta_fallback(self, current_users: int, queue_length: int) -> bool:
        """Determine if Yotta cloud fallback should be used"""
        return current_users >= 48 or queue_length >= 15

    def get_scaling_recommendations(self) -> List[str]:
        """Get scaling recommendations based on current load"""
        recommendations = []

        if len(self.load_history) < 10:
            return ["Collecting load data..."]

        # Analyze load trends
        recent_loads = [load for _, load, _ in self.load_history[-10:]]
        avg_load = statistics.mean(recent_loads)
        max_load = max(recent_loads)

        if avg_load > 40:
            recommendations.append("Consider scaling office GPU pool")
        if max_load > 45:
            recommendations.append("Enable Yotta cloud fallback")
        if avg_load > 30:
            recommendations.append("Implement request queuing")
        if len(recent_loads) > 5:
            trend = statistics.mean(recent_loads[-3:]) - statistics.mean(recent_loads[:3])
            if trend > 5:
                recommendations.append("Load increasing - prepare for scaling")

        return recommendations if recommendations else ["System operating normally"]


# Global instances
_load_tester = None
_degradation_manager = None

def get_load_tester() -> LoadTester:
    """Get global load tester instance"""
    global _load_tester
    if _load_tester is None:
        _load_tester = LoadTester()
    return _load_tester

def get_degradation_manager() -> GracefulDegradationManager:
    """Get global degradation manager instance"""
    global _degradation_manager
    if _degradation_manager is None:
        _degradation_manager = GracefulDegradationManager()
    return _degradation_manager