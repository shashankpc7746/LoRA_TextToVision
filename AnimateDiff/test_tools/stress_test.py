#!/usr/bin/env python3
"""
Stress Test Harness for Task-6 Production Hardening
Concurrent load testing for 50 users with aiohttp
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any
from datetime import datetime


class StressTestConfig:
    """Configuration for stress testing"""
    def __init__(self):
        self.preview_url = "http://localhost:8001/ttv/preview/generate"
        self.generate_url = "http://localhost:8001/ttv/generate"
        self.num_users = 50
        self.concurrency = 50
        self.test_duration_seconds = 30
        self.request_timeout = 120  # 2 minutes


class StressTester:
    """Handles concurrent stress testing of the API"""

    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None

    async def simulate_user_request(self, session: aiohttp.ClientSession, user_id: int,
                                  use_preview: bool = True) -> Dict[str, Any]:
        """Simulate a single user request"""
        request_start = time.time()

        try:
            url = self.config.preview_url if use_preview else self.config.generate_url

            # Create test payload
            payload = {
                "prompt": f"Stress test video generation for user {user_id} - {datetime.now().isoformat()}",
                "style": "realistic",
                "target_quality": "ultra_fast" if use_preview else "mobile_480p",
                "max_cost_usd": 0.05,
                "max_latency_sec": 30,
                "prefer_local": True
            }

            # Make request
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            ) as response:
                response_time = time.time() - request_start

                # Try to get response body
                try:
                    response_body = await response.text()
                    response_data = json.loads(response_body) if response_body else {}
                except:
                    response_data = {"error": "Failed to parse response"}

                return {
                    "user_id": user_id,
                    "success": response.status == 200,
                    "status_code": response.status,
                    "response_time": response_time,
                    "url": url,
                    "response_size": len(response_body) if 'response_body' in locals() else 0,
                    "timestamp": datetime.now().isoformat(),
                    "response_data": response_data
                }

        except asyncio.TimeoutError:
            return {
                "user_id": user_id,
                "success": False,
                "status_code": "timeout",
                "response_time": time.time() - request_start,
                "error": "Request timeout",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "user_id": user_id,
                "success": False,
                "status_code": "error",
                "response_time": time.time() - request_start,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def run_stress_test(self, use_preview: bool = True) -> Dict[str, Any]:
        """Run the complete stress test"""
        print(f"[StressTest] Starting {self.config.num_users} concurrent user simulation...")
        print(f"[StressTest] Target URL: {self.config.preview_url if use_preview else self.config.generate_url}")
        print(f"[StressTest] Test duration: {self.config.test_duration_seconds} seconds")

        self.start_time = time.time()

        async with aiohttp.ClientSession() as session:
            # Create concurrent tasks
            tasks = [
                self.simulate_user_request(session, i, use_preview)
                for i in range(self.config.num_users)
            ]

            # Run all tasks concurrently
            print(f"[StressTest] Executing {len(tasks)} concurrent requests...")
            self.results = await asyncio.gather(*tasks, return_exceptions=True)

        self.end_time = time.time()
        total_duration = self.end_time - self.start_time

        # Process results
        successful_requests = [r for r in self.results if isinstance(r, dict) and r.get("success", False)]
        failed_requests = [r for r in self.results if isinstance(r, dict) and not r.get("success", True)]

        # Calculate metrics
        success_rate = len(successful_requests) / len(self.results) * 100 if self.results else 0
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        throughput = len(self.results) / total_duration if total_duration > 0 else 0

        # Status code distribution
        status_distribution = {}
        for result in self.results:
            if isinstance(result, dict):
                status = result.get("status_code", "unknown")
                status_distribution[status] = status_distribution.get(status, 0) + 1

        # Performance analysis
        response_times = [r["response_time"] for r in successful_requests if isinstance(r, dict)]
        if response_times:
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
        else:
            p95_response_time = p99_response_time = 0

        test_summary = {
            "test_type": "preview_generation" if use_preview else "full_generation",
            "num_users": self.config.num_users,
            "concurrency": self.config.concurrency,
            "total_requests": len(self.results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "throughput_rps": throughput,
            "total_duration": total_duration,
            "status_distribution": status_distribution,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "test_passed": success_rate >= 95.0 and avg_response_time <= 10.0,  # Preview requirements
            "requirements": {
                "success_rate_threshold": 95.0,
                "avg_response_time_threshold": 10.0
            }
        }

        return test_summary

    def save_results(self, filename: str = "stress_test_results.json"):
        """Save test results to file"""
        output = {
            "summary": self.get_summary(),
            "detailed_results": self.results,
            "timestamp": datetime.now().isoformat()
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"[StressTest] Results saved to {filename}")

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        if not self.results:
            return {"error": "No test results available"}

        successful_requests = [r for r in self.results if isinstance(r, dict) and r.get("success", False)]
        success_rate = len(successful_requests) / len(self.results) * 100 if self.results else 0
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0

        return {
            "total_requests": len(self.results),
            "successful_requests": len(successful_requests),
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "test_passed": success_rate >= 95.0 and avg_response_time <= 10.0
        }

    def print_summary(self):
        """Print test summary to console"""
        if not self.results:
            print("[StressTest] No results to display")
            return

        summary = self.get_summary()
        print("\n" + "="*60)
        print("STRESS TEST RESULTS")
        print("="*60)
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Successful: {summary['successful_requests']}")
        print(".1f")
        print(".2f")
        print(f"Test Passed: {'✅ YES' if summary['test_passed'] else '❌ NO'}")
        print("="*60)


async def run_gradual_stress_test():
    """Run stress test with gradual scaling to prevent GPU crashes"""
    print("[StressTest] Task-6 Production Hardening - Gradual Stress Test")
    print("[StressTest] Testing with gradual scaling: 10 → 25 → 50 users")

    test_levels = [
        {"users": 10, "concurrency": 10, "name": "Level 1 (10 users)"},
        {"users": 25, "concurrency": 25, "name": "Level 2 (25 users)"},
        {"users": 50, "concurrency": 50, "name": "Level 3 (50 users)"}
    ]

    all_results = []
    overall_success = True

    for level in test_levels:
        print(f"\n{'='*60}")
        print(f"[StressTest] {level['name']}")
        print('='*60)

        # Configure test for this level
        config = StressTestConfig()
        config.num_users = level["users"]
        config.concurrency = level["concurrency"]
        config.test_duration_seconds = 30

        # Run test
        tester = StressTester(config)
        summary = await tester.run_stress_test(use_preview=True)

        # Print results
        tester.print_summary()

        # Store results
        all_results.append({
            "level": level["name"],
            "config": level,
            "summary": summary,
            "detailed_results": tester.results
        })

        # Check if this level passed
        if not summary.get("test_passed", False):
            print(f"[StressTest] ⚠️  {level['name']} failed requirements")
            overall_success = False
        else:
            print(f"[StressTest] ✅ {level['name']} passed")

        # Small delay between tests
        await asyncio.sleep(2)

    return all_results, overall_success

async def main():
    """Main stress test execution with gradual scaling"""
    print("[StressTest] Task-6 Production Hardening - Stress Test Harness")
    print("[StressTest] GPU-safe gradual scaling: 10 → 25 → 50 concurrent users")

    # Run gradual stress test
    all_results, overall_success = await run_gradual_stress_test()

    # Save comprehensive results
    output = {
        "test_type": "gradual_stress_test",
        "levels_tested": len(all_results),
        "overall_success": overall_success,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
        "gpu_safety_note": "Gradual scaling prevents GPU memory crashes"
    }

    with open("gradual_stress_test_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[StressTest] 📊 Comprehensive results saved to gradual_stress_test_results.json")

    # Final summary
    print("\n" + "="*60)
    print("GRADUAL STRESS TEST FINAL RESULTS")
    print("="*60)

    for result in all_results:
        level_name = result["level"]
        summary = result["summary"]
        success_rate = summary.get("success_rate", 0)
        avg_time = summary.get("average_response_time", 0)
        passed = summary.get("test_passed", False)

        status_icon = "✅" if passed else "❌"
        print(".1f")

    overall_status = "✅ ALL LEVELS PASSED" if overall_success else "❌ SOME LEVELS FAILED"
    print(f"\nOverall Result: {overall_status}")

    # Exit with appropriate code
    if overall_success:
        print("[StressTest] ✅ Gradual stress test PASSED - System ready for production")
        return 0
    else:
        print("[StressTest] ❌ Gradual stress test FAILED - Performance issues detected")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())