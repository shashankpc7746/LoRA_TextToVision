"""
Test script for Task 4 Day 4 components
50 concurrent users, graceful degradation, Yotta fallback, analytics
"""

import asyncio
import time
from pathlib import Path

# Import Day 4 components
from adaptive_engine.load_tester import get_load_tester, get_degradation_manager
from adaptive_engine.analytics import get_analytics, RequestMetrics


def test_load_simulation():
    """Test load simulation components"""
    print("\n=== Testing Load Simulation Components ===")

    load_tester = get_load_tester()
    degradation_manager = get_degradation_manager()

    # Test degradation levels
    print("Testing degradation levels...")
    levels = []
    for users in [10, 25, 40, 48]:
        level = degradation_manager.assess_load(users, 0)
        levels.append(level)
        print(f"  {users} users -> {level}")

    # Test tier determination
    print("Testing tier determination...")
    tiers = []
    for load in [10, 25, 40, 48]:
        tier = load_tester._determine_tier(load)
        tiers.append(tier)
        print(f"  Load {load} -> {tier}")

    # Test load stats
    stats = load_tester.get_load_stats()
    print(f"Load stats: {stats}")

    # Validate basic functionality
    assert "normal" in levels, "Should have normal degradation level"
    assert "yotta" in tiers, "Should use Yotta tier under high load"
    assert isinstance(stats, dict), "Should return stats dictionary"

    print("[OK] Load simulation components test passed")


def test_graceful_degradation():
    """Test graceful degradation system"""
    print("\n=== Testing Graceful Degradation ===")

    degradation_manager = get_degradation_manager()

    # Test different load levels
    test_scenarios = [
        (10, 2, "normal"),    # Low load
        (25, 5, "moderate"),  # Medium load
        (40, 8, "high"),      # High load
        (48, 12, "critical")  # Critical load
    ]

    for users, queue, expected_level in test_scenarios:
        level = degradation_manager.assess_load(users, queue)
        config = degradation_manager.get_degradation_config()

        print(f"Load: {users} users, {queue} queued -> Level: {level}")
        print(f"  Quality: {config['quality']}")
        print(f"  Max concurrent: {config['max_concurrent']}")
        print(f"  Features: {config['features']}")

        assert level == expected_level, f"Expected {expected_level}, got {level}"

    # Test Yotta fallback
    should_fallback = degradation_manager.should_use_yotta_fallback(50, 15)
    assert should_fallback, "Should trigger Yotta fallback at high load"

    print("[OK] Graceful degradation test passed")


def test_analytics_system():
    """Test analytics and reporting system"""
    print("\n=== Testing Analytics System ===")

    analytics = get_analytics()

    # Record some test metrics
    test_requests = [
        RequestMetrics(
            request_id=f"test_req_{i}",
            timestamp=time.time() - (i * 60),  # Spread over time
            user_id=f"user_{i % 5}",
            tier_used=["local", "office_gpu", "yotta"][i % 3],
            response_time_seconds=2.0 + (i % 3),
            cost_usd=0.01 + (i % 3) * 0.02,
            success=i % 10 != 0,  # 90% success rate
            metadata={"test": True}
        )
        for i in range(20)
    ]

    for request in test_requests:
        analytics.record_request(request)

    # Generate reports
    cost_report = analytics.generate_cost_report(days=1)
    latency_report = analytics.generate_latency_report(hours=1)

    print("Cost Report (1 day):")
    print(f"  Total cost: ${cost_report.total_cost_usd:.3f}")
    print(f"  Cost per request: ${cost_report.cost_per_request:.3f}")
    print(f"  Cost by tier: {cost_report.cost_by_tier}")
    print(f"  Efficiency score: {cost_report.cost_efficiency_score:.1f}/100")
    print(f"  Recommendations: {cost_report.recommendations}")

    print("\nLatency Report (1 hour):")
    print(f"  Average latency: {latency_report.average_latency_seconds:.2f}s")
    print(f"  P95 latency: {latency_report.p95_latency_seconds:.2f}s")
    print(f"  P99 latency: {latency_report.p99_latency_seconds:.2f}s")
    print(f"  Latency by tier: {latency_report.latency_by_tier}")
    print(f"  Performance score: {latency_report.performance_score:.1f}/100")
    print(f"  Bottlenecks: {latency_report.bottlenecks}")

    # Test system health
    health = analytics.get_system_health()
    print(f"\nSystem Health: {health['status']} - {health['message']}")

    # Validate reports
    assert cost_report.total_cost_usd > 0, "Should have cost data"
    assert latency_report.average_latency_seconds > 0, "Should have latency data"
    assert health['status'] in ['healthy', 'warning', 'critical'], "Should have valid health status"

    print("[OK] Analytics system test passed")


def test_integration():
    """Test Day 4 component integration"""
    print("\n=== Testing Day 4 Integration ===")

    # Get all managers
    load_tester = get_load_tester()
    degradation_manager = get_degradation_manager()
    analytics = get_analytics()

    # Test load stats
    load_stats = load_tester.get_load_stats()
    print(f"Load stats: {load_stats}")

    # Test degradation recommendations
    recommendations = degradation_manager.get_scaling_recommendations()
    print(f"Scaling recommendations: {recommendations}")

    # Test analytics health
    health = analytics.get_system_health()
    print(f"System health: {health}")

    # Test component communication
    assert 'active_users' in load_stats, "Load tester should provide user stats"
    assert isinstance(recommendations, list), "Degradation manager should provide recommendations"
    assert 'status' in health, "Analytics should provide health status"

    print("[OK] Day 4 integration test passed")


def main():
    """Run all Day 4 tests"""
    print("Starting Task 4 Day 4 Component Tests")
    print("=" * 50)

    try:
        test_load_simulation()
        test_graceful_degradation()
        test_analytics_system()
        test_integration()

        print("\n" + "=" * 50)
        print("All Day 4 tests passed successfully!")
        print("[OK] Load Simulation Components: Working")
        print("[OK] Graceful Degradation: Working")
        print("[OK] Yotta Fallback: Implemented")
        print("[OK] Cost/Latency Reporting: Active")
        print("[OK] Analytics System: Functional")

    except Exception as e:
        print(f"\n[ERROR] Day 4 tests failed: {e}")
        raise


if __name__ == "__main__":
    main()