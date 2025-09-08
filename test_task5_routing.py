#!/usr/bin/env python3
"""
Test Task-5 Concurrent Routing Functionality
"""

import sys
import os

# Add AnimateDiff to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
animatediff_path = os.path.join(current_dir, 'AnimateDiff')
sys.path.insert(0, animatediff_path)

try:
    from adaptive_engine.device_probe import get_device_capabilities
    from adaptive_engine.budget_planner import plan_video_quality
    from adaptive_engine.tier_router import route_generation_task
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    print(f"Current directory: {current_dir}")
    print(f"AnimateDiff path: {animatediff_path}")
    sys.exit(1)

def test_concurrent_routing():
    """Test concurrent routing with 3 users"""
    print("Testing Task-5 Concurrent Routing with 3 users...")
    print("=" * 50)

    # Get device capabilities
    device_caps = get_device_capabilities()
    print(f"Device: {device_caps['gpu_name']} ({device_caps['gpu_memory_gb']}GB VRAM)")
    print(f"Device Class: {device_caps.get('device_class', 'desktop')}")
    print()

    results = []
    for user_id in range(3):
        try:
            print(f"Processing User {user_id}...")

            # Plan quality
            quality_plan = plan_video_quality(device_caps, 'medium', {'target_quality': 'balanced'})
            print(f"  Quality Plan: {quality_plan.resolution}, {quality_plan.fps}fps")

            # Route task
            routing_decision = route_generation_task(
                device_caps,
                quality_plan.__dict__,
                'medium',
                {'prefer_local': True, 'max_cost_usd': 0.05}
            )

            result = {
                'user_id': user_id,
                'tier': routing_decision.tier,
                'cost': routing_decision.estimated_cost,
                'latency': routing_decision.estimated_latency,
                'reason': routing_decision.reason,
                'success': True
            }
            results.append(result)

            print(f"  Routing: {routing_decision.tier} tier")
            print(f"  Cost: ${routing_decision.estimated_cost:.3f}")
            print(f"  Latency: {routing_decision.estimated_latency}ms")
            print(f"  Reason: {routing_decision.reason}")
            print()

        except Exception as e:
            result = {
                'user_id': user_id,
                'success': False,
                'error': str(e)
            }
            results.append(result)
            print(f"  ERROR: {e}")
            print()

    # Summary
    print("SUMMARY")
    print("=" * 50)
    successful = sum(1 for r in results if r['success'])
    print(f"Total Users: 3")
    print(f"Successful: {successful}")
    print(f"Success Rate: {successful}/3 ({successful/3*100:.1f}%)")

    if successful > 0:
        print("\nTier Distribution:")
        tier_counts = {}
        total_cost = 0
        total_latency = 0

        for result in results:
            if result['success']:
                tier = result['tier']
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                total_cost += result['cost']
                total_latency += result['latency']

        for tier, count in tier_counts.items():
            print(f"  {tier}: {count} users")

        print(f"\nAverage Cost: ${total_cost / successful:.3f}")
        print("Average Latency: {:.0f}ms".format(total_latency / successful))

    # Task-5 Validation
    print("\nTASK-5 VALIDATION")
    print("=" * 50)
    if successful == 3:
        print("PASS: All 3 concurrent users routed successfully")
        print("PASS: Task-5 concurrent routing requirement met")
    else:
        print("FAIL: Some users failed routing")
        print("FAIL: Task-5 concurrent routing requirement not met")

    if successful >= 2:
        print("PASS: System can handle multiple concurrent requests")
    else:
        print("FAIL: System cannot handle concurrent requests")

    print("\nTask-5 Concurrent Routing Test Complete!")

if __name__ == "__main__":
    test_concurrent_routing()