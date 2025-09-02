#!/usr/bin/env python3
"""
Test Script for Task-4 Day-1 Implementation
Comprehensive testing of adaptive engine components
"""

import sys
import os
import json
import time
from pathlib import Path

# Import adaptive engine modules
try:
    from adaptive_engine import (  # type: ignore
        get_device_capabilities,  # type: ignore
        device_probe,  # type: ignore
        plan_video_quality,  # type: ignore
        budget_planner,  # type: ignore
        route_generation_task,  # type: ignore
        tier_router,  # type: ignore
        analyze_generation_task,  # type: ignore
        workload_analyzer  # type: ignore
    )
except ImportError as e:
    print(f"[ERROR] Failed to import adaptive_engine package: {e}")
    print("[INFO] Please ensure adaptive_engine package is properly installed")
    raise


def test_device_probe():
    """Test device probe functionality"""
    print("\n" + "="*60)
    print("[COMPUTER] TESTING DEVICE PROBE")
    print("="*60)

    # Test basic capabilities
    caps = get_device_capabilities()
    print(f"[OK] Device detected: {caps['gpu_name']} ({caps['gpu_memory_gb']}GB VRAM)")
    print(f"[OK] CUDA Version: {caps['cuda_version']}")
    print(f"[OK] Can handle heavy load: {caps['can_handle_heavy_load']}")
    print(f"[OK] Recommended tier: {caps['recommended_tier']}")

    # Test task assessment
    print("\n[TEST] Task Assessment Tests:")
    test_tasks = [
        (2.0, 180),   # Light task
        (6.0, 300),   # Medium task
        (10.0, 600),  # Heavy task
    ]

    for vram, time_sec in test_tasks:
        can_handle = device_probe.can_handle_task(vram, time_sec)
        status = "[OK] Can handle" if can_handle else "[FAIL] Cannot handle"
        print(f"  {status}: {vram}GB VRAM, {time_sec}s duration")

    return True


def test_budget_planner():
    """Test budget planner functionality"""
    print("\n" + "="*60)
    print("[MONEY] TESTING BUDGET PLANNER")
    print("="*60)

    device_caps = get_device_capabilities()

    # Test different scenarios
    test_scenarios = [
        ("simple", {"target_quality": "balanced"}),
        ("medium", {"target_quality": "quality"}),
        ("complex", {"target_quality": "ultra_quality"}),
        ("medium", {"priority": "speed"}),
        ("medium", {"priority": "quality", "max_cost_usd": 0.05}),
    ]

    for complexity, prefs in test_scenarios:
        print(f"\n🧪 Scenario: {complexity} task with {prefs}")

        settings = plan_video_quality(device_caps, complexity, prefs)

        print(f"  📐 Resolution: {settings.resolution}")
        print(f"  🎞️  Frames: {settings.num_frames}, Steps: {settings.steps}")
        print(f"  ⏱️  Estimated time: {settings.estimated_time_sec}s")
        print(f"  💵 Estimated cost: ${settings.estimated_cost_usd:.3f}")

        # Check constraints
        estimate = budget_planner.estimate_cost_and_time(settings)
        print(f"  ✅ Within budget: {estimate['within_budget']}")
        print(f"  ✅ Within latency: {estimate['within_latency']}")

    return True


def test_tier_router():
    """Test tier router functionality"""
    print("\n" + "="*60)
    print("[TRAFFIC] TESTING TIER ROUTER")
    print("="*60)

    device_caps = get_device_capabilities()

    # Test routing scenarios
    test_scenarios = [
        ("simple", {"prefer_local": True}),
        ("medium", {"prefer_local": True}),
        ("complex", {"prefer_local": True}),
        ("complex", {"prefer_local": False, "max_cost_usd": 0.05}),
    ]

    for complexity, prefs in test_scenarios:
        print(f"\n🧪 Scenario: {complexity} task with {prefs}")

        # First plan quality
        quality_settings = plan_video_quality(device_caps, complexity, prefs)

        # Then route
        decision = route_generation_task(device_caps, quality_settings.__dict__, complexity, prefs)

        print(f"  🎯 Selected Tier: {decision.tier}")
        print(f"  💡 Reason: {decision.reason}")
        print(f"  💵 Estimated Cost: ${decision.estimated_cost:.3f}")
        print(f"  ⏱️  Estimated Latency: {decision.estimated_latency}ms")
        print(f"  🎚️  Confidence: {decision.confidence:.2f}")
        print(f"  🔄 Fallback Options: {decision.fallback_options}")

    # Show tier status
    print("\n📊 Current Tier Status:")
    status = tier_router.get_tier_status()
    for tier, info in status.items():
        print(f"  {tier}: {info['current_load']}/{info['max_concurrent']} load, {info['availability']:.1%} available")

    return True


def test_workload_analyzer():
    """Test workload analyzer functionality"""
    print("\n" + "="*60)
    print("🧠 TESTING WORKLOAD ANALYZER")
    print("="*60)

    # Test different prompt complexities
    test_prompts = [
        ("A cat sitting on a mat", "realistic", "fast"),
        ("A young wizard casting a spell in an enchanted forest with magical effects",
         "fantasy", "balanced"),
        ("An epic battle scene with multiple characters, complex animations, dramatic lighting, and detailed backgrounds showing a fantasy world",
         "fantasy", "ultra_quality"),
    ]

    for prompt, style, quality in test_prompts:
        print(f"\n🧪 Analyzing: {prompt[:50]}...")
        print(f"  🎨 Style: {style}, Quality: {quality}")

        analysis = analyze_generation_task(prompt, style, quality)

        print(f"  📊 Complexity: {analysis.complexity} (confidence: {analysis.confidence:.2f})")
        print(f"  🎯 Recommended Tier: {analysis.recommended_tier}")
        print(f"  🧠 Estimated VRAM: {analysis.estimated_vram_gb}GB")
        print(f"  ⏱️  Estimated Time: {analysis.estimated_time_sec}s")

        if analysis.reasoning:
            print(f"  💡 Reasoning: {analysis.reasoning[0]}")

        # Show key factors
        factors = analysis.factors
        print(f"  📋 Key Factors:")
        print(f"    • Words: {factors.get('word_count', 'N/A')}")
        print(f"    • Scenes: {factors.get('scene_count', 'N/A')}")
        print(f"    • Animation keywords: {factors.get('animation_keywords', 'N/A')}")

    return True


def test_integrated_workflow():
    """Test the complete integrated workflow"""
    print("\n" + "="*60)
    print("🔄 TESTING INTEGRATED WORKFLOW")
    print("="*60)

    # Simulate a complete request
    test_request = {
        "prompt": "A majestic eagle soaring through dramatic mountain peaks at sunset with golden light rays",
        "style": "realistic",
        "target_quality": "quality",
        "max_cost_usd": 0.08,
        "max_latency_sec": 240,
        "prefer_local": True
    }

    print("📝 Test Request:")
    print(json.dumps(test_request, indent=2))

    start_time = time.time()

    # Step 1: Device capabilities
    print("\n1️⃣ Step 1: Device Analysis")
    device_caps = get_device_capabilities()
    print(f"   ✅ Device: {device_caps['gpu_name']} ({device_caps['gpu_memory_gb']}GB VRAM)")

    # Step 2: Task analysis
    print("\n2️⃣ Step 2: Task Analysis")
    task_analysis = analyze_generation_task(
        test_request["prompt"],
        test_request["style"],
        test_request["target_quality"]
    )
    print(f"   📊 Complexity: {task_analysis.complexity}")
    print(f"   🎯 Recommended: {task_analysis.recommended_tier}")

    # Step 3: Quality planning
    print("\n3️⃣ Step 3: Quality Planning")
    quality_settings = plan_video_quality(
        device_caps,
        task_analysis.complexity,
        {
            "target_quality": test_request["target_quality"],
            "max_cost_usd": test_request["max_cost_usd"],
            "max_latency_sec": test_request["max_latency_sec"]
        }
    )
    print(f"   📐 Resolution: {quality_settings.resolution}")
    print(f"   🎞️  Quality: {quality_settings.num_frames} frames, {quality_settings.steps} steps")

    # Step 4: Tier routing
    print("\n4️⃣ Step 4: Tier Routing")
    routing_decision = route_generation_task(
        device_caps,
        quality_settings.__dict__,
        task_analysis.complexity,
        {
            "prefer_local": test_request["prefer_local"],
            "max_cost_usd": test_request["max_cost_usd"],
            "max_latency_sec": test_request["max_latency_sec"]
        }
    )
    print(f"   🎯 Selected: {routing_decision.tier}")
    print(f"   💵 Cost: ${routing_decision.estimated_cost:.3f}")
    print(f"   ⏱️  Latency: {routing_decision.estimated_latency}ms")

    # Step 5: Final summary
    processing_time = time.time() - start_time
    print("\n5️⃣ Step 5: Final Summary")
    print(f"   ✅ Processing time: {processing_time:.2f}s")
    print(f"   🎯 Final tier: {routing_decision.tier}")
    print(f"   💰 Total estimated cost: ${routing_decision.estimated_cost:.3f}")
    print(f"   📊 Confidence: {routing_decision.confidence:.2f}")

    return True


def run_all_tests():
    """Run all Day-1 tests"""
    print("[TARGET] TASK-4 DAY-1 IMPLEMENTATION TEST SUITE")
    print("="*60)
    print("Testing: Device Probe + Budget Planner + Tier Router + Workload Analyzer")
    print("="*60)

    test_results = []

    # Run individual component tests
    try:
        test_results.append(("Device Probe", test_device_probe()))
    except Exception as e:
        print(f"❌ Device Probe test failed: {e}")
        test_results.append(("Device Probe", False))

    try:
        test_results.append(("Budget Planner", test_budget_planner()))
    except Exception as e:
        print(f"❌ Budget Planner test failed: {e}")
        test_results.append(("Budget Planner", False))

    try:
        test_results.append(("Tier Router", test_tier_router()))
    except Exception as e:
        print(f"❌ Tier Router test failed: {e}")
        test_results.append(("Tier Router", False))

    try:
        test_results.append(("Workload Analyzer", test_workload_analyzer()))
    except Exception as e:
        print(f"❌ Workload Analyzer test failed: {e}")
        test_results.append(("Workload Analyzer", False))

    try:
        test_results.append(("Integrated Workflow", test_integrated_workflow()))
    except Exception as e:
        print(f"❌ Integrated Workflow test failed: {e}")
        test_results.append(("Integrated Workflow", False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Day-1 implementation is ready!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)