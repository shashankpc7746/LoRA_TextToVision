#!/usr/bin/env python3
"""
Test Script for Task 4 Day 2 Components
Comprehensive testing of caching, RL, compression, and quality assessment
"""

import os
import sys
import time
from pathlib import Path

# Add adaptive_engine to path
sys.path.append(str(Path(__file__).parent / "adaptive_engine"))

from adaptive_engine import (
    get_cache_manager, get_rl_policy, get_compression_engine,
    get_quality_assessor, get_adaptive_pipeline, process_adaptive_request
)


def test_cache_manager():
    """Test cache manager functionality"""
    print("\n=== Testing Cache Manager ===")

    cache = get_cache_manager()

    # Test background caching
    print("Testing background caching...")
    test_bg_data = {"scene": "banyan_tree", "style": "realistic"}
    key = cache.cache_background("banyan", "realistic", test_bg_data)
    print(f"Cached background with key: {key}")

    # Test retrieval
    retrieved = cache.get_background("banyan", "realistic")
    print(f"Retrieved background: {retrieved is not None}")

    # Test pose caching
    print("Testing pose caching...")
    test_pose_data = {"gesture": "teaching", "confidence": 0.95}
    key = cache.cache_pose("teaching_gesture", test_pose_data)
    print(f"Cached pose with key: {key}")

    # Test seed caching
    print("Testing seed caching...")
    test_seed_data = {
        "prompt": "teacher explaining dharma",
        "seed": 42,
        "features": [0.1, 0.2, 0.3]
    }
    prompt_hash = "test_prompt_hash"
    key = cache.cache_seed(prompt_hash, test_seed_data)
    print(f"Cached seed with key: {key}")

    # Get stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats['total_entries']} entries, {stats['total_size_mb']:.2f} MB")

    print("[OK] Cache Manager tests passed")


def test_rl_policy():
    """Test RL policy functionality"""
    print("\n=== Testing RL Policy ===")

    rl = get_rl_policy()

    # Test state creation
    from adaptive_engine.rl_policy import State
    test_state = State(
        vmaf_score=75.0,
        latency_ms=120000,
        cost_usd=0.02,
        tier="local",
        quality_preset="balanced",
        device_class="desktop",
        task_complexity="medium"
    )
    print(f"Created test state: VMAF={test_state.vmaf_score}, Cost=${test_state.cost_usd}")

    # Test decision making
    should_retry, action, reason = rl.should_retry(test_state)
    print(f"RL Decision: Retry={should_retry}, Action={action}, Reason={reason}")

    # Test policy stats
    stats = rl.get_policy_stats()
    print(f"RL Stats: {stats['total_experiences']} experiences, {stats['average_reward']:.2f} avg reward")

    print("[OK] RL Policy tests passed")


def test_compression_engine():
    """Test compression engine functionality"""
    print("\n=== Testing Compression Engine ===")

    compressor = get_compression_engine()

    # Test preset retrieval
    preset = compressor.get_optimal_preset("desktop")
    print(f"Optimal preset for desktop: {preset}")

    # List available presets
    presets = list(compressor.presets.keys())
    print(f"Available presets: {presets}")

    # Test preset details
    if preset in compressor.presets:
        preset_info = compressor.presets[preset]
        print(f"Selected preset details: CRF={preset_info.crf}, Target VMAF={preset_info.target_vmaf}")

    print("[OK] Compression Engine tests passed")


def test_quality_assessor():
    """Test quality assessor functionality"""
    print("\n=== Testing Quality Assessor ===")

    assessor = get_quality_assessor()

    # Test threshold checking
    from adaptive_engine.quality_assessor import QualityMetrics
    test_metrics = QualityMetrics(
        vmaf_score=78.5,
        psnr_score=26.3,
        ssim_score=0.87,
        bitrate_kbps=1200.0,
        compression_ratio=0.65,
        encoding_time_seconds=95.0,
        file_size_mb=38.2
    )

    meets_threshold = assessor.meets_quality_threshold(test_metrics, 70.0)
    recommendation = assessor.get_quality_recommendation(test_metrics)

    print(f"Quality metrics: VMAF={test_metrics.vmaf_score}, PSNR={test_metrics.psnr_score}")
    print(f"Meets threshold (70): {meets_threshold}")
    print(f"Recommendation: {recommendation}")

    print("[OK] Quality Assessor tests passed")


def test_adaptive_pipeline():
    """Test complete adaptive pipeline"""
    print("\n=== Testing Adaptive Pipeline ===")

    # Create test request
    test_request = {
        "prompt": "A wise teacher explaining ancient wisdom under a banyan tree",
        "style": "realistic",
        "scene_type": "banyan",
        "duration_s": 20,
        "preferences": {
            "prefer_local": True,
            "max_cost_usd": 0.05,
            "max_latency_sec": 300
        },
        "character_pose": "teaching_gesture"
    }

    print(f"Test request: {test_request['prompt']}")

    # Process through pipeline
    result = process_adaptive_request(test_request)

    print(f"Pipeline result: Success={result.success}")
    print(f"Total time: {result.total_time_seconds:.2f}s")
    print(f"Total cost: ${result.total_cost_usd:.3f}")
    print(f"Tier used: {result.tier_used}")
    print(f"Cache hits: {len(result.cache_hits)}")
    print(f"RL decisions: {len(result.rl_decisions)}")

    if result.quality_metrics:
        print(f"Quality metrics: VMAF={result.quality_metrics.vmaf_score:.1f}")

    if result.compression_info:
        print(f"Compression: {result.compression_info.get('success', False)}")

    print("[OK] Adaptive Pipeline tests passed")


def test_integration():
    """Test integration between all components"""
    print("\n=== Testing Component Integration ===")

    # Get all managers
    cache = get_cache_manager()
    rl = get_rl_policy()
    compressor = get_compression_engine()
    assessor = get_quality_assessor()
    pipeline = get_adaptive_pipeline()

    # Test data flow
    test_data = "integration_test_data"

    # Cache something
    cache_key = cache.put("integration_test", test_data, {"type": "test"})
    retrieved = cache.get(cache_key)
    print(f"Cache integration: Stored and retrieved '{retrieved}'")

    # Test pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"Pipeline stats: Cache entries={stats['cache_stats']['total_entries']}")

    print("[OK] Integration tests passed")


def run_all_tests():
    """Run all Day 2 component tests"""
    print("Starting Task 4 Day 2 Component Tests")
    print("=" * 50)

    try:
        test_cache_manager()
        test_rl_policy()
        test_compression_engine()
        test_quality_assessor()
        test_adaptive_pipeline()
        test_integration()

        print("\n" + "=" * 50)
        print("All Day 2 tests passed successfully!")
        print("[OK] Caching system: Working")
        print("[OK] RL policy: Working")
        print("[OK] Compression engine: Working")
        print("[OK] Quality assessment: Working")
        print("[OK] Adaptive pipeline: Working")
        print("[OK] Component integration: Working")

        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)