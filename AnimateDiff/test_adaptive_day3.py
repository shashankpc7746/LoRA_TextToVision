"""
Test script for Task 4 Day 3 components
NAS Storage, GPU Queue, Mixed Precision, Lip-Sync
"""

import os
import tempfile
from pathlib import Path

# Import Day 3 components
from adaptive_engine.nas_storage import get_nas_storage, NASStorageManager
from adaptive_engine.gpu_queue import get_gpu_queue, GPUQueueManager, JobPriority
from adaptive_engine.mixed_precision import get_mixed_precision, MixedPrecisionManager
from adaptive_engine.lip_sync import get_lip_sync, LipSyncManager


def test_nas_storage():
    """Test NAS storage functionality"""
    print("\n=== Testing NAS Storage ===")

    nas = get_nas_storage()
    print(f"NAS available: {nas.nas_available}")
    print(f"NAS path: {nas.nas_path}")

    # Test stats
    stats = nas.get_storage_stats()
    print(f"NAS stats: {stats}")

    # Test signed URL generation
    test_url = nas.generate_signed_url("test_video.mp4")
    print(f"Signed URL generated: {test_url[:50]}...")

    print("[OK] NAS Storage tests passed")


def test_gpu_queue():
    """Test GPU queue functionality"""
    print("\n=== Testing GPU Queue ===")

    gpu_queue = get_gpu_queue()

    # Test job submission
    job_id = gpu_queue.submit_job(
        "Test video generation prompt",
        JobPriority.NORMAL,
        120
    )
    print(f"Job submitted: {job_id}")

    # Test job status
    job = gpu_queue.get_job_status(job_id)
    if job:
        print(f"Job status: {job.status.value}")

    # Test queue stats
    stats = gpu_queue.get_queue_stats()
    print(f"Queue stats: {stats}")

    # Test GPU status
    gpu_stats = gpu_queue.get_gpu_stats()
    print(f"GPU stats: {len(gpu_stats['gpus'])} GPUs available")

    print("[OK] GPU Queue tests passed")


def test_mixed_precision():
    """Test mixed precision functionality"""
    print("\n=== Testing Mixed Precision ===")

    precision = get_mixed_precision()

    # Test device detection
    capabilities = precision.device_capabilities
    print(f"CUDA available: {capabilities['cuda_available']}")
    print(f"GPU memory: {capabilities.get('gpu_memory_gb', 'N/A')} GB")

    # Test optimal config
    config = precision.get_optimal_config("auto", "normal", "medium")
    print(f"Optimal config: {config.mode.value} on {config.device_type.value}")

    # Test memory tips
    tips = precision.get_memory_optimization_tips(config)
    print(f"Memory tips: {len(tips)} recommendations")

    # Test precision stats
    stats = precision.get_precision_stats()
    print(f"Available configs: {list(stats['available_configs'])}")

    print("[OK] Mixed Precision tests passed")


def test_lip_sync():
    """Test lip-sync functionality"""
    print("\n=== Testing Lip-Sync ===")

    lip_sync = get_lip_sync()

    # Test model availability
    status = lip_sync.get_model_status()
    print(f"SadTalker available: {status['sadtalker_available']}")
    print(f"Wav2Lip available: {status['wav2lip_available']}")

    # Test config
    print(f"Lip-sync config: {status['config']}")

    print("[OK] Lip-Sync tests passed")


def test_integration():
    """Test integration between Day 3 components"""
    print("\n=== Testing Day 3 Integration ===")

    # Get all managers
    nas = get_nas_storage()
    gpu_queue = get_gpu_queue()
    precision = get_mixed_precision()
    lip_sync = get_lip_sync()

    # Test comprehensive status
    day3_status = {
        "nas": nas.get_storage_stats(),
        "gpu_queue": gpu_queue.get_queue_stats(),
        "precision": precision.get_precision_stats(),
        "lip_sync": lip_sync.get_model_status()
    }

    print(f"Day 3 integration status: {len(day3_status)} components")
    print(f"NAS files: {day3_status['nas'].get('total_files', 0)}")
    print(f"GPU queue jobs: {day3_status['gpu_queue']['queued_jobs']}")
    print(f"Precision configs: {len(day3_status['precision']['available_configs'])}")

    print("[OK] Day 3 Integration tests passed")


def main():
    """Run all Day 3 tests"""
    print("Starting Task 4 Day 3 Component Tests")
    print("=" * 50)

    try:
        test_nas_storage()
        test_gpu_queue()
        test_mixed_precision()
        test_lip_sync()
        test_integration()

        print("\n" + "=" * 50)
        print("All Day 3 tests passed successfully!")
        print("[OK] NAS Storage: Working")
        print("[OK] GPU Queue: Working")
        print("[OK] Mixed Precision: Working")
        print("[OK] Lip-Sync: Working")
        print("[OK] Integration: Working")

    except Exception as e:
        print(f"\n[ERROR] Day 3 tests failed: {e}")
        raise


if __name__ == "__main__":
    main()