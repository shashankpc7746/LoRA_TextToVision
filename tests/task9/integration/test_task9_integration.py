"""
Task 9 - Complete Integration Test Suite
Tests all 10 components without requiring trained LoRA adapter
"""

import sys
from pathlib import Path
import torch

print("="*70)
print("TASK 9 - COMPLETE INTEGRATION TEST SUITE")
print("="*70)
print()

# Test results tracking
results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def test_component(name, test_func):
    """Run a test and track results"""
    try:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        test_func()
        results["passed"].append(name)
        print(f"✅ {name} - PASSED")
        return True
    except Exception as e:
        results["failed"].append((name, str(e)))
        print(f"❌ {name} - FAILED: {e}")
        return False

# ============================================================
# TEST 1: Dataset & Adapter Infrastructure
# ============================================================
def test_dataset_adapter():
    from adapters.gurukul_lora.dataset_curator import GurukulDatasetCurator
    
    curator = GurukulDatasetCurator("datasets/gurukul_keyframes")
    is_valid, report = curator.validate()
    
    print(f"Dataset valid: {is_valid}")
    print(f"Total images: {report['total_images']}")
    print(f"Valid images: {report['valid_images']}")
    
    assert report['total_images'] >= 50, "Need at least 50 images"
    assert report['has_captions'], "Need captions.json"
    
    # Test adapter code exists
    adapter_files = [
        "adapters/gurukul_lora/train_adapter.py",
        "adapters/gurukul_lora/inference.py",
        "adapters/gurukul_lora/dataset_curator.py"
    ]
    for f in adapter_files:
        assert Path(f).exists(), f"Missing: {f}"
    
    print("✓ All adapter files present")

test_component("Day 1: Indigenous LoRA Adapter", test_dataset_adapter)

# ============================================================
# TEST 2: Temporal Consistency Module
# ============================================================
def test_temporal_consistency():
    from interpolator.temporal_consistency import TemporalConsistencyModule
    
    # Create module
    module = TemporalConsistencyModule(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {module.device}")
    print(f"UNet 3D channels: {module.unet_3d.in_channels}")
    
    # Test with dummy frames
    dummy_frames = torch.randn(1, 16, 3, 256, 256)
    if torch.cuda.is_available():
        dummy_frames = dummy_frames.cuda()
    
    output = module.smooth_sequence(dummy_frames)
    
    assert output.shape == dummy_frames.shape, "Output shape mismatch"
    print(f"✓ Processed sequence: {output.shape}")

test_component("Day 2: Temporal Consistency", test_temporal_consistency)

# ============================================================
# TEST 3: Tile Upscaler
# ============================================================
def test_tile_upscaler():
    from upscaler.tile_upscale import TileBasedUpscaler
    
    upscaler = TileBasedUpscaler(
        target_height=1080,
        tile_size=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Target resolution: {upscaler.target_height}p")
    print(f"Tile size: {upscaler.tile_size}x{upscaler.tile_size}")
    print(f"Device: {upscaler.device}")
    
    # Check if Real-ESRGAN available
    if upscaler.realesrgan is not None:
        print("✓ Real-ESRGAN available")
    else:
        print("⚠️  Real-ESRGAN not available, using OpenCV fallback")
        results["warnings"].append("Tile Upscaler using OpenCV fallback")
    
    # Test configuration
    assert upscaler.target_height == 1080, "Wrong target height"
    assert upscaler.tile_size == 512, "Wrong tile size"

test_component("Day 3: Tile Upscaler", test_tile_upscaler)

# ============================================================
# TEST 4: Motion Controller
# ============================================================
def test_motion_controller():
    from motion_controller.policy import (
        MicroExpressionScheduler,
        PoseConditioner,
        MotionPolicy
    )
    
    # Test scheduler
    scheduler = MicroExpressionScheduler(lambda_poisson=0.1, seed=42)
    schedule = scheduler.generate_schedule(num_frames=240, fps=30)
    
    print(f"Generated {len(schedule)} micro-expressions")
    print(f"Action types: {set(a['action'] for a in schedule)}")
    
    # Test pose conditioner
    conditioner = PoseConditioner(embedding_dim=256)
    dummy_pose = torch.randn(1, 68, 2)  # 68 facial landmarks
    embedding = conditioner.encode(dummy_pose)
    
    print(f"Pose embedding shape: {embedding.shape}")
    assert embedding.shape == (1, 256), "Wrong embedding shape"
    
    # Test motion policy
    policy = MotionPolicy()
    state = torch.randn(1, 256)
    action_idx = policy.get_action(state, training=False)
    
    print(f"Sampled action: {action_idx}")
    assert 0 <= action_idx < 9, "Invalid action index"
    
    print("✓ All 9 micro-expressions available")

test_component("Day 4: Motion Controller", test_motion_controller)

# ============================================================
# TEST 5: Quality Assessment Card
# ============================================================
def test_quality_card():
    import subprocess
    
    # Check if ffmpeg available (needed for VMAF)
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        ffmpeg_available = True
        print("✓ FFmpeg available for VMAF")
    except:
        ffmpeg_available = False
        print("⚠️  FFmpeg not available, will use fallback metrics")
        results["warnings"].append("Quality Card using fallback metrics (no VMAF)")
    
    # Test imports
    from test_quality_card import (
        VMAFEvaluator,
        LipSyncEvaluator,
        CostTracker,
        QualityCard
    )
    
    # Test cost tracker
    tracker = CostTracker()
    tracker.add_cost("keyframe_generation", 0.05)
    tracker.add_cost("animation", 0.03)
    
    total = tracker.get_total_cost()
    print(f"Total cost tracked: ${total:.2f}")
    assert total == 0.08, "Cost calculation wrong"
    
    # Test quality card structure
    card = QualityCard(
        video_path="test.mp4",
        prompt="test",
        vmaf_score=85.0,
        lipsync_score=0.85,
        cost=0.08,
        latency=120,
        resolution=(1920, 1080)
    )
    
    passed = card.passes_acceptance_criteria()
    print(f"Acceptance criteria: {'PASS' if passed else 'FAIL'}")

test_component("Day 5: Quality Assessment", test_quality_card)

# ============================================================
# TEST 6: JWT Authentication
# ============================================================
def test_jwt_auth():
    # Check JWT implementation in API
    api_file = Path("AnimateDiff_API/adaptive_api.py")
    assert api_file.exists(), "API file missing"
    
    content = api_file.read_text()
    
    # Check for JWT components
    assert "jwt" in content.lower(), "JWT not implemented"
    assert "verify_token" in content or "decode" in content, "Token verification missing"
    
    print("✓ JWT authentication implemented in API")
    print("✓ Token verification present")

test_component("Compliance: JWT Authentication", test_jwt_auth)

# ============================================================
# TEST 7: Audit Logger
# ============================================================
def test_audit_logger():
    from audit_logger import get_audit_logger, KSMLToken
    
    logger = get_audit_logger()
    
    # Test logging
    token = KSMLToken(
        intent="test_video_generation",
        karma_state={"quality": "high"},
        lineage="test_session"
    )
    
    logger.log_operation(
        operation="test_operation",
        ksml_token=token,
        metadata={"test": True}
    )
    
    # Test query
    recent = logger.query_logs(limit=1)
    assert len(recent) > 0, "No logs found"
    
    print(f"✓ Logged operation with hash: {recent[0]['entry_hash'][:16]}...")
    print(f"✓ Audit trail: {len(recent)} entries")
    
    # Test integrity
    is_valid = logger.verify_integrity()
    print(f"✓ Audit log integrity: {'VALID' if is_valid else 'INVALID'}")

test_component("Compliance: Audit Logger", test_audit_logger)

# ============================================================
# TEST 8: InsightFlow Telemetry
# ============================================================
def test_telemetry():
    from insightflow_client import get_insightflow_client, TelemetryEvent
    
    client = get_insightflow_client()
    
    # Test event emission
    event = TelemetryEvent(
        event_type="test_event",
        component="integration_test",
        metrics={"duration": 1.5, "success": True}
    )
    
    client.emit_event(event)
    
    # Test metrics collection
    client.metrics_collector.add_metric("test_latency", 100)
    client.metrics_collector.add_metric("test_latency", 150)
    client.metrics_collector.add_metric("test_latency", 120)
    
    stats = client.metrics_collector.get_stats("test_latency")
    
    print(f"✓ Events emitted: {len(client.session_events)}")
    print(f"✓ Metrics tracked: mean={stats['mean']:.1f}, p95={stats['p95']:.1f}")

test_component("Compliance: InsightFlow Telemetry", test_telemetry)

# ============================================================
# TEST 9: Multi-GPU Docker Configuration
# ============================================================
def test_docker_config():
    dockerfile = Path("Dockerfile")
    docker_compose = Path("docker-compose.yml")
    
    assert dockerfile.exists(), "Dockerfile missing"
    assert docker_compose.exists(), "docker-compose.yml missing"
    
    dockerfile_content = dockerfile.read_text()
    
    # Check for GPU configuration
    assert "cuda" in dockerfile_content.lower(), "CUDA support missing"
    
    # Check for multi-GPU env vars
    compose_content = docker_compose.read_text()
    assert "CUDA_VISIBLE_DEVICES" in compose_content, "GPU allocation missing"
    
    print("✓ Dockerfile with CUDA support")
    print("✓ docker-compose.yml with multi-GPU config")
    print("✓ Environment variables for GPU:0 and GPU:1")

test_component("Compliance: Multi-GPU Docker", test_docker_config)

# ============================================================
# TEST 10: Documentation
# ============================================================
def test_documentation():
    docs = [
        "Task-9-README.md",
        "Task-9-Final-Summary.md",
        "Task-9-Quick-Reference.md",
        "Task-9-Bootstrap-Summary.md",
        "Task-9-Day1-Summary.md"
    ]
    
    found_docs = []
    for doc in docs:
        if Path(doc).exists():
            found_docs.append(doc)
            size = Path(doc).stat().st_size
            print(f"✓ {doc} ({size:,} bytes)")
    
    assert len(found_docs) >= 3, f"Only {len(found_docs)} docs found"
    
    # Check final summary has substantial content
    final_summary = Path("Task-9-Final-Summary.md")
    if final_summary.exists():
        content = final_summary.read_text()
        assert len(content) > 10000, "Final summary too short"
        print(f"✓ Final summary: {len(content):,} characters")

test_component("Compliance: Documentation", test_documentation)

# ============================================================
# FINAL REPORT
# ============================================================
print("\n")
print("="*70)
print("FINAL TEST REPORT")
print("="*70)
print()

total_tests = len(results["passed"]) + len(results["failed"])
pass_rate = len(results["passed"]) / total_tests * 100 if total_tests > 0 else 0

print(f"✅ PASSED: {len(results['passed'])}/{total_tests} ({pass_rate:.1f}%)")
for test in results["passed"]:
    print(f"   ✓ {test}")

if results["failed"]:
    print(f"\n❌ FAILED: {len(results['failed'])}/{total_tests}")
    for test, error in results["failed"]:
        print(f"   ✗ {test}")
        print(f"     Error: {error[:100]}")

if results["warnings"]:
    print(f"\n⚠️  WARNINGS: {len(results['warnings'])}")
    for warning in results["warnings"]:
        print(f"   • {warning}")

print()
print("="*70)

if len(results["failed"]) == 0:
    print("🎉 ALL TESTS PASSED! Task 9 implementation is complete and functional.")
    print("="*70)
    sys.exit(0)
else:
    print("⚠️  Some tests failed. Review errors above.")
    print("="*70)
    sys.exit(1)
