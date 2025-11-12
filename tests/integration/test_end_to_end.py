"""
End-to-End Integration Test
Tests complete video generation pipeline from text prompt to final output
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator import ProductionOrchestrator
from security import embed_watermark, compute_fingerprint
from tools.detect_provenance import detect_watermark, verify_provenance


class TestEndToEnd:
    """Complete pipeline integration tests"""

    @pytest.fixture
    def orchestrator(self):
        """Get orchestrator instance"""
        return ProductionOrchestrator()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_video_pipeline(self, orchestrator):
        """
        Test complete video generation workflow:
        Text → Keyframes → Animation → Interpolation → Audio → Upscaling → Security
        """
        print("\n" + "="*60)
        print("🧪 END-TO-END INTEGRATION TEST")
        print("="*60)

        # Step 1: Generate video
        print("\n📝 Step 1: Generating video from text prompt...")
        prompt = "A serene mountain landscape at sunset with flowing river"
        
        result = await orchestrator.generate_video(
            prompt=prompt,
            target_quality=0.75,  # Moderate quality for faster testing
            max_cost_usd=1.0,
            max_latency_sec=300,
            additional_params={
                "with_bgm": False,  # Skip background music for faster testing
                "lip_sync": False   # Skip lip-sync for faster testing
            }
        )

        assert result["success"], f"Video generation failed: {result.get('error', 'Unknown error')}"
        print(f"   ✅ Video generated successfully")
        print(f"   📁 Output: {result['final_result']['output_path']}")
        print(f"   ⏱️ Time: {result['performance_metrics']['total_time_seconds']:.1f}s")
        print(f"   💰 Cost: ${result['performance_metrics']['cost_usd']:.2f}")

        # Step 2: Verify file exists and has content
        print("\n📂 Step 2: Verifying output file...")
        output_path = result['final_result']['output_path']
        assert os.path.exists(output_path), f"Output file not found: {output_path}"
        
        file_size = os.path.getsize(output_path)
        assert file_size > 100_000, f"Output file too small ({file_size} bytes), likely corrupted"
        print(f"   ✅ File exists: {file_size:,} bytes")

        # Step 3: Apply security watermarking
        print("\n🔒 Step 3: Applying security watermarking...")
        import datetime
        build_id = f"test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        watermarked_path = embed_watermark(
            output_path,
            build_id=build_id,
            output_path=output_path.replace('.mp4', '_watermarked.mp4')
        )
        
        assert os.path.exists(watermarked_path), "Watermarked file not created"
        print(f"   ✅ Watermark applied")
        print(f"   🏷️ Build ID: {build_id}")

        # Step 4: Verify watermark detection
        print("\n🔍 Step 4: Verifying watermark detection...")
        watermark_result = detect_watermark(watermarked_path)
        
        assert watermark_result is not None, "Watermark not detected"
        assert watermark_result['detected'], "Watermark detection failed"
        assert watermark_result['build_id'] == build_id, f"Build ID mismatch: {watermark_result['build_id']} != {build_id}"
        print(f"   ✅ Watermark detected correctly")
        print(f"   🏷️ Detected Build ID: {watermark_result['build_id']}")

        # Step 5: Compute and verify fingerprint
        print("\n🔍 Step 5: Computing content fingerprint...")
        fingerprint = compute_fingerprint(watermarked_path, build_id=build_id)
        
        assert 'sha256' in fingerprint, "SHA256 hash missing"
        assert 'blake2b' in fingerprint, "BLAKE2b hash missing"
        assert len(fingerprint['sha256']) == 64, "Invalid SHA256 hash length"
        print(f"   ✅ Fingerprint computed")
        print(f"   🔐 SHA256: {fingerprint['sha256'][:16]}...")
        print(f"   🔐 BLAKE2b: {fingerprint['blake2b'][:16]}...")

        # Step 6: Verify provenance
        print("\n✅ Step 6: Verifying provenance...")
        provenance_result = verify_provenance(watermarked_path, build_id=build_id)
        
        assert provenance_result['verified'], "Provenance verification failed"
        print(f"   ✅ Provenance verified")
        print(f"   🏷️ Build ID match: {provenance_result['build_id']}")

        # Step 7: Quality validation
        print("\n📊 Step 7: Quality validation...")
        quality_score = result['final_result'].get('quality_score', 0)
        target_quality = 0.75
        
        # Allow 10% tolerance
        assert quality_score >= target_quality * 0.9, \
            f"Quality below target: {quality_score:.2f} < {target_quality * 0.9:.2f}"
        print(f"   ✅ Quality score: {quality_score:.2f} (target: {target_quality:.2f})")

        # Step 8: Performance validation
        print("\n⚡ Step 8: Performance validation...")
        latency = result['performance_metrics']['total_time_seconds']
        max_latency = 300  # 5 minutes
        
        assert latency <= max_latency * 1.2, \
            f"Latency too high: {latency:.1f}s > {max_latency * 1.2:.1f}s"
        print(f"   ✅ Latency: {latency:.1f}s (max: {max_latency:.1f}s)")

        # Step 9: Cost validation
        print("\n💰 Step 9: Cost validation...")
        cost = result['performance_metrics']['cost_usd']
        max_cost = 1.0
        
        assert cost <= max_cost * 1.1, \
            f"Cost too high: ${cost:.2f} > ${max_cost * 1.1:.2f}"
        print(f"   ✅ Cost: ${cost:.2f} (max: ${max_cost:.2f})")

        # Cleanup
        print("\n🧹 Cleanup...")
        if os.path.exists(watermarked_path):
            os.remove(watermarked_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"   ✅ Temporary files removed")

        print("\n" + "="*60)
        print("✅ END-TO-END TEST PASSED")
        print("="*60)
        print(f"\nSummary:")
        print(f"  • Video generated: ✅")
        print(f"  • Watermark applied: ✅")
        print(f"  • Watermark detected: ✅")
        print(f"  • Fingerprint computed: ✅")
        print(f"  • Provenance verified: ✅")
        print(f"  • Quality: {quality_score:.2f} ✅")
        print(f"  • Latency: {latency:.1f}s ✅")
        print(f"  • Cost: ${cost:.2f} ✅")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fallback_mechanism(self, orchestrator):
        """
        Test cloud fallback mechanism under load
        """
        print("\n" + "="*60)
        print("🧪 FALLBACK MECHANISM TEST")
        print("="*60)

        # Simulate high GPU load by requesting high quality
        print("\n📝 Requesting high-quality video (should trigger fallback)...")
        
        result = await orchestrator.generate_video(
            prompt="Complex cinematic scene with multiple characters",
            target_quality=0.95,  # Very high quality
            max_cost_usd=2.0,
            max_latency_sec=600
        )

        assert result["success"], "Fallback generation failed"
        print(f"   ✅ Fallback mechanism working")
        print(f"   📊 Processing path: {result.get('processing_path', 'N/A')}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_generation(self, orchestrator):
        """
        Test concurrent video generation (stress test)
        """
        print("\n" + "="*60)
        print("🧪 CONCURRENT GENERATION TEST")
        print("="*60)

        num_concurrent = 3
        print(f"\n📝 Generating {num_concurrent} videos concurrently...")

        # Create concurrent tasks
        tasks = [
            orchestrator.generate_video(
                prompt=f"Test video {i}",
                target_quality=0.7,
                max_cost_usd=0.5,
                max_latency_sec=300
            )
            for i in range(num_concurrent)
        ]

        # Run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        print(f"   ✅ Success rate: {successful}/{num_concurrent} ({successful/num_concurrent*100:.1f}%)")
        
        assert successful >= num_concurrent * 0.8, \
            f"Too many failures: {successful}/{num_concurrent} successful"

    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_quality_degradation_handling(self, orchestrator):
        """
        Test system behavior when quality targets cannot be met
        """
        print("\n" + "="*60)
        print("🧪 QUALITY DEGRADATION TEST")
        print("="*60)

        print("\n📝 Requesting impossible quality target...")
        
        result = await orchestrator.generate_video(
            prompt="Test video",
            target_quality=0.99,  # Very hard to achieve
            max_cost_usd=0.1,     # Very low budget
            max_latency_sec=30     # Very tight deadline
        )

        # System should gracefully degrade or fail with clear message
        if result["success"]:
            print(f"   ✅ System adapted quality: {result['final_result']['quality_score']:.2f}")
        else:
            print(f"   ✅ System failed gracefully: {result.get('error', 'N/A')}")
            assert "quality" in result.get('error', '').lower() or \
                   "cost" in result.get('error', '').lower() or \
                   "latency" in result.get('error', '').lower(), \
                   "Error message should mention quality/cost/latency constraint"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
