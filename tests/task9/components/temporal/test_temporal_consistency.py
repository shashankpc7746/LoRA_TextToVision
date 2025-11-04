"""
Test Suite for Temporal Consistency Module
==========================================

Tests:
    1. Temporal UNet architecture
    2. Histogram matching
    3. Optical flow computation
    4. End-to-end processing
    5. KSML compliance
"""

import unittest
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from temporal_consistency import (
    TemporalUNet3D,
    HistogramMatcher,
    OpticalFlowGuide,
    TemporalConsistencyProcessor,
    process_frames_consistent
)


class TestTemporalConsistency(unittest.TestCase):
    """Test suite for temporal consistency module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        
        # Create test frames
        cls.test_frames_dir = Path(cls.temp_dir) / "test_frames"
        cls.test_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic test frames (10 frames, 256x256)
        for i in range(10):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(cls.test_frames_dir / f"frame_{i:04d}.png"), frame)
        
        print(f"\n✅ Test setup complete. Device: {cls.device}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        print("\n✅ Test cleanup complete")
    
    def test_1_temporal_unet_architecture(self):
        """Test Temporal UNet 3D architecture."""
        print("\n🧪 Test 1: Temporal UNet Architecture")
        
        model = TemporalUNet3D(in_channels=3, out_channels=3, base_channels=32)
        
        # Test forward pass
        batch_size = 1
        channels = 3
        frames = 8
        height = 256
        width = 256
        
        x = torch.randn(batch_size, channels, frames, height, width)
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check residual connection
        self.assertTrue(torch.allclose(output - x, model(x) - x, atol=1e-5))
        
        print(f"   ✅ Model architecture valid")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
    
    def test_2_histogram_matching(self):
        """Test histogram matching for de-flicker."""
        print("\n🧪 Test 2: Histogram Matching")
        
        # Create test images with different brightness
        source = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        reference = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        
        # Match histograms
        matched = HistogramMatcher.match_histograms(source, reference)
        
        # Check output shape
        self.assertEqual(matched.shape, source.shape)
        
        # Check that matched histogram is closer to reference
        source_mean = np.mean(source)
        reference_mean = np.mean(reference)
        matched_mean = np.mean(matched)
        
        self.assertLess(
            abs(matched_mean - reference_mean),
            abs(source_mean - reference_mean)
        )
        
        print(f"   ✅ Histogram matching works")
        print(f"   Source mean: {source_mean:.2f}")
        print(f"   Reference mean: {reference_mean:.2f}")
        print(f"   Matched mean: {matched_mean:.2f}")
    
    def test_3_temporal_smooth_histograms(self):
        """Test temporal histogram smoothing."""
        print("\n🧪 Test 3: Temporal Histogram Smoothing")
        
        # Create frames with flicker
        frames = []
        for i in range(10):
            brightness = 100 + 50 * (i % 2)  # Alternating brightness
            frame = np.full((256, 256, 3), brightness, dtype=np.uint8)
            frames.append(frame)
        
        # Smooth histograms
        smoothed = HistogramMatcher.temporal_smooth_histograms(frames, alpha=0.5)
        
        # Check that variance decreased
        original_variance = np.var([np.mean(f) for f in frames])
        smoothed_variance = np.var([np.mean(f) for f in smoothed])
        
        self.assertLess(smoothed_variance, original_variance)
        
        print(f"   ✅ Temporal smoothing reduces flicker")
        print(f"   Original variance: {original_variance:.2f}")
        print(f"   Smoothed variance: {smoothed_variance:.2f}")
    
    def test_4_optical_flow_computation(self):
        """Test optical flow computation."""
        print("\n🧪 Test 4: Optical Flow Computation")
        
        # Create two frames with simple motion
        frame1 = np.zeros((256, 256, 3), dtype=np.uint8)
        frame2 = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Draw square in different positions
        frame1[100:150, 100:150] = 255
        frame2[100:150, 110:160] = 255  # Shifted right by 10 pixels
        
        # Compute flow
        flow = OpticalFlowGuide.compute_flow(frame1, frame2)
        
        # Check flow shape
        self.assertEqual(flow.shape, (256, 256, 2))
        
        # Check that flow indicates rightward motion in the square region
        flow_x = flow[120, 120, 0]  # Center of square
        self.assertGreater(flow_x, 0)  # Should be positive (rightward)
        
        print(f"   ✅ Optical flow computed correctly")
        print(f"   Flow shape: {flow.shape}")
        print(f"   Average horizontal flow: {np.mean(flow[:, :, 0]):.2f}")
    
    def test_5_flow_guided_blend(self):
        """Test flow-guided blending."""
        print("\n🧪 Test 5: Flow-Guided Blending")
        
        # Create frames
        frames = []
        for i in range(5):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Apply flow-guided blending
        blended = OpticalFlowGuide.flow_guided_blend(frames, alpha=0.3)
        
        # Check output
        self.assertEqual(len(blended), len(frames))
        self.assertEqual(blended[0].shape, frames[0].shape)
        
        print(f"   ✅ Flow-guided blending works")
        print(f"   Input frames: {len(frames)}")
        print(f"   Output frames: {len(blended)}")
    
    def test_6_temporal_consistency_processor_init(self):
        """Test TemporalConsistencyProcessor initialization."""
        print("\n🧪 Test 6: Processor Initialization")
        
        processor = TemporalConsistencyProcessor(
            device=self.device,
            use_histogram=True,
            use_flow=True
        )
        
        # Check device
        self.assertEqual(str(processor.device), self.device)
        
        # Check model
        self.assertIsNotNone(processor.model)
        
        print(f"   ✅ Processor initialized")
        print(f"   Device: {processor.device}")
        print(f"   Histogram: {processor.use_histogram}")
        print(f"   Flow: {processor.use_flow}")
    
    def test_7_end_to_end_processing(self):
        """Test end-to-end temporal consistency processing."""
        print("\n🧪 Test 7: End-to-End Processing")
        
        # Set up directories
        in_dir = self.test_frames_dir
        out_dir = Path(self.temp_dir) / "output_frames"
        
        # Process frames
        processor = TemporalConsistencyProcessor(
            device=self.device,
            use_histogram=True,
            use_flow=True
        )
        
        metadata = processor.process_frames_consistent(str(in_dir), str(out_dir))
        
        # Check output directory
        self.assertTrue(out_dir.exists())
        
        # Check frames were created
        output_frames = list(out_dir.glob("*.png"))
        self.assertGreater(len(output_frames), 0)
        
        # Check metadata
        self.assertIn("operation", metadata)
        self.assertEqual(metadata["operation"], "temporal_consistency")
        self.assertIn("ksml_lineage", metadata)
        
        # Check metadata file
        metadata_file = out_dir / "temporal_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        print(f"   ✅ End-to-end processing successful")
        print(f"   Input frames: {len(list(in_dir.glob('*.png')))}")
        print(f"   Output frames: {len(output_frames)}")
        print(f"   Processing time: {metadata['processing_time_seconds']:.2f}s")
    
    def test_8_ksml_compliance(self):
        """Test KSML compliance and metadata."""
        print("\n🧪 Test 8: KSML Compliance")
        
        # Create KSML token
        ksml_token = {
            "ksml_token": "test_parent_token_123",
            "intent": "test_temporal_consistency",
            "karma_state": "generated"
        }
        
        # Process with KSML token
        in_dir = self.test_frames_dir
        out_dir = Path(self.temp_dir) / "ksml_output"
        
        processor = TemporalConsistencyProcessor(device=self.device)
        metadata = processor.process_frames_consistent(
            str(in_dir),
            str(out_dir),
            ksml_token=ksml_token
        )
        
        # Check KSML lineage
        self.assertIn("ksml_lineage", metadata)
        lineage = metadata["ksml_lineage"]
        
        self.assertEqual(lineage["parent_token"], "test_parent_token_123")
        self.assertEqual(lineage["operation"], "temporal_consistency")
        self.assertEqual(lineage["karma_state"], "temporal_smoothed")
        self.assertIn("lineage", lineage)
        
        print(f"   ✅ KSML compliance verified")
        print(f"   Parent token: {lineage['parent_token']}")
        print(f"   Operation: {lineage['operation']}")
        print(f"   Karma state: {lineage['karma_state']}")
    
    def test_9_convenience_function(self):
        """Test convenience function."""
        print("\n🧪 Test 9: Convenience Function")
        
        in_dir = self.test_frames_dir
        out_dir = Path(self.temp_dir) / "convenience_output"
        
        # Use convenience function
        metadata = process_frames_consistent(
            str(in_dir),
            str(out_dir),
            device=self.device
        )
        
        # Check output
        self.assertTrue(out_dir.exists())
        self.assertGreater(len(list(out_dir.glob("*.png"))), 0)
        
        print(f"   ✅ Convenience function works")
        print(f"   Frames processed: {metadata['num_frames']}")


def run_tests(verbose=True):
    """Run all tests with detailed output."""
    print("=" * 60)
    print("Temporal Consistency Module - Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTemporalConsistency)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Temporal Consistency Module")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    result = run_tests(verbose=args.verbose)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
