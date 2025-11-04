"""
Test Suite for Tile Upscaler Module
====================================

Tests:
    1. Real-ESRGAN upscaler initialization
    2. Tile-based processing
    3. Temporal seam blending
    4. LUT color grading
    5. End-to-end upscaling
    6. KSML compliance
"""

import unittest
import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from upscaler.tile_upscale import (
    RealESRGANUpscaler,
    TemporalSeamBlender,
    LUTColorGrader,
    TileUpscaler,
    upscale_frames
)


class TestTileUpscaler(unittest.TestCase):
    """Test suite for tile upscaler module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.device = "cuda:0" if os.system("nvidia-smi > nul 2>&1") == 0 else "cpu"
        
        # Create test frames directory
        cls.test_frames_dir = Path(cls.temp_dir) / "test_frames"
        cls.test_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic test frames (5 frames, 256x256)
        for i in range(5):
            # Create frames with gradient pattern
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            frame[:, :, 0] = (np.arange(256).reshape(1, -1) * np.ones((256, 1))).astype(np.uint8)
            frame[:, :, 1] = (np.arange(256).reshape(-1, 1) * np.ones((1, 256))).astype(np.uint8)
            frame[:, :, 2] = 128 + i * 10
            
            cv2.imwrite(str(cls.test_frames_dir / f"frame_{i:04d}.png"), frame)
        
        print(f"\n✅ Test setup complete. Device: {cls.device}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        print("\n✅ Test cleanup complete")
    
    def test_1_realesrgan_upscaler_init(self):
        """Test Real-ESRGAN upscaler initialization."""
        print("\n🧪 Test 1: Real-ESRGAN Upscaler Initialization")
        
        upscaler = RealESRGANUpscaler(
            model_name="RealESRGAN_x4plus",
            device=self.device,
            tile_size=256
        )
        
        # Check initialization
        self.assertEqual(upscaler.scale, 4)
        self.assertEqual(upscaler.tile_size, 256)
        
        print(f"   ✅ Upscaler initialized")
        print(f"   Device: {upscaler.device}")
        print(f"   Scale: {upscaler.scale}x")
        print(f"   Tile size: {upscaler.tile_size}")
    
    def test_2_upscale_small_image(self):
        """Test upscaling a small image."""
        print("\n🧪 Test 2: Upscale Small Image")
        
        upscaler = RealESRGANUpscaler(device=self.device, tile_size=256)
        
        # Create test image
        input_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Upscale
        output_img = upscaler.upscale_image(input_img)
        
        # Check output shape (4x upscale)
        expected_shape = (128 * 4, 128 * 4, 3)
        self.assertEqual(output_img.shape, expected_shape)
        
        print(f"   ✅ Image upscaled successfully")
        print(f"   Input shape: {input_img.shape}")
        print(f"   Output shape: {output_img.shape}")
    
    def test_3_tile_based_upscaling(self):
        """Test tile-based upscaling for large images."""
        print("\n🧪 Test 3: Tile-based Upscaling")
        
        upscaler = RealESRGANUpscaler(device=self.device, tile_size=256, tile_pad=32)
        
        # Create larger test image
        input_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Upscale with tiles
        output_img = upscaler.upscale_with_tiles(input_img)
        
        # Check output shape
        expected_shape = (512 * 4, 512 * 4, 3)
        self.assertEqual(output_img.shape, expected_shape)
        
        print(f"   ✅ Tile-based upscaling works")
        print(f"   Input: {input_img.shape}")
        print(f"   Output: {output_img.shape}")
    
    def test_4_temporal_seam_blending(self):
        """Test temporal seam blending."""
        print("\n🧪 Test 4: Temporal Seam Blending")
        
        # Create frames with slight variations
        frames = []
        for i in range(10):
            frame = np.full((256, 256, 3), 100 + i * 5, dtype=np.uint8)
            frames.append(frame)
        
        # Apply temporal blending
        blended = TemporalSeamBlender.blend_frames_temporal(frames, blend_factor=0.2)
        
        # Check output
        self.assertEqual(len(blended), len(frames))
        
        # Check that blending smooths variations
        original_variance = np.var([np.mean(f) for f in frames])
        blended_variance = np.var([np.mean(f) for f in blended])
        
        self.assertLess(blended_variance, original_variance)
        
        print(f"   ✅ Temporal blending reduces variance")
        print(f"   Original variance: {original_variance:.2f}")
        print(f"   Blended variance: {blended_variance:.2f}")
    
    def test_5_scene_change_detection(self):
        """Test scene change detection."""
        print("\n🧪 Test 5: Scene Change Detection")
        
        # Create frames with a scene change
        frames = []
        
        # First scene (10 frames)
        for i in range(10):
            frame = np.full((256, 256, 3), 50, dtype=np.uint8)
            frames.append(frame)
        
        # Scene change
        frames.append(np.full((256, 256, 3), 200, dtype=np.uint8))
        
        # Second scene (5 frames)
        for i in range(5):
            frame = np.full((256, 256, 3), 200, dtype=np.uint8)
            frames.append(frame)
        
        # Detect scene changes
        scene_changes = TemporalSeamBlender.detect_scene_changes(frames, threshold=0.3)
        
        # Should detect the change at frame 10
        self.assertIn(0, scene_changes)  # First frame is always a scene change
        self.assertTrue(any(10 <= idx <= 11 for idx in scene_changes))
        
        print(f"   ✅ Scene change detection works")
        print(f"   Detected changes at frames: {scene_changes}")
    
    def test_6_lut_color_grading(self):
        """Test LUT color grading."""
        print("\n🧪 Test 6: LUT Color Grading")
        
        color_grader = LUTColorGrader(lut_path=None)  # Use default LUT
        
        # Create test image
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Apply LUT
        graded_img = color_grader.apply_lut(test_img)
        
        # Check output shape
        self.assertEqual(graded_img.shape, test_img.shape)
        
        # Check that values are modified (not identical)
        self.assertFalse(np.array_equal(test_img, graded_img))
        
        print(f"   ✅ LUT color grading applied")
        print(f"   Input mean: {np.mean(test_img):.2f}")
        print(f"   Output mean: {np.mean(graded_img):.2f}")
    
    def test_7_tile_upscaler_init(self):
        """Test TileUpscaler initialization."""
        print("\n🧪 Test 7: TileUpscaler Initialization")
        
        upscaler = TileUpscaler(
            device=self.device,
            tile_size=256,
            use_temporal_blend=True,
            use_color_grade=True
        )
        
        # Check initialization
        self.assertIsNotNone(upscaler.upscaler)
        self.assertTrue(upscaler.use_temporal_blend)
        self.assertTrue(upscaler.use_color_grade)
        self.assertIsNotNone(upscaler.color_grader)
        
        print(f"   ✅ TileUpscaler initialized")
        print(f"   Device: {upscaler.device}")
        print(f"   Temporal blend: {upscaler.use_temporal_blend}")
        print(f"   Color grade: {upscaler.use_color_grade}")
    
    def test_8_end_to_end_upscaling(self):
        """Test end-to-end upscaling pipeline."""
        print("\n🧪 Test 8: End-to-End Upscaling")
        
        # Setup directories
        in_dir = self.test_frames_dir
        out_dir = Path(self.temp_dir) / "upscaled_frames"
        
        # Create upscaler
        upscaler = TileUpscaler(
            device=self.device,
            tile_size=256,
            use_temporal_blend=True,
            use_color_grade=True
        )
        
        # Upscale frames
        metadata = upscaler.upscale_frames(
            str(in_dir),
            str(out_dir),
            target_height=512  # Smaller target for faster testing
        )
        
        # Check output directory exists
        self.assertTrue(out_dir.exists())
        
        # Check frames were created
        output_frames = list(out_dir.glob("*.png"))
        self.assertGreater(len(output_frames), 0)
        
        # Check metadata
        self.assertIn("operation", metadata)
        self.assertEqual(metadata["operation"], "tile_upscale")
        self.assertIn("ksml_lineage", metadata)
        
        # Check metadata file
        metadata_file = out_dir / "upscale_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        # Check output resolution
        first_frame = cv2.imread(str(output_frames[0]))
        self.assertIsNotNone(first_frame)
        
        print(f"   ✅ End-to-end upscaling successful")
        print(f"   Input frames: {len(list(in_dir.glob('*.png')))}")
        print(f"   Output frames: {len(output_frames)}")
        print(f"   Output resolution: {first_frame.shape[1]}x{first_frame.shape[0]}")
        print(f"   Processing time: {metadata['processing_time_seconds']:.2f}s")
    
    def test_9_ksml_compliance(self):
        """Test KSML compliance and metadata."""
        print("\n🧪 Test 9: KSML Compliance")
        
        # Create KSML token
        ksml_token = {
            "ksml_token": "test_parent_upscale_123",
            "intent": "test_upscaling",
            "karma_state": "temporal_smoothed"
        }
        
        # Setup directories
        in_dir = self.test_frames_dir
        out_dir = Path(self.temp_dir) / "ksml_upscaled"
        
        # Create upscaler
        upscaler = TileUpscaler(device=self.device)
        
        # Upscale with KSML token
        metadata = upscaler.upscale_frames(
            str(in_dir),
            str(out_dir),
            target_height=512,
            ksml_token=ksml_token
        )
        
        # Check KSML lineage
        self.assertIn("ksml_lineage", metadata)
        lineage = metadata["ksml_lineage"]
        
        self.assertEqual(lineage["parent_token"], "test_parent_upscale_123")
        self.assertEqual(lineage["operation"], "tile_upscale")
        self.assertEqual(lineage["karma_state"], "upscaled_1080p")
        self.assertIn("lineage", lineage)
        
        print(f"   ✅ KSML compliance verified")
        print(f"   Parent token: {lineage['parent_token']}")
        print(f"   Operation: {lineage['operation']}")
        print(f"   Karma state: {lineage['karma_state']}")
    
    def test_10_convenience_function(self):
        """Test convenience function."""
        print("\n🧪 Test 10: Convenience Function")
        
        in_dir = self.test_frames_dir
        out_dir = Path(self.temp_dir) / "convenience_upscaled"
        
        # Use convenience function
        metadata = upscale_frames(
            str(in_dir),
            str(out_dir),
            target_height=512,
            device=self.device
        )
        
        # Check output
        self.assertTrue(out_dir.exists())
        self.assertGreater(len(list(out_dir.glob("*.png"))), 0)
        
        print(f"   ✅ Convenience function works")
        print(f"   Frames processed: {metadata['num_frames']}")
        print(f"   Output resolution: {metadata['output_resolution']}")


def run_tests(verbose=True):
    """Run all tests with detailed output."""
    print("=" * 60)
    print("Tile Upscaler Module - Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTileUpscaler)
    
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
    
    parser = argparse.ArgumentParser(description="Test Tile Upscaler Module")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    result = run_tests(verbose=args.verbose)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
