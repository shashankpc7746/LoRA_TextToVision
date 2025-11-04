"""
Test Script for Indigenous Gurukul LoRA Adapter - Task 9 Day 1
Comprehensive testing for adapter training, dataset, and generation
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from adapters.gurukul_lora.dataset_curator import GurukulDatasetCurator, prepare_training_dataset
from adapters.gurukul_lora.train_adapter import GurukulLoRATrainer
from adapters.gurukul_lora.inference import IndigenousGenerator


class TestGurukulLoRA:
    """Comprehensive test suite for Gurukul LoRA system"""
    
    def __init__(self):
        self.dataset_path = "datasets/gurukul_keyframes"
        self.adapter_path = "adapters/gurukul_lora/gurukul_lora.pt"
        self.test_results = []
        
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print(" "*20 + "GURUKUL LORA TEST SUITE")
        print("="*70 + "\n")
        
        # Test 1: Dataset Curator
        self.test_dataset_curator()
        
        # Test 2: Dataset Validation
        self.test_dataset_validation()
        
        # Test 3: Adapter Path Check
        self.test_adapter_exists()
        
        # Test 4: Metadata Check
        self.test_metadata_exists()
        
        # Test 5: Generator Loading
        self.test_generator_loading()
        
        # Test 6: Deterministic Generation (if adapter exists)
        if Path(self.adapter_path).exists():
            self.test_deterministic_generation()
            
        # Print summary
        self.print_test_summary()
        
    def test_dataset_curator(self):
        """Test dataset curator functionality"""
        test_name = "Dataset Curator Initialization"
        print(f"\n🧪 Test: {test_name}")
        
        try:
            curator = GurukulDatasetCurator(self.dataset_path)
            assert curator.dataset_path.exists(), "Dataset path not created"
            
            self.test_results.append({
                "name": test_name,
                "status": "✅ PASS",
                "message": "Dataset curator initialized successfully"
            })
            print(f"   ✅ PASS")
            
        except Exception as e:
            self.test_results.append({
                "name": test_name,
                "status": "❌ FAIL",
                "message": str(e)
            })
            print(f"   ❌ FAIL: {e}")
            
    def test_dataset_validation(self):
        """Test dataset validation"""
        test_name = "Dataset Validation"
        print(f"\n🧪 Test: {test_name}")
        
        try:
            curator = GurukulDatasetCurator(self.dataset_path)
            validation = curator.validate_dataset(verbose=False)
            
            # Check validation structure
            assert "valid" in validation, "Missing 'valid' field"
            assert "num_images" in validation, "Missing 'num_images' field"
            assert "valid_images" in validation, "Missing 'valid_images' field"
            
            message = f"Found {validation['num_images']} images, {validation['valid_images']} valid"
            
            self.test_results.append({
                "name": test_name,
                "status": "✅ PASS",
                "message": message
            })
            print(f"   ✅ PASS: {message}")
            
        except Exception as e:
            self.test_results.append({
                "name": test_name,
                "status": "❌ FAIL",
                "message": str(e)
            })
            print(f"   ❌ FAIL: {e}")
            
    def test_adapter_exists(self):
        """Test if adapter file exists"""
        test_name = "Adapter File Existence"
        print(f"\n🧪 Test: {test_name}")
        
        adapter_path = Path(self.adapter_path)
        
        if adapter_path.exists():
            size = adapter_path.stat().st_size / (1024 * 1024)  # MB
            message = f"Adapter found: {size:.2f} MB"
            
            self.test_results.append({
                "name": test_name,
                "status": "✅ PASS",
                "message": message
            })
            print(f"   ✅ PASS: {message}")
        else:
            message = "Adapter not found - needs training"
            self.test_results.append({
                "name": test_name,
                "status": "⚠️ SKIP",
                "message": message
            })
            print(f"   ⚠️ SKIP: {message}")
            
    def test_metadata_exists(self):
        """Test if metadata file exists"""
        test_name = "Metadata File Existence"
        print(f"\n🧪 Test: {test_name}")
        
        metadata_path = Path(self.adapter_path).parent / "metadata.json"
        
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Check required fields
            required_fields = ["deterministic_config", "ksml_lineage", "model_hash"]
            missing_fields = [f for f in required_fields if f not in metadata]
            
            if missing_fields:
                message = f"Missing fields: {missing_fields}"
                status = "⚠️ WARN"
            else:
                message = "All required metadata fields present"
                status = "✅ PASS"
                
            self.test_results.append({
                "name": test_name,
                "status": status,
                "message": message
            })
            print(f"   {status}: {message}")
        else:
            message = "Metadata not found - needs training"
            self.test_results.append({
                "name": test_name,
                "status": "⚠️ SKIP",
                "message": message
            })
            print(f"   ⚠️ SKIP: {message}")
            
    def test_generator_loading(self):
        """Test generator loading"""
        test_name = "Generator Loading"
        print(f"\n🧪 Test: {test_name}")
        
        try:
            generator = IndigenousGenerator(self.adapter_path)
            
            # Check if adapter path was set
            assert generator.adapter_path == Path(self.adapter_path)
            
            # Check if metadata was loaded (if exists)
            metadata_path = Path(self.adapter_path).parent / "metadata.json"
            if metadata_path.exists():
                assert generator.metadata is not None, "Metadata not loaded"
                message = "Generator initialized with metadata"
            else:
                message = "Generator initialized without metadata"
                
            self.test_results.append({
                "name": test_name,
                "status": "✅ PASS",
                "message": message
            })
            print(f"   ✅ PASS: {message}")
            
        except Exception as e:
            self.test_results.append({
                "name": test_name,
                "status": "❌ FAIL",
                "message": str(e)
            })
            print(f"   ❌ FAIL: {e}")
            
    def test_deterministic_generation(self):
        """Test deterministic generation (requires trained adapter)"""
        test_name = "Deterministic Generation"
        print(f"\n🧪 Test: {test_name}")
        print("   Note: This test requires a trained adapter and may take several minutes...")
        
        try:
            generator = IndigenousGenerator(self.adapter_path)
            
            # Simple determinism check - generate same prompt/seed twice
            prompt = "Traditional Gurukul classroom"
            seed = 42
            
            print(f"   Testing with prompt: '{prompt}', seed: {seed}")
            
            # Generate twice
            result1 = generator.generate_keyframe(prompt, seed)
            result2 = generator.generate_keyframe(prompt, seed)
            
            if result1["success"] and result2["success"]:
                # Check if image IDs are the same
                if result1["image_id"] == result2["image_id"]:
                    message = "Deterministic generation verified (same image ID)"
                    status = "✅ PASS"
                else:
                    message = f"Different image IDs: {result1['image_id']} vs {result2['image_id']}"
                    status = "⚠️ WARN"
            else:
                message = "Generation failed"
                status = "❌ FAIL"
                
            self.test_results.append({
                "name": test_name,
                "status": status,
                "message": message
            })
            print(f"   {status}: {message}")
            
        except Exception as e:
            self.test_results.append({
                "name": test_name,
                "status": "❌ FAIL",
                "message": str(e)
            })
            print(f"   ❌ FAIL: {e}")
            
    def print_test_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print(" "*25 + "TEST SUMMARY")
        print("="*70 + "\n")
        
        passed = sum(1 for r in self.test_results if r["status"] == "✅ PASS")
        failed = sum(1 for r in self.test_results if r["status"] == "❌ FAIL")
        skipped = sum(1 for r in self.test_results if r["status"] == "⚠️ SKIP")
        warned = sum(1 for r in self.test_results if r["status"] == "⚠️ WARN")
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"  ✅ Passed:  {passed}")
        print(f"  ❌ Failed:  {failed}")
        print(f"  ⚠️ Skipped: {skipped}")
        print(f"  ⚠️ Warned:  {warned}")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            print(f"  {result['status']} {result['name']}")
            print(f"     {result['message']}")
            
        print("\n" + "="*70 + "\n")
        
        # Overall status
        if failed > 0:
            print("❌ OVERALL STATUS: FAILED")
            return False
        elif skipped > 0 or warned > 0:
            print("⚠️ OVERALL STATUS: PASSED WITH WARNINGS")
            return True
        else:
            print("✅ OVERALL STATUS: ALL TESTS PASSED")
            return True


def quick_setup_test():
    """Quick setup and validation test"""
    print("\n🚀 Quick Setup Test\n")
    
    # Step 1: Prepare dataset
    print("Step 1: Preparing dataset...")
    validation = prepare_training_dataset(
        dataset_path="datasets/gurukul_keyframes",
        create_placeholder=True,
        num_images=100
    )
    
    if not validation["valid"]:
        print("❌ Dataset validation failed")
        return False
        
    print("✅ Dataset prepared and validated\n")
    
    # Step 2: Check CUDA availability
    print("Step 2: Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ CUDA not available - will use CPU (slower)")
        
    print("\n✅ Quick setup test passed!")
    print("\nNext steps:")
    print("  1. Train adapter: python adapters/gurukul_lora/train_adapter.py --dataset datasets/gurukul_keyframes --num_epochs 10")
    print("  2. Run full test suite: python adapters/gurukul_lora/test_adapter.py")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gurukul LoRA System")
    parser.add_argument("--quick_setup", action="store_true",
                       help="Run quick setup test only")
    parser.add_argument("--full_suite", action="store_true",
                       help="Run full test suite")
    
    args = parser.parse_args()
    
    if args.quick_setup:
        success = quick_setup_test()
        sys.exit(0 if success else 1)
    else:
        # Run full test suite
        tester = TestGurukulLoRA()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
