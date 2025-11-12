"""
Task 10 Integration Test
Tests security features integration into video generation pipeline
"""
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_security_modules_import():
    """Test 1: Verify security modules can be imported"""
    print("\n" + "="*70)
    print("TEST 1: Security Modules Import")
    print("="*70)
    
    try:
        from security import (
            ksml_encrypt, ksml_decrypt,
            sign_artifact, verify_artifact,
            embed_watermark, detect_watermark, compute_fingerprint
        )
        from security.visible_watermark import add_visible_watermark
        from security.runtime_validator import RuntimeKeyIssuer, RuntimeKeyValidator
        from audit_logger import get_audit_logger
        
        print("✅ All security modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_watermarking():
    """Test 2: Verify watermarking works on a test video"""
    print("\n" + "="*70)
    print("TEST 2: Watermarking Integration")
    print("="*70)
    
    try:
        from security import embed_watermark, detect_watermark, compute_fingerprint
        from security.visible_watermark import add_visible_watermark
        import cv2
        import numpy as np
        
        # Create test video
        print("   📹 Creating test video...")
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            test_video = f.name
        
        width, height = 640, 480
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video, fourcc, fps, (width, height))
        
        for i in range(30):  # 1 second
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [100 + i, 150, 200]
            out.write(frame)
        
        out.release()
        print(f"   ✅ Test video created: {os.path.basename(test_video)}")
        
        # Test invisible watermark
        print("   💧 Testing invisible watermark...")
        build_id = "test_build_001"
        watermarked = embed_watermark(test_video, build_id=build_id)
        
        if os.path.exists(watermarked):
            print(f"   ✅ Invisible watermark applied")
        else:
            print(f"   ⚠️ Invisible watermarking skipped (ffmpeg may not be available)")
        
        # Test visible watermark
        print("   🎨 Testing visible watermark...")
        logo_watermarked = add_visible_watermark(test_video, style="subtle", build_id=build_id)
        
        if os.path.exists(logo_watermarked):
            print(f"   ✅ Visible logo watermark applied")
        else:
            print(f"   ❌ Visible watermarking failed")
            return False
        
        # Test fingerprinting
        print("   🔍 Testing fingerprinting...")
        fingerprint = compute_fingerprint(test_video, build_id=build_id)
        
        if fingerprint and 'sha256' in fingerprint:
            print(f"   ✅ Fingerprint computed: {fingerprint['sha256'][:16]}...")
        else:
            print(f"   ❌ Fingerprinting failed")
            return False
        
        # Cleanup
        os.unlink(test_video)
        if os.path.exists(watermarked):
            os.unlink(watermarked)
        if os.path.exists(logo_watermarked):
            os.unlink(logo_watermarked)
        
        print("✅ Watermarking test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Watermarking test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_runtime_key_validation():
    """Test 3: Verify runtime key validation works"""
    print("\n" + "="*70)
    print("TEST 3: Runtime Key Validation")
    print("="*70)
    
    try:
        from security.runtime_validator import RuntimeKeyIssuer, RuntimeKeyValidator
        from datetime import timedelta
        
        # Issue a test key
        print("   🔑 Issuing test runtime key...")
        issuer = RuntimeKeyIssuer()
        runtime_key = issuer.issue_runtime_key(
            worker_id="test-worker-001",
            lifetime_hours=12
        )
        
        print(f"   ✅ Key issued: {runtime_key[:32]}...")
        
        # Validate the key
        print("   🔍 Validating runtime key...")
        # Get public key from private key and save temporarily
        public_key = issuer.private_key.public_key()
        from cryptography.hazmat.primitives import serialization
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
            f.write(public_key_pem)
            temp_public_key_path = f.name
        
        try:
            validator = RuntimeKeyValidator(public_key_path=temp_public_key_path)
            is_valid, key_data = validator.validate_runtime_key(runtime_key)
        
            if is_valid and key_data and key_data.get('worker_id') == "test-worker-001":
                print(f"   ✅ Key validation PASSED")
            else:
                print(f"   ❌ Key validation FAILED")
                return False
            
            # Test invalid key
            print("   🔍 Testing invalid key rejection...")
            is_valid, _ = validator.validate_runtime_key("invalid_key_12345")
            
            if not is_valid:
                print(f"   ✅ Invalid key correctly rejected")
            else:
                print(f"   ❌ Invalid key was accepted (should have been rejected)")
                return False
            
            print("✅ Runtime key validation test PASSED")
            return True
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_public_key_path):
                os.unlink(temp_public_key_path)
        
    except Exception as e:
        print(f"❌ Runtime key validation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_artifact_signing():
    """Test 4: Verify artifact signing works"""
    print("\n" + "="*70)
    print("TEST 4: Artifact Signing")
    print("="*70)
    
    try:
        from security import sign_artifact, verify_artifact
        
        # Create test artifact
        print("   📦 Creating test artifact...")
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(b'test model weights data')
            test_artifact = f.name
        
        print(f"   ✅ Test artifact created: {os.path.basename(test_artifact)}")
        
        # Sign the artifact
        print("   🔏 Signing artifact...")
        signature_path = sign_artifact(test_artifact, metadata={
            "model_type": "test_model",
            "version": "1.0.0",
            "build_id": "test_build_001"
        })
        
        if os.path.exists(signature_path):
            print(f"   ✅ Signature created: {os.path.basename(signature_path)}")
        else:
            print(f"   ❌ Signature creation failed")
            return False
        
        # Verify the signature
        print("   🔍 Verifying signature...")
        is_valid = verify_artifact(test_artifact)
        
        if is_valid:
            print(f"   ✅ Signature verification PASSED")
        else:
            print(f"   ❌ Signature verification FAILED")
            os.unlink(test_artifact)
            if os.path.exists(signature_path):
                os.unlink(signature_path)
            return False
        
        # Cleanup
        os.unlink(test_artifact)
        if os.path.exists(signature_path):
            os.unlink(signature_path)
        
        print("✅ Artifact signing test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Artifact signing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audit_logging():
    """Test 5: Verify audit logging works"""
    print("\n" + "="*70)
    print("TEST 5: Audit Logging")
    print("="*70)
    
    try:
        from audit_logger import get_audit_logger
        
        # Get audit logger
        print("   📝 Initializing audit logger...")
        logger = get_audit_logger(log_dir="logs/audit_test")
        
        # Log a test operation
        print("   📝 Logging test operation...")
        entry_id = logger.log_video_generation(
            prompt="Test video generation",
            output_path="test_output.mp4",
            ksml_token={"ksml_token": "test_token", "intent": "test", "karma_state": "test", "lineage": {}},
            quality_metrics={"duration": 10.0, "fps": 30},
            security_metadata={
                "build_id": "test_build_001",
                "artifact_hash": "abc123",
                "watermark_id": "test_build_001",
                "signed": False
            }
        )
        
        print(f"   ✅ Log entry created: {entry_id}")
        
        # Verify log file exists
        log_dir = Path("logs/audit_test")
        log_files = list(log_dir.glob("audit_*.jsonl"))
        
        if log_files:
            print(f"   ✅ Log file created: {log_files[0].name}")
        else:
            print(f"   ❌ Log file not found")
            return False
        
        # Read and verify log entry
        import json
        with open(log_files[0], 'r') as f:
            last_line = list(f)[-1]
            entry = json.loads(last_line)
            
            if entry.get('entry_id') == entry_id:
                print(f"   ✅ Log entry verified")
            else:
                print(f"   ❌ Log entry mismatch")
                return False
        
        print("✅ Audit logging test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Audit logging test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("TASK 10: SECURITY INTEGRATION TESTS")
    print("="*70)
    print(f"Testing security features integration...")
    print(f"Project root: {project_root}")
    
    tests = [
        ("Security Modules Import", test_security_modules_import),
        ("Watermarking Integration", test_watermarking),
        ("Runtime Key Validation", test_runtime_key_validation),
        ("Artifact Signing", test_artifact_signing),
        ("Audit Logging", test_audit_logging)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Security integration complete!")
        return 0
    else:
        print(f"⚠️ {total - passed} test(s) failed - review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
