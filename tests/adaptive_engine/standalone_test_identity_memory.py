#!/usr/bin/env python3
"""
Standalone Test for Identity Memory - Task 11 Day 1
Tests character identity tracking without pytest dependency
"""
import sys
import os
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Setup path
animatediff_path = Path(__file__).parent.parent.parent / "AnimateDiff"
sys.path.insert(0, str(animatediff_path))
os.chdir(str(animatediff_path))

from adaptive_engine.identity_memory import (
    IdentityMemory,
    get_identity_memory,
    CharacterIdentity,
    IdentityMatch
)

def run_tests():
    """Run all identity memory tests"""
    print("=" * 70)
    print("🧪 Running Identity Memory Tests - Task 11 Day 1")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    # Create temp directory for tests
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Initialization
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            assert memory is not None
            assert len(memory.identities) == 0
            print("✅ test_initialization PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ test_initialization FAILED: {e}")
            failed += 1
        
        # Test 2: Register character without image
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            identity = memory.register_character("seeker_001", "female", None, 0)
            assert identity is not None
            assert identity.char_id == "seeker_001"
            assert "seeker_001" in memory.identities
            assert memory.identities["seeker_001"].gender == "female"
            print("✅ test_register_character_without_image PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ test_register_character_without_image FAILED: {e}")
            failed += 1
        
        # Test 3: Register character with synthetic image
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            identity = memory.register_character("teacher_001", "male", test_image, 0)
            assert identity is not None
            assert identity.char_id == "teacher_001"
            assert "teacher_001" in memory.identities
            print("✅ test_register_character_with_image PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ test_register_character_with_image FAILED: {e}")
            failed += 1
        
        # Test 4: Get character info
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            memory.register_character("seeker_001", "female", None, 0)
            info = memory.get_character_info("seeker_001")
            assert info is not None
            assert info["char_id"] == "seeker_001"
            assert info["gender"] == "female"
            print("✅ test_get_character_info PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ test_get_character_info FAILED: {e}")
            failed += 1
        
        # Test 5: Character consistency
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            memory.register_character("seeker_001", "female", None, 0)
            consistency = memory.get_character_consistency("seeker_001")
            assert consistency == 1.0  # Single appearance = perfect consistency
            print(f"✅ test_character_consistency PASSED (score: {consistency:.2f})")
            passed += 1
        except Exception as e:
            print(f"❌ test_character_consistency FAILED: {e}")
            failed += 1
        
        # Test 6: Similarity calculation
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            embedding1 = np.random.rand(100)
            embedding2 = embedding1.copy()  # Identical
            similarity = memory._calculate_similarity(embedding1, embedding2)
            assert similarity == 1.0
            print(f"✅ test_similarity_identical PASSED (similarity: {similarity:.2f})")
            passed += 1
        except Exception as e:
            print(f"❌ test_similarity_identical FAILED: {e}")
            failed += 1
        
        # Test 7: Get all characters
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            memory.register_character("seeker_001", "female", None, 0)
            memory.register_character("teacher_001", "male", None, 1)
            characters = memory.get_all_characters()
            assert len(characters) == 2
            assert "seeker_001" in characters
            assert "teacher_001" in characters
            print(f"✅ test_get_all_characters PASSED ({len(characters)} characters)")
            passed += 1
        except Exception as e:
            print(f"❌ test_get_all_characters FAILED: {e}")
            failed += 1
        
        # Test 8: Face embedding extraction
        try:
            memory = IdentityMemory(cache_dir=temp_dir)
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            embedding = memory._extract_face_embedding(test_image)
            assert embedding is not None
            assert len(embedding) > 0
            print(f"✅ test_extract_face_embedding PASSED (embedding size: {len(embedding)})")
            passed += 1
        except Exception as e:
            print(f"❌ test_extract_face_embedding FAILED: {e}")
            failed += 1
        
        # Test 9: Cache persistence
        try:
            # Create memory and save
            memory1 = IdentityMemory(cache_dir=temp_dir)
            memory1.register_character("seeker_001", "female", None, 0)
            memory1.save_cache()
            
            # Load in new instance
            memory2 = IdentityMemory(cache_dir=temp_dir)
            memory2.load_cache()
            
            assert "seeker_001" in memory2.identities
            assert memory2.identities["seeker_001"].gender == "female"
            print("✅ test_cache_persistence PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ test_cache_persistence FAILED: {e}")
            failed += 1
        
        # Test 10: Singleton pattern
        try:
            memory1 = get_identity_memory()
            memory2 = get_identity_memory()
            assert memory1 is memory2
            print("✅ test_singleton_pattern PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ test_singleton_pattern FAILED: {e}")
            failed += 1
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print()
    print("=" * 70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
