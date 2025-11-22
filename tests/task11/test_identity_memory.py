#!/usr/bin/env python3
"""
Unit Tests for Identity Memory - Task 11 Day 1
Tests character identity tracking across video scenes
"""

import pytest
import sys
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Path setup is handled by conftest.py
from adaptive_engine.identity_memory import (
    IdentityMemory,
    get_identity_memory,
    CharacterIdentity,
    IdentityMatch
)


class TestIdentityMemory:
    """Test suite for identity memory"""
    
    def setup_method(self):
        """Setup test instance with temporary cache"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = IdentityMemory(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup temporary cache"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test identity memory initialization"""
        assert self.memory is not None
        assert len(self.memory.identities) == 0
        assert self.memory.face_detector is not None
    
    def test_singleton_pattern(self):
        """Test singleton pattern"""
        memory1 = get_identity_memory()
        memory2 = get_identity_memory()
        assert memory1 is memory2
    
    def test_register_character_without_image(self):
        """Test registering character without image"""
        char_id = self.memory.register_character(
            char_id="seeker_001",
            gender="female",
            image=None,
            scene_idx=0
        )
        
        assert char_id == "seeker_001"
        assert "seeker_001" in self.memory.identities
        
        identity = self.memory.identities["seeker_001"]
        assert identity.gender == "female"
        assert identity.embedding is None
    
    def test_register_character_with_synthetic_image(self):
        """Test registering character with synthetic test image"""
        # Create synthetic test image (100x100 RGB)
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        char_id = self.memory.register_character(
            char_id="teacher_001",
            gender="male",
            image=test_image,
            scene_idx=0
        )
        
        assert char_id == "teacher_001"
        assert "teacher_001" in self.memory.identities
        
        identity = self.memory.identities["teacher_001"]
        assert identity.gender == "male"
        # May or may not extract embedding depending on face detection
        # Just check it doesn't crash
    
    def test_get_character_info(self):
        """Test getting character info"""
        self.memory.register_character("seeker_001", "female", None, 0)
        
        info = self.memory.get_character_info("seeker_001")
        
        assert info is not None
        assert info["char_id"] == "seeker_001"
        assert info["gender"] == "female"
        assert info["total_appearances"] >= 1
    
    def test_get_character_info_nonexistent(self):
        """Test getting info for nonexistent character"""
        info = self.memory.get_character_info("nonexistent_char")
        assert info is None
    
    def test_character_consistency_no_history(self):
        """Test consistency calculation with no history"""
        self.memory.register_character("seeker_001", "female", None, 0)
        
        consistency = self.memory.get_character_consistency("seeker_001")
        
        # With no embeddings or only one appearance, should return 1.0
        assert consistency == 1.0
    
    def test_identity_drift_calculation(self):
        """Test identity drift calculation"""
        # Register character
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.memory.register_character("seeker_001", "female", test_image, 0)
        
        # Add more appearances
        identity = self.memory.identities["seeker_001"]
        identity.confidence_scores = [0.95, 0.90, 0.85, 0.80]
        
        drift = self.memory.get_identity_drift("seeker_001", window_size=4)
        
        assert 0.0 <= drift <= 1.0
    
    def test_identity_drift_nonexistent(self):
        """Test drift calculation for nonexistent character"""
        drift = self.memory.get_identity_drift("nonexistent_char")
        assert drift == 0.0
    
    def test_get_all_characters(self):
        """Test getting all characters"""
        self.memory.register_character("seeker_001", "female", None, 0)
        self.memory.register_character("teacher_001", "male", None, 1)
        
        characters = self.memory.get_all_characters()
        
        assert len(characters) == 2
        assert "seeker_001" in characters
        assert "teacher_001" in characters
    
    def test_calculate_similarity(self):
        """Test similarity calculation"""
        # Create two similar embeddings
        embedding1 = np.random.rand(100)
        embedding2 = embedding1 + np.random.rand(100) * 0.1  # Small difference
        
        similarity = self.memory._calculate_similarity(embedding1, embedding2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be similar
    
    def test_calculate_similarity_identical(self):
        """Test similarity with identical embeddings"""
        embedding = np.random.rand(100)
        
        similarity = self.memory._calculate_similarity(embedding, embedding)
        
        assert similarity == 1.0
    
    def test_cache_persistence(self):
        """Test cache saving and loading"""
        # Register character
        self.memory.register_character("seeker_001", "female", None, 0)
        
        # Save cache
        self.memory.save_cache()
        
        # Create new instance with same cache dir
        new_memory = IdentityMemory(cache_dir=self.temp_dir)
        new_memory.load_cache()
        
        # Check character was loaded
        assert "seeker_001" in new_memory.identities
        assert new_memory.identities["seeker_001"].gender == "female"
    
    def test_clear_cache(self):
        """Test clearing cache"""
        self.memory.register_character("seeker_001", "female", None, 0)
        assert len(self.memory.identities) > 0
        
        self.memory.clear_cache()
        
        assert len(self.memory.identities) == 0


class TestIdentityMemoryRecognition:
    """Test character recognition functionality"""
    
    def setup_method(self):
        """Setup test instance"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = IdentityMemory(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_recognize_character_no_match(self):
        """Test recognition when no characters registered"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        match = self.memory.recognize_character(test_image, scene_idx=0)
        
        assert match is None
    
    def test_recognize_character_with_embedding(self):
        """Test recognition with manual embedding"""
        # Create character with manual embedding
        test_embedding = np.random.rand(100)
        
        identity = CharacterIdentity(
            char_id="seeker_001",
            gender="female",
            embedding=test_embedding
        )
        self.memory.identities["seeker_001"] = identity
        
        # Create synthetic image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Try to recognize (may or may not match depending on embedding extraction)
        match = self.memory.recognize_character(test_image, scene_idx=0, threshold=0.5)
        
        # Just check it doesn't crash
        assert match is None or isinstance(match, IdentityMatch)
    
    def test_extract_face_embedding_synthetic(self):
        """Test face embedding extraction from synthetic image"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        embedding = self.memory._extract_face_embedding(test_image)
        
        # Should return embedding (histogram + resized image)
        assert embedding is not None
        assert len(embedding) > 0
    
    def test_extract_face_embedding_none(self):
        """Test face embedding extraction from None"""
        embedding = self.memory._extract_face_embedding(None)
        assert embedding is None


class TestIdentityMemoryEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup test instance"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = IdentityMemory(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_duplicate_character_id(self):
        """Test registering character with duplicate ID"""
        self.memory.register_character("seeker_001", "female", None, 0)
        
        # Register again with same ID
        char_id = self.memory.register_character("seeker_001", "male", None, 1)
        
        # Should update existing character
        assert char_id == "seeker_001"
        identity = self.memory.identities["seeker_001"]
        # Should have multiple appearances
        assert len(identity.scene_appearances) > 1
    
    def test_invalid_image_dimensions(self):
        """Test handling invalid image dimensions"""
        # 1D image
        invalid_image = np.array([1, 2, 3, 4])
        
        char_id = self.memory.register_character(
            "seeker_001",
            "female",
            invalid_image,
            0
        )
        
        # Should handle gracefully
        assert char_id == "seeker_001"
    
    def test_very_small_image(self):
        """Test handling very small images"""
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        char_id = self.memory.register_character(
            "seeker_001",
            "female",
            small_image,
            0
        )
        
        # Should handle gracefully
        assert char_id == "seeker_001"
    
    def test_performance_many_characters(self):
        """Test performance with many characters"""
        import time
        
        start = time.time()
        
        # Register 50 characters
        for i in range(50):
            self.memory.register_character(f"char_{i:03d}", "female", None, i)
        
        duration = time.time() - start
        
        assert duration < 5.0  # Should complete in under 5 seconds
        assert len(self.memory.identities) == 50
    
    def test_consistency_single_appearance(self):
        """Test consistency with single appearance"""
        self.memory.register_character("seeker_001", "female", None, 0)
        
        consistency = self.memory.get_character_consistency("seeker_001")
        
        # Single appearance should have perfect consistency
        assert consistency == 1.0


class TestCharacterIdentity:
    """Test CharacterIdentity dataclass"""
    
    def test_character_identity_creation(self):
        """Test creating CharacterIdentity"""
        identity = CharacterIdentity(
            char_id="seeker_001",
            gender="female"
        )
        
        assert identity.char_id == "seeker_001"
        assert identity.gender == "female"
        assert identity.embedding is None
        assert len(identity.scene_appearances) == 0
    
    def test_character_identity_with_embedding(self):
        """Test CharacterIdentity with embedding"""
        embedding = np.random.rand(100)
        
        identity = CharacterIdentity(
            char_id="teacher_001",
            gender="male",
            embedding=embedding
        )
        
        assert identity.char_id == "teacher_001"
        assert identity.embedding is not None
        assert len(identity.embedding) == 100


class TestIdentityMatch:
    """Test IdentityMatch dataclass"""
    
    def test_identity_match_creation(self):
        """Test creating IdentityMatch"""
        match = IdentityMatch(
            char_id="seeker_001",
            confidence=0.95,
            drift=0.05
        )
        
        assert match.char_id == "seeker_001"
        assert match.confidence == 0.95
        assert match.drift == 0.05
    
    def test_identity_match_thresholds(self):
        """Test match confidence thresholds"""
        high_match = IdentityMatch("char_001", 0.95, 0.05)
        medium_match = IdentityMatch("char_002", 0.75, 0.25)
        low_match = IdentityMatch("char_003", 0.55, 0.45)
        
        assert high_match.confidence > medium_match.confidence
        assert medium_match.confidence > low_match.confidence


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
