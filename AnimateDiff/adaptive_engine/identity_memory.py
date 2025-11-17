#!/usr/bin/env python3
"""
Identity Memory - Task 11 Day 1
Persistent character identity tracking across video scenes
Uses face embeddings and visual features for character recognition

Created: November 13, 2025
"""

import os
import cv2
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time


@dataclass
class CharacterIdentity:
    """Character identity information"""
    char_id: str
    gender: str
    embedding: Optional[np.ndarray] = None
    face_images: List[np.ndarray] = field(default_factory=list)
    appearance_features: Dict[str, any] = field(default_factory=dict)
    scene_appearances: List[int] = field(default_factory=list)  # Scene indices
    confidence_scores: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class IdentityMatch:
    """Identity matching result"""
    char_id: str
    confidence: float
    drift: float  # How much appearance has drifted from original


class IdentityMemory:
    """
    Character identity memory system
    
    Tracks characters across video scenes using:
    1. Face embeddings (OpenCV)
    2. Visual features (color, texture)
    3. Consistency scoring
    """
    
    def __init__(self, cache_dir: str = "outputs/identity_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.identities: Dict[str, CharacterIdentity] = {}
        
        # Initialize face detector
        print("🎭 Initializing Identity Memory...")
        self._init_face_detector()
        print("   ✅ Identity Memory ready")
    
    def _init_face_detector(self):
        """Initialize OpenCV face detector"""
        try:
            # Use OpenCV's Haar Cascade (simple and fast)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("   ⚠️ Face detector not loaded, using basic mode")
                self.face_cascade = None
            else:
                print("   ✅ OpenCV face detector initialized")
        except Exception as e:
            print(f"   ⚠️ Face detector init failed: {e}")
            self.face_cascade = None
    
    def register_character(
        self,
        char_id: str,
        gender: str,
        reference_image: Optional[np.ndarray] = None,
        scene_idx: int = 0
    ) -> CharacterIdentity:
        """
        Register a new character identity
        
        Args:
            char_id: Unique character identifier
            gender: Character gender ('male', 'female', 'neutral')
            reference_image: Reference image for character (optional)
            scene_idx: Scene index where character first appears
            
        Returns:
            CharacterIdentity object
        """
        print(f"   📝 Registering character: {char_id} ({gender})")
        
        identity = CharacterIdentity(
            char_id=char_id,
            gender=gender,
            scene_appearances=[scene_idx]
        )
        
        if reference_image is not None:
            # Extract face embedding
            embedding = self._extract_face_embedding(reference_image)
            identity.embedding = embedding
            identity.face_images.append(reference_image)
            
            # Extract appearance features
            features = self._extract_appearance_features(reference_image)
            identity.appearance_features = features
            
            print(f"      ✅ Extracted features from reference image")
        
        self.identities[char_id] = identity
        self._save_identity(identity)
        
        return identity
    
    def recognize_character(
        self,
        image: np.ndarray,
        scene_idx: int,
        threshold: float = 0.7
    ) -> Optional[IdentityMatch]:
        """
        Recognize character from image
        
        Args:
            image: Image containing character
            scene_idx: Current scene index
            threshold: Confidence threshold for matching
            
        Returns:
            IdentityMatch if recognized, None otherwise
        """
        if not self.identities:
            return None
        
        # Extract embedding from current image
        current_embedding = self._extract_face_embedding(image)
        
        if current_embedding is None:
            return None
        
        # Compare with all known identities
        best_match = None
        best_confidence = 0.0
        
        for char_id, identity in self.identities.items():
            if identity.embedding is None:
                continue
            
            # Calculate similarity
            confidence = self._calculate_similarity(
                current_embedding,
                identity.embedding
            )
            
            if confidence > best_confidence and confidence >= threshold:
                best_confidence = confidence
                
                # Calculate drift
                drift = 1.0 - confidence
                
                best_match = IdentityMatch(
                    char_id=char_id,
                    confidence=confidence,
                    drift=drift
                )
        
        # Update identity if matched
        if best_match:
            identity = self.identities[best_match.char_id]
            identity.scene_appearances.append(scene_idx)
            identity.confidence_scores.append(best_match.confidence)
            identity.face_images.append(image)
            self._save_identity(identity)
        
        return best_match
    
    def get_character_consistency(self, char_id: str) -> float:
        """
        Calculate character consistency score across scenes
        
        Args:
            char_id: Character identifier
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if char_id not in self.identities:
            return 0.0
        
        identity = self.identities[char_id]
        
        # If no confidence scores yet (new character or no embeddings), return perfect consistency
        if len(identity.confidence_scores) == 0:
            return 1.0
        
        # Average confidence across all appearances
        avg_confidence = np.mean(identity.confidence_scores)
        
        # Penalize for drift
        if len(identity.confidence_scores) > 1:
            consistency = avg_confidence * (1.0 - np.std(identity.confidence_scores))
        else:
            consistency = avg_confidence
        
        return max(0.0, min(1.0, consistency))
    
    def get_identity_drift(self, char_id: str) -> float:
        """
        Calculate identity drift for character
        
        Args:
            char_id: Character identifier
            
        Returns:
            Drift score (0.0 = no drift, 1.0 = maximum drift)
        """
        if char_id not in self.identities:
            return 1.0
        
        identity = self.identities[char_id]
        
        if len(identity.confidence_scores) <= 1:
            return 0.0
        
        # Calculate drift as decrease in confidence over time
        first_half = identity.confidence_scores[:len(identity.confidence_scores)//2]
        second_half = identity.confidence_scores[len(identity.confidence_scores)//2:]
        
        if len(first_half) > 0 and len(second_half) > 0:
            drift = np.mean(first_half) - np.mean(second_half)
            return max(0.0, drift)
        
        return 0.0
    
    def _extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        Uses simple feature extraction (OpenCV + histogram)
        """
        if image is None or image.size == 0:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect face
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # Use first detected face
                    x, y, w, h = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                else:
                    # Use whole image if no face detected
                    face_roi = gray
            else:
                # Use whole image
                face_roi = gray
            
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (64, 64))
            
            # Extract features using histogram
            hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Combine with resized image as embedding
            img_flat = face_resized.flatten() / 255.0
            embedding = np.concatenate([hist, img_flat])
            
            return embedding
            
        except Exception as e:
            print(f"      ⚠️ Error extracting face embedding: {e}")
            return None
    
    def _extract_appearance_features(self, image: np.ndarray) -> Dict[str, any]:
        """Extract appearance features from image"""
        features = {}
        
        try:
            # Dominant color
            pixels = image.reshape(-1, 3)
            dominant_color = np.mean(pixels, axis=0).astype(int)
            features['dominant_color'] = dominant_color.tolist()
            
            # Brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features['brightness'] = float(np.mean(gray))
            
            # Texture (using standard deviation)
            features['texture'] = float(np.std(gray))
            
        except Exception as e:
            print(f"      ⚠️ Error extracting appearance features: {e}")
        
        return features
    
    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two embeddings
        Uses cosine similarity
        """
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Normalize to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            print(f"      ⚠️ Error calculating similarity: {e}")
            return 0.0
    
    def _save_identity(self, identity: CharacterIdentity):
        """Save identity to cache"""
        try:
            cache_file = self.cache_dir / f"{identity.char_id}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(identity, f)
        except Exception as e:
            print(f"      ⚠️ Error saving identity: {e}")
    
    def load_identity(self, char_id: str) -> Optional[CharacterIdentity]:
        """Load identity from cache"""
        try:
            cache_file = self.cache_dir / f"{char_id}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    identity = pickle.load(f)
                self.identities[char_id] = identity
                return identity
        except Exception as e:
            print(f"      ⚠️ Error loading identity: {e}")
        return None
    
    def get_all_identities(self) -> Dict[str, CharacterIdentity]:
        """Get all registered identities"""
        return self.identities.copy()
    
    def get_all_characters(self) -> Dict[str, Dict]:
        """Get all characters as dictionary (for testing/compatibility)"""
        return {char_id: {
            "char_id": char_id,
            "gender": identity.gender,
            "scenes": identity.scene_appearances,
            "total_appearances": len(identity.scene_appearances)
        } for char_id, identity in self.identities.items()}
    
    def get_character_info(self, char_id: str) -> Optional[Dict]:
        """
        Get character information as dictionary
        
        Returns:
            Dictionary with character info or None if not found
        """
        if char_id not in self.identities:
            return None
        
        identity = self.identities[char_id]
        return {
            "char_id": char_id,
            "gender": identity.gender,
            "total_appearances": len(identity.scene_appearances),
            "scenes": identity.scene_appearances,
            "avg_confidence": np.mean(identity.confidence_scores) if identity.confidence_scores else 1.0,
            "consistency": self.get_character_consistency(char_id),
            "drift": self.get_identity_drift(char_id)
        }
    
    def save_cache(self):
        """Save all identities to cache"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        for char_id, identity in self.identities.items():
            cache_file = self.cache_dir / f"{char_id}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(identity, f)
            except Exception as e:
                print(f"      ⚠️ Error saving {char_id}: {e}")
        
        print(f"   ✅ Saved {len(self.identities)} identities to cache")
    
    def load_cache(self):
        """Load all identities from cache"""
        if not self.cache_dir.exists():
            return
        
        loaded = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    identity = pickle.load(f)
                    self.identities[identity.char_id] = identity
                    loaded += 1
            except Exception as e:
                print(f"      ⚠️ Error loading {cache_file.name}: {e}")
        
        if loaded > 0:
            print(f"   ✅ Loaded {loaded} identities from cache")
    
    def clear_cache(self):
        """Clear identity cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.identities.clear()
        print("   ✅ Identity cache cleared")
    
    def print_summary(self):
        """Print summary of all identities"""
        print("\n" + "="*60)
        print("🎭 IDENTITY MEMORY SUMMARY")
        print("="*60)
        
        if not self.identities:
            print("\n   No identities registered yet")
            return
        
        for char_id, identity in self.identities.items():
            print(f"\n• {char_id}")
            print(f"  Gender: {identity.gender}")
            print(f"  Scenes appeared: {len(identity.scene_appearances)}")
            print(f"  Scenes: {identity.scene_appearances}")
            
            if identity.confidence_scores:
                avg_confidence = np.mean(identity.confidence_scores)
                print(f"  Average confidence: {avg_confidence:.2f}")
            
            consistency = self.get_character_consistency(char_id)
            print(f"  Consistency score: {consistency:.2f}")
            
            drift = self.get_identity_drift(char_id)
            print(f"  Identity drift: {drift:.2f}")
        
        print("\n" + "="*60 + "\n")


# Singleton instance
_identity_memory: Optional[IdentityMemory] = None


def get_identity_memory(cache_dir: str = "outputs/identity_cache") -> IdentityMemory:
    """Get singleton identity memory instance"""
    global _identity_memory
    if _identity_memory is None:
        _identity_memory = IdentityMemory(cache_dir=cache_dir)
    return _identity_memory


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Identity Memory System")
    print("="*60)
    
    # Create test instance
    memory = IdentityMemory(cache_dir="outputs/test_identity_cache")
    
    # Test 1: Register character without image
    print("\n📝 Test 1: Register character (no image)")
    char1 = memory.register_character(
        char_id="seeker_001",
        gender="female",
        scene_idx=0
    )
    print(f"   ✅ Registered: {char1.char_id}")
    
    # Test 2: Register character with dummy image
    print("\n📝 Test 2: Register character (with image)")
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    char2 = memory.register_character(
        char_id="teacher_001",
        gender="male",
        reference_image=dummy_image,
        scene_idx=0
    )
    print(f"   ✅ Registered: {char2.char_id}")
    
    # Test 3: Character recognition
    print("\n📝 Test 3: Recognize character")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    match = memory.recognize_character(test_image, scene_idx=1, threshold=0.5)
    if match:
        print(f"   ✅ Recognized: {match.char_id} (confidence: {match.confidence:.2f})")
    else:
        print("   ℹ️ No match found (expected with random images)")
    
    # Test 4: Get consistency
    print("\n📝 Test 4: Character consistency")
    consistency = memory.get_character_consistency("teacher_001")
    print(f"   Consistency score: {consistency:.2f}")
    
    # Test 5: Print summary
    print("\n📝 Test 5: Identity summary")
    memory.print_summary()
    
    # Cleanup
    memory.clear_cache()
    
    print("\n✅ All tests complete!")
