#!/usr/bin/env python3
"""
Phase 3: Face Identity Preservation System
Advanced face recognition and identity preservation across clips
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import os
import pickle
from pathlib import Path

try:
    # Try to import face recognition libraries
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition not available. Install with: pip install face_recognition")

try:
    # Try to import InsightFace for better face embeddings
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("WARNING: insightface not available. Install with: pip install insightface")

class FaceIdentityPreserver:
    """Advanced face identity preservation system"""
    
    def __init__(self):
        self.reference_embeddings = {}
        self.face_detector = None
        self.embedding_model = None
        self.identity_threshold = 0.6  # Similarity threshold for same identity
        
        # Initialize face detection and embedding models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize face detection and embedding models"""
        
        if INSIGHTFACE_AVAILABLE:
            try:
                # Use InsightFace for better embeddings
                self.face_detector = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
                self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
                print("✅ InsightFace initialized for face identity preservation")
                return
            except Exception as e:
                print(f"⚠️ InsightFace initialization failed: {e}")
        
        if FACE_RECOGNITION_AVAILABLE:
            try:
                # Fallback to face_recognition
                print("✅ face_recognition initialized for face identity preservation")
                return
            except Exception as e:
                print(f"⚠️ face_recognition initialization failed: {e}")
        
        # Fallback to OpenCV Haar cascades
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            print("SUCCESS: OpenCV face detection initialized (basic mode)")
        except Exception as e:
            print(f"❌ No face detection available: {e}")
    
    def extract_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face embedding from image"""
        
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                return None
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_path
        
        if INSIGHTFACE_AVAILABLE and self.face_detector is not None:
            return self._extract_insightface_embedding(image_rgb)
        elif FACE_RECOGNITION_AVAILABLE:
            return self._extract_face_recognition_embedding(image_rgb)
        else:
            return self._extract_basic_face_features(image_rgb)
    
    def _extract_insightface_embedding(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using InsightFace"""
        
        try:
            faces = self.face_detector.get(image_rgb)
            if faces:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
                return largest_face.embedding
        except Exception as e:
            print(f"⚠️ InsightFace embedding extraction failed: {e}")
        
        return None
    
    def _extract_face_recognition_embedding(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using face_recognition"""
        
        try:
            face_encodings = face_recognition.face_encodings(image_rgb)
            if face_encodings:
                return face_encodings[0]  # Return first face encoding
        except Exception as e:
            print(f"⚠️ face_recognition embedding extraction failed: {e}")
        
        return None
    
    def _extract_basic_face_features(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract basic face features using OpenCV (fallback)"""
        
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_region = gray[y:y+h, x:x+w]
                
                # Resize to standard size and flatten as basic "embedding"
                face_resized = cv2.resize(face_region, (64, 64))
                return face_resized.flatten().astype(np.float32)
        except Exception as e:
            print(f"⚠️ Basic face feature extraction failed: {e}")
        
        return None
    
    def calculate_face_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two face embeddings"""
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def establish_identity_reference(self, image_path: str, identity_name: str = "main_character") -> bool:
        """Establish reference identity from first good frame"""
        
        embedding = self.extract_face_embedding(image_path)
        
        if embedding is not None:
            self.reference_embeddings[identity_name] = embedding
            print(f"✅ Identity reference established for '{identity_name}'")
            return True
        else:
            print(f"⚠️ Could not establish identity reference from {image_path}")
            return False
    
    def verify_identity_consistency(self, image_path: str, identity_name: str = "main_character") -> Dict:
        """Verify if face in image matches reference identity"""
        
        if identity_name not in self.reference_embeddings:
            return {
                'is_consistent': False,
                'similarity': 0.0,
                'reason': 'No reference identity established'
            }
        
        current_embedding = self.extract_face_embedding(image_path)
        
        if current_embedding is None:
            return {
                'is_consistent': False,
                'similarity': 0.0,
                'reason': 'No face detected in current image'
            }
        
        reference_embedding = self.reference_embeddings[identity_name]
        similarity = self.calculate_face_similarity(reference_embedding, current_embedding)
        
        is_consistent = similarity >= self.identity_threshold
        
        return {
            'is_consistent': is_consistent,
            'similarity': similarity,
            'threshold': self.identity_threshold,
            'reason': 'Identity verified' if is_consistent else f'Similarity {similarity:.3f} below threshold {self.identity_threshold}'
        }
    
    def save_identity_reference(self, output_dir: str, identity_name: str = "main_character"):
        """Save identity reference to disk"""
        
        if identity_name in self.reference_embeddings:
            reference_path = os.path.join(output_dir, f"{identity_name}_identity.pkl")
            
            with open(reference_path, 'wb') as f:
                pickle.dump(self.reference_embeddings[identity_name], f)
            
            print(f"💾 Identity reference saved to {reference_path}")
    
    def load_identity_reference(self, output_dir: str, identity_name: str = "main_character") -> bool:
        """Load identity reference from disk"""
        
        reference_path = os.path.join(output_dir, f"{identity_name}_identity.pkl")
        
        if os.path.exists(reference_path):
            try:
                with open(reference_path, 'rb') as f:
                    self.reference_embeddings[identity_name] = pickle.load(f)
                
                print(f"✅ Identity reference loaded from {reference_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load identity reference: {e}")
        
        return False
    
    def enhance_prompt_for_identity(self, prompt: str, identity_verification: Dict, style: str = "realistic") -> str:
        """Enhance prompt based on identity verification results and style"""

        if not identity_verification['is_consistent']:
            # Style-specific identity preservation terms
            if style == "realistic":
                identity_terms = [
                    "same person throughout video",
                    "identical facial features",
                    "consistent face structure",
                    "preserve facial identity",
                    "maintain same eye shape",
                    "same nose and lips",
                    "stable facial characteristics"
                ]
            elif style == "anime":
                identity_terms = [
                    "same anime character throughout",
                    "consistent anime face",
                    "identical anime features",
                    "preserve anime character design",
                    "maintain anime character identity",
                    "same anime eye style",
                    "consistent anime character appearance"
                ]
            else:  # artistic
                identity_terms = [
                    "same artistic character throughout",
                    "consistent artistic portrait",
                    "identical artistic features",
                    "preserve artistic character design",
                    "maintain artistic character identity",
                    "same artistic style features",
                    "consistent artistic character appearance"
                ]

            # Add all identity terms - NO TRUNCATION TO PRESERVE STORY
            enhanced_prompt = f"{prompt}, {', '.join(identity_terms)}"
            return enhanced_prompt

        return prompt
    
    def get_identity_consistency_score(self, video_frames: List[str], identity_name: str = "main_character") -> Dict:
        """Calculate identity consistency score across multiple frames"""
        
        if identity_name not in self.reference_embeddings:
            return {'overall_score': 0.0, 'frame_scores': [], 'consistent_frames': 0}
        
        frame_scores = []
        consistent_frames = 0
        
        for frame_path in video_frames:
            verification = self.verify_identity_consistency(frame_path, identity_name)
            frame_scores.append(verification['similarity'])
            
            if verification['is_consistent']:
                consistent_frames += 1
        
        overall_score = np.mean(frame_scores) if frame_scores else 0.0
        consistency_rate = consistent_frames / len(video_frames) if video_frames else 0.0
        
        return {
            'overall_score': overall_score,
            'consistency_rate': consistency_rate,
            'frame_scores': frame_scores,
            'consistent_frames': consistent_frames,
            'total_frames': len(video_frames)
        }

# Global face identity preserver instance
face_identity_preserver = FaceIdentityPreserver()
