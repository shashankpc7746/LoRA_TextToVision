#!/usr/bin/env python3
"""
Character Detection and Extraction System
Detects and extracts character faces from AnimateDiff generated videos for SadTalker processing
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import tempfile
import subprocess
from dataclasses import dataclass

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SadTalker'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SadTalker', 'src'))

@dataclass
class DetectedCharacter:
    """Information about a detected character"""
    frame_number: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    face_image_path: str
    gender_prediction: str  # 'male', 'female', 'unknown'

class CharacterDetector:
    """Character detection and extraction from videos"""
    
    def __init__(self):
        """Initialize the character detector"""
        self.temp_dir = tempfile.mkdtemp(prefix="chardet_")
        self.face_cascade = None
        self._load_face_detector()
        print(f"✅ Character Detector initialized. Temp dir: {self.temp_dir}")
    
    def _load_face_detector(self):
        """Load OpenCV face detection cascade"""
        try:
            # Try to load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("⚠️ Warning: Could not load face cascade")
                self.face_cascade = None
            else:
                print("✅ Face detection cascade loaded successfully")
                
        except Exception as e:
            print(f"⚠️ Warning: Error loading face detector: {e}")
            self.face_cascade = None
    
    def extract_frames(self, video_path: str, max_frames: int = 30) -> List[str]:
        """Extract frames from video for analysis"""
        print(f"🎬 Extracting frames from: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        # Calculate frame sampling interval
        frame_interval = max(1, total_frames // max_frames)
        
        frame_paths = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at intervals
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(self.temp_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
                if len(frame_paths) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Extracted {len(frame_paths)} frames")
        return frame_paths
    
    def detect_faces_in_frame(self, frame_path: str) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces in a single frame"""
        if not self.face_cascade:
            return []
        
        # Read frame
        frame = cv2.imread(frame_path)
        if frame is None:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list with confidence scores (simplified)
        face_list = []
        for (x, y, w, h) in faces:
            # Simple confidence based on face size
            confidence = min(1.0, (w * h) / (100 * 100))
            face_list.append((x, y, w, h, confidence))
        
        return face_list
    
    def extract_face_image(self, frame_path: str, bbox: Tuple[int, int, int, int]) -> str:
        """Extract face region from frame"""
        x, y, w, h = bbox
        
        # Read frame
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        
        # Add padding around face
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract face region
        face_region = frame[y1:y2, x1:x2]
        
        # Save face image
        face_filename = f"face_{hash(frame_path)}_{x}_{y}.jpg"
        face_path = os.path.join(self.temp_dir, face_filename)
        cv2.imwrite(face_path, face_region)
        
        return face_path
    
    def predict_gender(self, face_image_path: str) -> str:
        """Predict gender from face image (simplified)"""
        # This is a simplified gender prediction
        # In a real implementation, you'd use a trained model
        
        try:
            # Read face image
            face = cv2.imread(face_image_path)
            if face is None:
                return "unknown"
            
            # Simple heuristic based on image properties
            # This is very basic and should be replaced with a proper model
            height, width = face.shape[:2]
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            
            # Simple heuristic (not accurate, just for demo)
            # In practice, use a trained gender classification model
            avg_hue = np.mean(hsv[:, :, 0])
            
            # Placeholder logic - replace with actual model
            if avg_hue > 100:
                return "female"
            else:
                return "male"
                
        except Exception as e:
            print(f"⚠️ Error predicting gender: {e}")
            return "unknown"
    
    def detect_characters_in_video(self, video_path: str) -> List[DetectedCharacter]:
        """Detect all characters in a video"""
        print(f"🔍 Detecting characters in: {os.path.basename(video_path)}")
        
        # Extract frames
        frame_paths = self.extract_frames(video_path)
        
        detected_characters = []
        
        for i, frame_path in enumerate(frame_paths):
            print(f"🔍 Analyzing frame {i+1}/{len(frame_paths)}")
            
            # Detect faces in frame
            faces = self.detect_faces_in_frame(frame_path)
            
            for face_bbox in faces:
                x, y, w, h, confidence = face_bbox
                
                # Extract face image
                try:
                    face_image_path = self.extract_face_image(frame_path, (x, y, w, h))
                    
                    # Predict gender
                    gender = self.predict_gender(face_image_path)
                    
                    # Create character detection
                    character = DetectedCharacter(
                        frame_number=i,
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        face_image_path=face_image_path,
                        gender_prediction=gender
                    )
                    
                    detected_characters.append(character)
                    print(f"✅ Found character: {gender} (confidence: {confidence:.2f})")
                    
                except Exception as e:
                    print(f"⚠️ Error processing face: {e}")
                    continue
        
        print(f"✅ Detected {len(detected_characters)} character instances")
        return detected_characters
    
    def get_best_character_image(self, characters: List[DetectedCharacter]) -> Optional[str]:
        """Get the best character image for SadTalker processing"""
        if not characters:
            return None
        
        # Sort by confidence and select best
        best_character = max(characters, key=lambda c: c.confidence)
        
        print(f"🎯 Selected best character: {best_character.gender_prediction} "
              f"(confidence: {best_character.confidence:.2f})")
        
        return best_character.face_image_path
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning up: {e}")

def test_character_detector():
    """Test the character detector"""
    print("🧪 Testing Character Detector...")
    
    detector = CharacterDetector()
    
    # Look for test video files
    test_video_paths = [
        "../AnimateDiff/outputs",
        "../tts_module/results",
        "../tts_module/tts_module/results"
    ]
    
    test_video = None
    for path in test_video_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    test_video = os.path.join(path, file)
                    break
            if test_video:
                break
    
    if test_video:
        print(f"🎬 Testing with video: {test_video}")
        
        try:
            characters = detector.detect_characters_in_video(test_video)
            
            if characters:
                best_image = detector.get_best_character_image(characters)
                print(f"🎯 Best character image: {best_image}")
                
                # Show summary
                print(f"\n{'='*60}")
                print("📊 CHARACTER DETECTION RESULTS:")
                print(f"{'='*60}")
                print(f"Total characters detected: {len(characters)}")
                
                gender_counts = {}
                for char in characters:
                    gender_counts[char.gender_prediction] = gender_counts.get(char.gender_prediction, 0) + 1
                
                for gender, count in gender_counts.items():
                    print(f"{gender.title()}: {count}")
                    
            else:
                print("❌ No characters detected")
                
        except Exception as e:
            print(f"❌ Error testing character detection: {e}")
    else:
        print("⚠️ No test video found. Please generate a video first.")
    
    detector.cleanup()

if __name__ == "__main__":
    test_character_detector()
