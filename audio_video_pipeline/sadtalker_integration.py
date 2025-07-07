#!/usr/bin/env python3
"""
SadTalker Lip-Sync Integration
Integrates SadTalker functionality for lip-sync processing of character faces
"""

import os
import sys
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple
from dataclasses import dataclass

# Add SadTalker to path
SADTALKER_DIR = os.path.join(os.path.dirname(__file__), '..', 'SadTalker')
sys.path.append(SADTALKER_DIR)
sys.path.append(os.path.join(SADTALKER_DIR, 'src'))

@dataclass
class LipSyncResult:
    """Result of lip-sync processing"""
    success: bool
    output_video_path: str
    error_message: str = ""
    processing_time: float = 0.0

class SadTalkerIntegration:
    """SadTalker integration for lip-sync processing"""
    
    def __init__(self, checkpoint_dir: str = None, result_dir: str = None):
        """Initialize SadTalker integration"""
        self.sadtalker_dir = SADTALKER_DIR
        self.checkpoint_dir = checkpoint_dir or os.path.join(self.sadtalker_dir, 'checkpoints')
        self.result_dir = result_dir or os.path.join(os.path.dirname(__file__), 'results')
        self.temp_dir = tempfile.mkdtemp(prefix="sadtalker_")
        
        # Ensure directories exist
        os.makedirs(self.result_dir, exist_ok=True)
        
        print(f"✅ SadTalker Integration initialized")
        print(f"📁 SadTalker dir: {self.sadtalker_dir}")
        print(f"📁 Checkpoint dir: {self.checkpoint_dir}")
        print(f"📁 Result dir: {self.result_dir}")
        print(f"📁 Temp dir: {self.temp_dir}")
    
    def check_dependencies(self) -> bool:
        """Check if SadTalker dependencies are available"""
        try:
            # Check if inference.py exists
            inference_path = os.path.join(self.sadtalker_dir, 'inference.py')
            if not os.path.exists(inference_path):
                print(f"❌ SadTalker inference.py not found at: {inference_path}")
                return False
            
            # Check if checkpoints directory exists
            if not os.path.exists(self.checkpoint_dir):
                print(f"❌ Checkpoint directory not found: {self.checkpoint_dir}")
                return False
            
            print("✅ SadTalker dependencies check passed")
            return True
            
        except Exception as e:
            print(f"❌ Error checking dependencies: {e}")
            return False
    
    def prepare_inputs(self, audio_path: str, image_path: str) -> Tuple[str, str]:
        """Prepare and validate input files"""
        # Convert to absolute paths
        audio_path = os.path.abspath(audio_path)
        image_path = os.path.abspath(image_path)
        
        # Validate files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"✅ Input validation passed")
        print(f"🎵 Audio: {os.path.basename(audio_path)}")
        print(f"🖼️ Image: {os.path.basename(image_path)}")
        
        return audio_path, image_path
    
    def run_sadtalker_inference(self, audio_path: str, image_path: str, 
                               still_mode: bool = True, size: int = 256,
                               enhancer: str = None) -> LipSyncResult:
        """Run SadTalker inference for lip-sync generation"""
        import time
        start_time = time.time()
        
        try:
            print(f"🎬 Starting SadTalker lip-sync processing...")
            
            # Prepare inputs
            audio_path, image_path = self.prepare_inputs(audio_path, image_path)
            
            # Create unique result directory
            import uuid
            session_id = str(uuid.uuid4())[:8]
            session_result_dir = os.path.join(self.result_dir, f"sadtalker_{session_id}")
            os.makedirs(session_result_dir, exist_ok=True)
            
            # Build SadTalker command
            python_executable = sys.executable
            cmd = [
                python_executable, "inference.py",
                "--driven_audio", audio_path,
                "--source_image", image_path,
                "--result_dir", session_result_dir,
                "--size", str(size),
                "--batch_size", "2",  # Faster processing
            ]
            
            # Add optional parameters
            if still_mode:
                cmd.extend(["--still"])
                cmd.extend(["--preprocess", "crop"])
            
            if enhancer:
                cmd.extend(["--enhancer", enhancer])
            
            print(f"🔧 Running command: {' '.join(cmd)}")
            
            # Run SadTalker inference
            result = subprocess.run(
                cmd,
                cwd=self.sadtalker_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ SadTalker inference completed successfully")
            
            # Find the generated video
            output_video = self._find_generated_video(session_result_dir)
            
            if output_video:
                # Copy to final location
                final_output = os.path.join(self.result_dir, f"lipsync_{session_id}.mp4")
                shutil.copy2(output_video, final_output)
                
                processing_time = time.time() - start_time
                
                print(f"🎉 Lip-sync video generated: {os.path.basename(final_output)}")
                print(f"⏱️ Processing time: {processing_time:.2f}s")
                
                return LipSyncResult(
                    success=True,
                    output_video_path=final_output,
                    processing_time=processing_time
                )
            else:
                return LipSyncResult(
                    success=False,
                    output_video_path="",
                    error_message="Generated video not found in results",
                    processing_time=time.time() - start_time
                )
                
        except subprocess.CalledProcessError as e:
            error_msg = f"SadTalker inference failed: {e.stderr}"
            print(f"❌ {error_msg}")
            
            return LipSyncResult(
                success=False,
                output_video_path="",
                error_message=error_msg,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            
            return LipSyncResult(
                success=False,
                output_video_path="",
                error_message=error_msg,
                processing_time=time.time() - start_time
            )
    
    def _find_generated_video(self, result_dir: str) -> Optional[str]:
        """Find the generated video file in results directory"""
        try:
            # Look for common SadTalker output patterns
            patterns = [
                "*.mp4",
                "*enhanced.mp4",
                "*_enhanced.mp4"
            ]
            
            import glob
            for pattern in patterns:
                files = glob.glob(os.path.join(result_dir, "**", pattern), recursive=True)
                if files:
                    # Return the first (and usually only) match
                    return files[0]
            
            # If no pattern matches, list all mp4 files
            all_mp4s = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
            if all_mp4s:
                return all_mp4s[0]
            
            print(f"⚠️ No video files found in: {result_dir}")
            return None
            
        except Exception as e:
            print(f"⚠️ Error finding generated video: {e}")
            return None
    
    def apply_lipsync_to_character(self, character_image_path: str, 
                                  dialogue_audio_path: str) -> LipSyncResult:
        """Apply lip-sync to a character image with dialogue audio"""
        print(f"👄 Applying lip-sync to character...")
        print(f"🖼️ Character: {os.path.basename(character_image_path)}")
        print(f"🎵 Dialogue: {os.path.basename(dialogue_audio_path)}")
        
        return self.run_sadtalker_inference(
            audio_path=dialogue_audio_path,
            image_path=character_image_path,
            still_mode=True,  # Better for single character
            size=256,  # Faster processing
            enhancer=None  # Skip enhancement for speed
        )
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning up: {e}")

def test_sadtalker_integration():
    """Test SadTalker integration"""
    print("🧪 Testing SadTalker Integration...")
    
    integration = SadTalkerIntegration()
    
    # Check dependencies
    if not integration.check_dependencies():
        print("❌ SadTalker dependencies not available")
        return
    
    # Look for test files
    test_audio = None
    test_image = None
    
    # Look for audio files
    audio_paths = [
        "../tts_module/results",
        "../audio_video_pipeline"
    ]
    
    for path in audio_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.wav'):
                    test_audio = os.path.join(path, file)
                    break
            if test_audio:
                break
    
    # Look for image files (from character detector)
    if not test_image:
        # Create a simple test image if none available
        import cv2
        import numpy as np
        
        # Create a simple test face image
        test_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        cv2.circle(test_img, (128, 128), 80, (200, 200, 200), -1)  # Face
        cv2.circle(test_img, (100, 100), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_img, (156, 100), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(test_img, (128, 140), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        test_image = os.path.join(integration.temp_dir, "test_face.jpg")
        cv2.imwrite(test_image, test_img)
        print(f"📷 Created test image: {test_image}")
    
    if test_audio and test_image:
        print(f"🎵 Test audio: {test_audio}")
        print(f"🖼️ Test image: {test_image}")
        
        # Run lip-sync test
        result = integration.apply_lipsync_to_character(test_image, test_audio)
        
        if result.success:
            print(f"✅ Lip-sync test successful!")
            print(f"📹 Output: {result.output_video_path}")
            print(f"⏱️ Time: {result.processing_time:.2f}s")
        else:
            print(f"❌ Lip-sync test failed: {result.error_message}")
    else:
        print("⚠️ No test files available. Please generate audio first.")
    
    integration.cleanup()

if __name__ == "__main__":
    test_sadtalker_integration()
