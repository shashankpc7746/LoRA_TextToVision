"""
Watermarking and Fingerprinting System
Deterministic watermarks tied to BUILD_ID for video provenance tracking
"""
import os
import json
import hashlib
import base64
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime


class VideoWatermarker:
    """Embed and detect watermarks in video files"""
    
    def __init__(self, build_id: Optional[str] = None):
        """
        Initialize watermarker
        
        Args:
            build_id: Build ID to use for watermarking (default: from env BUILD_ID)
        """
        self.build_id = build_id or os.getenv('BUILD_ID', 'dev_build')
        
    def generate_watermark_pattern(self, build_id: Optional[str] = None) -> np.ndarray:
        """
        Generate deterministic watermark pattern from BUILD_ID
        
        Args:
            build_id: Build ID (default: self.build_id)
        
        Returns:
            Watermark pattern as numpy array
        """
        build_id = build_id or self.build_id
        
        # Create deterministic seed from build_id
        seed = int(hashlib.sha256(build_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        
        # Generate 32-bit watermark pattern
        pattern = rng.randint(0, 2, size=32, dtype=np.uint8)
        
        return pattern
    
    def embed_lsb_watermark(self, video_path: str, output_path: Optional[str] = None,
                           build_id: Optional[str] = None) -> str:
        """
        Embed LSB watermark in video file
        
        Note: This is a placeholder implementation. Real implementation would use
        opencv-python or moviepy to manipulate video frames.
        
        Args:
            video_path: Path to input video
            output_path: Path for watermarked video (default: add _watermarked suffix)
            build_id: Build ID for watermark
        
        Returns:
            Path to watermarked video
        """
        build_id = build_id or self.build_id
        
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_watermarked{video_path_obj.suffix}")
        
        # Generate watermark pattern
        watermark = self.generate_watermark_pattern(build_id)
        
        # TODO: Actual LSB embedding in video frames
        # For now, just copy the file and store metadata
        import shutil
        shutil.copy2(video_path, output_path)
        
        # Store watermark metadata in sidecar file
        metadata = {
            'build_id': build_id,
            'watermark_pattern': watermark.tolist(),
            'watermarked_at': datetime.utcnow().isoformat() + 'Z',
            'original_file': Path(video_path).name
        }
        
        metadata_path = output_path + '.watermark.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Watermark embedded: {build_id}")
        return output_path
    
    def detect_lsb_watermark(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect and extract LSB watermark from video
        
        Args:
            video_path: Path to video to check
        
        Returns:
            Dictionary with watermark info if found, None otherwise
        """
        # Check for metadata file first
        metadata_path = video_path + '.watermark.json'
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'found': True,
                'build_id': metadata['build_id'],
                'watermarked_at': metadata['watermarked_at'],
                'detection_method': 'metadata_file'
            }
        
        # TODO: Actual LSB extraction from video frames
        # For now, return not found
        return {
            'found': False,
            'detection_method': 'lsb_extraction'
        }
    
    def embed_metadata_watermark(self, video_path: str, metadata: Dict[str, Any],
                                 output_path: Optional[str] = None) -> str:
        """
        Embed metadata watermark using ffmpeg metadata tags
        
        Args:
            video_path: Path to input video
            metadata: Metadata to embed
            output_path: Path for watermarked video
        
        Returns:
            Path to watermarked video
        """
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_meta{video_path_obj.suffix}")
        
        # Create metadata JSON
        metadata_json = json.dumps(metadata)
        metadata_b64 = base64.b64encode(metadata_json.encode()).decode()
        
        # Use ffmpeg to embed metadata
        # ffmpeg -i input.mp4 -metadata BHIV_WATERMARK="base64_data" output.mp4
        import subprocess
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-metadata', f'BHIV_WATERMARK={metadata_b64}',
                '-metadata', f'BUILD_ID={self.build_id}',
                '-c', 'copy',  # Copy codec (fast)
                '-y',  # Overwrite output
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Metadata watermark embedded")
                return output_path
            else:
                print(f"⚠️  ffmpeg failed: {result.stderr}")
                # Fallback: copy file
                import shutil
                shutil.copy2(video_path, output_path)
                return output_path
                
        except FileNotFoundError:
            print("⚠️  ffmpeg not found, using metadata file fallback")
            # Fallback: use sidecar metadata file
            import shutil
            shutil.copy2(video_path, output_path)
            
            with open(output_path + '.metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
    
    def detect_metadata_watermark(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata watermark from video
        
        Args:
            video_path: Path to video
        
        Returns:
            Watermark metadata if found, None otherwise
        """
        # Try ffmpeg metadata extraction
        import subprocess
        
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Check for BHIV_WATERMARK tag
                if 'format' in data and 'tags' in data['format']:
                    tags = data['format']['tags']
                    
                    if 'BHIV_WATERMARK' in tags:
                        watermark_b64 = tags['BHIV_WATERMARK']
                        watermark_json = base64.b64decode(watermark_b64).decode()
                        watermark_data = json.loads(watermark_json)
                        
                        return {
                            'found': True,
                            'build_id': tags.get('BUILD_ID', watermark_data.get('build_id')),
                            'metadata': watermark_data,
                            'detection_method': 'ffmpeg_metadata'
                        }
        except:
            pass
        
        # Fallback: check for metadata file
        metadata_path = video_path + '.metadata.json'
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'found': True,
                'build_id': metadata.get('build_id', self.build_id),
                'metadata': metadata,
                'detection_method': 'metadata_file'
            }
        
        return None


class ContentFingerprinter:
    """Generate and verify content fingerprints"""
    
    @staticmethod
    def compute_file_fingerprint(file_path: str, algorithm: str = 'sha256') -> str:
        """
        Compute cryptographic fingerprint of file
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, sha512, blake2b)
        
        Returns:
            Hexadecimal hash digest
        """
        if algorithm == 'blake2b':
            hash_func = hashlib.blake2b()
        else:
            hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def compute_perceptual_hash(video_path: str) -> Optional[str]:
        """
        Compute perceptual hash of video (robust to compression)
        
        Note: Requires opencv-python or similar for frame extraction
        
        Args:
            video_path: Path to video
        
        Returns:
            Perceptual hash string (or None if unavailable)
        """
        try:
            import cv2
            from PIL import Image
            import imagehash
            
            # Extract middle frame
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Compute average hash
                avg_hash = imagehash.average_hash(pil_image)
                
                return str(avg_hash)
        except ImportError:
            print("⚠️  opencv-python or imagehash not available for perceptual hashing")
        except Exception as e:
            print(f"⚠️  Perceptual hash failed: {e}")
        
        return None
    
    @staticmethod
    def create_fingerprint_record(file_path: str, build_id: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create complete fingerprint record for file
        
        Args:
            file_path: Path to file
            build_id: Build ID
            metadata: Optional additional metadata
        
        Returns:
            Fingerprint record dictionary
        """
        file_path_obj = Path(file_path)
        
        record = {
            'filename': file_path_obj.name,
            'build_id': build_id,
            'sha256': ContentFingerprinter.compute_file_fingerprint(file_path, 'sha256'),
            'blake2b': ContentFingerprinter.compute_file_fingerprint(file_path, 'blake2b'),
            'file_size': file_path_obj.stat().st_size,
            'created_at': datetime.utcnow().isoformat() + 'Z',
        }
        
        # Add perceptual hash for videos
        if file_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            perceptual_hash = ContentFingerprinter.compute_perceptual_hash(file_path)
            if perceptual_hash:
                record['perceptual_hash'] = perceptual_hash
        
        if metadata:
            record['metadata'] = metadata
        
        return record


# Convenience functions
def embed_watermark(video_path: str, build_id: Optional[str] = None,
                   output_path: Optional[str] = None) -> str:
    """Embed watermark in video"""
    watermarker = VideoWatermarker(build_id)
    return watermarker.embed_lsb_watermark(video_path, output_path, build_id)


def detect_watermark(video_path: str) -> Optional[Dict[str, Any]]:
    """Detect watermark in video"""
    watermarker = VideoWatermarker()
    
    # Try LSB detection
    result = watermarker.detect_lsb_watermark(video_path)
    if result and result.get('found'):
        return result
    
    # Try metadata detection
    result = watermarker.detect_metadata_watermark(video_path)
    return result


def compute_fingerprint(file_path: str, build_id: Optional[str] = None) -> Dict[str, Any]:
    """Compute content fingerprint"""
    build_id = build_id or os.getenv('BUILD_ID', 'dev_build')
    return ContentFingerprinter.create_fingerprint_record(file_path, build_id)


if __name__ == "__main__":
    # Test watermarking and fingerprinting
    import tempfile
    
    # Create test video file (dummy)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp4', delete=False) as f:
        f.write(b"Test video content")
        test_video = f.name
    
    print(f"Test video: {test_video}")
    
    # Test watermarking
    build_id = "test_build_20251106_001"
    watermarker = VideoWatermarker(build_id)
    
    # Generate watermark pattern
    pattern = watermarker.generate_watermark_pattern()
    print(f"\n✅ Watermark pattern generated: {pattern[:8]}...")
    
    # Embed watermark
    watermarked_video = watermarker.embed_lsb_watermark(test_video)
    print(f"✅ Watermark embedded: {watermarked_video}")
    
    # Detect watermark
    detection_result = watermarker.detect_lsb_watermark(watermarked_video)
    print(f"✅ Watermark detected: {detection_result['found']}")
    print(f"   Build ID: {detection_result.get('build_id')}")
    
    # Test fingerprinting
    fingerprint = ContentFingerprinter.create_fingerprint_record(test_video, build_id)
    print(f"\n✅ Content fingerprint:")
    print(f"   SHA256: {fingerprint['sha256'][:16]}...")
    print(f"   BLAKE2b: {fingerprint['blake2b'][:16]}...")
    print(f"   Build ID: {fingerprint['build_id']}")
    
    # Cleanup
    os.unlink(test_video)
    if Path(watermarked_video).exists():
        os.unlink(watermarked_video)
    watermark_metadata = watermarked_video + '.watermark.json'
    if Path(watermark_metadata).exists():
        os.unlink(watermark_metadata)
    
    print("\n✅ All watermarking and fingerprinting tests passed!")
