"""
Visible Watermarking Module
Adds visible watermark overlays to videos for psychological deterrent
Works alongside invisible watermarks for multi-layer security
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Literal
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


class VisibleWatermarker:
    """Add visible watermarks to videos"""
    
    # Watermark styles
    STYLE_SUBTLE = "subtle"        # Small corner logo (5% opacity)
    STYLE_MODERATE = "moderate"    # Larger corner logo (15% opacity)
    STYLE_PROMINENT = "prominent"  # Full watermark (30% opacity)
    STYLE_DEMO = "demo"           # Large "DEMO" overlay (restricted mode)
    
    def __init__(self, logo_path: Optional[str] = None):
        """
        Initialize visible watermarker
        
        Args:
            logo_path: Path to logo image (PNG with transparency)
                      Default: security/watermark_logo/BHI_logo.png
        """
        # Default logo path
        if logo_path is None:
            current_dir = Path(__file__).parent
            default_logo = current_dir / "watermark_logo" / "BHI_logo.png"
            if default_logo.exists():
                logo_path = str(default_logo)
        
        self.logo_path = logo_path
        self.logo_image = None
        
        if logo_path and Path(logo_path).exists():
            self.logo_image = Image.open(logo_path).convert('RGBA')
            print(f"✅ Loaded logo: {Path(logo_path).name}")
    
    def add_corner_watermark(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        position: Literal["top-right", "top-left", "bottom-right", "bottom-left"] = "bottom-right",
        opacity: float = 0.5,
        scale: float = 0.1,
        text: Optional[str] = None,
        build_id: Optional[str] = None
    ) -> str:
        """
        Add logo watermark to video corner using OpenCV
        
        Args:
            video_path: Input video path
            output_path: Output video path
            position: Corner position
            opacity: Watermark opacity (0.0 to 1.0)
            scale: Logo size relative to video width (0.05 to 0.3)
            text: Text to display below logo (optional)
            build_id: Build ID to include in text (optional)
        
        Returns:
            Path to watermarked video
        """
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_visible_wm{video_path_obj.suffix}")
        
        # Check if logo is available
        if self.logo_image is None:
            print("⚠️  No logo image found, skipping visible watermark")
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create output writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Resize logo
        logo_width = int(width * scale)
        logo_pil = self.logo_image.copy()
        
        # Maintain aspect ratio
        aspect_ratio = logo_pil.height / logo_pil.width
        logo_height = int(logo_width * aspect_ratio)
        logo_pil = logo_pil.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        
        # Convert PIL to OpenCV
        logo_array = np.array(logo_pil)
        
        # Check if logo has alpha channel
        if logo_array.shape[2] == 4:
            logo_rgb = cv2.cvtColor(logo_array[:, :, :3], cv2.COLOR_RGB2BGR)
            logo_alpha = logo_array[:, :, 3] / 255.0 * opacity  # Apply opacity
        else:
            logo_rgb = cv2.cvtColor(logo_array, cv2.COLOR_RGB2BGR)
            logo_alpha = np.ones((logo_height, logo_width)) * opacity
        
        # Calculate position with padding
        padding = 20
        positions_map = {
            "top-right": (width - logo_width - padding, padding),
            "top-left": (padding, padding),
            "bottom-right": (width - logo_width - padding, height - logo_height - padding),
            "bottom-left": (padding, height - logo_height - padding)
        }
        x, y = positions_map[position]
        
        print(f"📹 Processing video: {os.path.basename(video_path)}")
        print(f"   Logo: {Path(self.logo_path).name}")
        print(f"   Position: {position}")
        print(f"   Opacity: {opacity * 100:.0f}%")
        print(f"   Logo size: {logo_width}x{logo_height} ({scale*100:.0f}% of video width)")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ensure logo fits in frame
            if x + logo_width <= width and y + logo_height <= height:
                # Extract region of interest
                roi = frame[y:y+logo_height, x:x+logo_width]
                
                # Blend logo with frame
                for c in range(3):  # BGR channels
                    roi[:, :, c] = (logo_alpha * logo_rgb[:, :, c] + 
                                   (1 - logo_alpha) * roi[:, :, c])
                
                # Place blended region back
                frame[y:y+logo_height, x:x+logo_width] = roi
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"   Processed {frame_count} frames...", end='\r')
        
        cap.release()
        out.release()
        
        print(f"\n✅ Logo watermark added: {frame_count} frames")
        return output_path
    
    def add_dynamic_watermark(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        watermark_type: Literal["timestamp", "frame_number", "build_id"] = "timestamp",
        position: str = "bottom-right",
        opacity: float = 0.6,
        build_id: Optional[str] = None
    ) -> str:
        """
        Add dynamic watermark that changes per frame
        
        Args:
            video_path: Input video path
            output_path: Output video path
            watermark_type: Type of dynamic watermark
            position: Corner position
            opacity: Watermark opacity
            build_id: Build ID
        
        Returns:
            Path to watermarked video
        """
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_dynamic_wm{video_path_obj.suffix}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = datetime.now()
        
        print(f"📹 Adding dynamic {watermark_type} watermark...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Generate dynamic text
            if watermark_type == "timestamp":
                elapsed = frame_count / fps
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                milliseconds = int((elapsed % 1) * 1000)
                text = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            elif watermark_type == "frame_number":
                text = f"Frame: {frame_count:05d}"
            elif watermark_type == "build_id":
                text = f"{build_id or 'dev_build'} | F{frame_count:05d}"
            
            # Add text to frame
            overlay = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Position
            if position == "bottom-right":
                x = width - text_width - 20
                y = height - 20
            elif position == "bottom-left":
                x = 20
                y = height - 20
            elif position == "top-right":
                x = width - text_width - 20
                y = 30
            else:  # top-left
                x = 20
                y = 30
            
            # Background rectangle
            cv2.rectangle(
                overlay,
                (x - 5, y - text_height - 5),
                (x + text_width + 5, y + 5),
                (0, 0, 0),
                -1
            )
            
            # Text
            cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            # Blend
            frame = cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"✅ Dynamic watermark added: {frame_count} frames")
        return output_path
    
    def add_demo_watermark(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Add large "DEMO" watermark for restricted mode
        
        Args:
            video_path: Input video path
            output_path: Output video path
        
        Returns:
            Path to watermarked video
        """
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_DEMO{video_path_obj.suffix}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"📹 Adding DEMO watermark (restricted mode)...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Large diagonal "DEMO" text
            overlay = frame.copy()
            font = cv2.FONT_HERSHEY_BOLD
            font_scale = width / 200  # Scale with video width
            font_thickness = max(2, int(font_scale * 3))
            
            text = "DEMO - RESTRICTED"
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Center position
            x = (width - text_width) // 2
            y = (height + text_height) // 2
            
            # Add text with outline
            cv2.putText(overlay, text, (x, y), font, font_scale, (0, 0, 0), font_thickness + 4, cv2.LINE_AA)  # Outline
            cv2.putText(overlay, text, (x, y), font, font_scale, (255, 50, 50), font_thickness, cv2.LINE_AA)  # Red text
            
            # Blend (30% opacity)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"✅ DEMO watermark added: {frame_count} frames")
        return output_path


# Convenience functions
def add_visible_watermark(
    video_path: str,
    style: str = "subtle",
    build_id: Optional[str] = None,
    restricted_mode: bool = False
) -> str:
    """
    Add visible watermark with predefined style
    
    Args:
        video_path: Input video path
        style: Watermark style (subtle, moderate, prominent, demo)
        build_id: Build ID to include
        restricted_mode: If True, adds DEMO watermark
    
    Returns:
        Path to watermarked video
    """
    watermarker = VisibleWatermarker()
    
    if restricted_mode:
        return watermarker.add_demo_watermark(video_path)
    
    # Style presets
    style_config = {
        "subtle": {"opacity": 0.15, "scale": 0.08, "position": "bottom-right"},
        "moderate": {"opacity": 0.30, "scale": 0.12, "position": "bottom-right"},
        "prominent": {"opacity": 0.50, "scale": 0.15, "position": "top-right"},
    }
    
    config = style_config.get(style, style_config["subtle"])
    
    return watermarker.add_corner_watermark(
        video_path,
        position=config["position"],
        opacity=config["opacity"],
        scale=config["scale"],
        build_id=build_id
    )


if __name__ == "__main__":
    import tempfile
    
    print("\n" + "="*70)
    print("VISIBLE WATERMARK DEMO")
    print("="*70)
    
    # Create dummy video with opencv
    print("\nCreating test video...")
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        test_video = f.name
    
    # Create simple video (10 frames)
    width, height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, fps, (width, height))
    
    for i in range(10):
        # Create gradient frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [100 + i * 10, 150, 200]  # Blue gradient
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {os.path.basename(test_video)}")
    
    # Test 1: Subtle watermark
    print("\n" + "="*70)
    print("Test 1: Subtle Corner Watermark (Production Style)")
    print("="*70)
    watermarker = VisibleWatermarker()
    result1 = watermarker.add_corner_watermark(
        test_video,
        position="bottom-right",
        opacity=0.15,
        scale=0.08,
        build_id="build_20251106_001"
    )
    
    # Test 2: Demo watermark
    print("\n" + "="*70)
    print("Test 2: DEMO Watermark (Restricted Mode)")
    print("="*70)
    result2 = watermarker.add_demo_watermark(test_video)
    
    # Cleanup
    print("\n" + "="*70)
    print("Cleanup")
    print("="*70)
    os.unlink(test_video)
    print(f"✅ Deleted test files")
    
    if os.path.exists(result1):
        print(f"📹 Subtle watermark: {result1}")
        print("   (Would delete in real test)")
    
    if os.path.exists(result2):
        print(f"📹 Demo watermark: {result2}")
        print("   (Would delete in real test)")
    
    print("\n✅ Visible watermark demo complete!")
