"""
Smart Video Extender - Day 5 of TTV Studio Intelligence Stack

Purpose:
    Extends short video clips intelligently without repetitive looping.
    Fixes the production issue: "2-sec clip loops 3x → Repetitive"

Features:
    1. Slow Motion Extension - 24fps → 16fps smooth slowdown (NO RIFE)
    2. Smart Freeze - Freeze last frame with subtle zoom effect
    3. Frame Blending - Smooth interpolation between frames
    4. Duration Matching - Extend clips to match audio duration

Solution:
    Instead of: Loop(2s clip × 3) = Repetitive 6s video ❌
    We do: SlowMotion(2s → 4s) + SmartFreeze(2s) = Natural 6s video ✅

Author: TTV Studio Team
Created: November 17, 2025
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2


class ExtensionMethod(Enum):
    """Video extension methods"""
    SLOW_MOTION = "slow_motion"      # 24fps → 16fps (1.5x duration)
    SMART_FREEZE = "smart_freeze"    # Freeze with zoom
    BLEND = "blend"                   # Frame blending
    COMBINED = "combined"             # SlowMo + Freeze


@dataclass
class ExtensionParams:
    """Parameters for video extension"""
    method: ExtensionMethod
    target_duration: float           # Desired duration in seconds
    slow_motion_factor: float = 1.5  # Speed reduction (1.5 = 50% slower)
    freeze_zoom_amount: float = 0.1  # Zoom during freeze (10%)
    freeze_zoom_speed: float = 0.02  # Zoom speed per frame
    blend_window: int = 3            # Frames to blend together


class SmartVideoExtender:
    """
    Extends video clips intelligently without repetitive looping
    
    CRITICAL: NO RIFE! Using simple frame interpolation to avoid black screens.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(SmartVideoExtender, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize smart video extender"""
        if self._initialized:
            return
        
        self.extension_stats = {
            'total_extended': 0,
            'slow_motion_count': 0,
            'freeze_count': 0,
            'blend_count': 0
        }
        
        self._initialized = True
        print("🎬 Smart Video Extender initialized (NO RIFE - Safe mode)")
    
    # ======================== SLOW MOTION EXTENSION ========================
    
    def apply_slow_motion(
        self,
        frames: np.ndarray,
        slow_factor: float = 1.5,
        fps: float = 24.0
    ) -> Tuple[np.ndarray, float]:
        """
        Apply slow motion to extend video duration
        
        Method: Simple frame duplication (NO RIFE to avoid black screens)
        24fps → 16fps = 1.5x duration increase
        
        Args:
            frames: Video frames (T, H, W, C)
            slow_factor: Speed reduction factor (1.5 = 50% slower)
            fps: Original framerate
        
        Returns:
            Tuple of (extended_frames, new_fps)
        """
        if len(frames) == 0:
            return frames, fps
        
        # Calculate new framerate
        new_fps = fps / slow_factor
        
        # Simple frame duplication for smooth slow motion
        # Instead of complex RIFE interpolation, just repeat frames strategically
        num_original_frames = len(frames)
        num_new_frames = int(num_original_frames * slow_factor)
        
        extended_frames = []
        for i in range(num_new_frames):
            # Map new frame index to original frame
            original_idx = int(i / slow_factor)
            original_idx = min(original_idx, num_original_frames - 1)
            extended_frames.append(frames[original_idx])
        
        extended_frames = np.array(extended_frames)
        
        self.extension_stats['slow_motion_count'] += 1
        self.extension_stats['total_extended'] += 1
        
        return extended_frames, new_fps
    
    def apply_slow_motion_blend(
        self,
        frames: np.ndarray,
        slow_factor: float = 1.5,
        fps: float = 24.0,
        blend_weight: float = 0.3
    ) -> Tuple[np.ndarray, float]:
        """
        Apply slow motion with frame blending for smoother result
        
        Args:
            frames: Video frames (T, H, W, C)
            slow_factor: Speed reduction factor
            fps: Original framerate
            blend_weight: Blending weight (0.0 - 1.0)
        
        Returns:
            Tuple of (extended_frames, new_fps)
        """
        if len(frames) == 0:
            return frames, fps
        
        # First apply basic slow motion
        extended_frames, new_fps = self.apply_slow_motion(frames, slow_factor, fps)
        
        # Then blend adjacent frames for smoothness
        blended_frames = []
        for i in range(len(extended_frames)):
            current_frame = extended_frames[i].astype(np.float32)
            
            # Blend with next frame if available
            if i < len(extended_frames) - 1:
                next_frame = extended_frames[i + 1].astype(np.float32)
                blended = (1 - blend_weight) * current_frame + blend_weight * next_frame
                blended_frames.append(blended.astype(np.uint8))
            else:
                blended_frames.append(current_frame.astype(np.uint8))
        
        self.extension_stats['blend_count'] += 1
        
        return np.array(blended_frames), new_fps
    
    # ======================== SMART FREEZE EXTENSION ========================
    
    def apply_smart_freeze(
        self,
        frames: np.ndarray,
        freeze_duration: float,
        fps: float = 24.0,
        zoom_amount: float = 0.1,
        zoom_speed: float = 0.02
    ) -> np.ndarray:
        """
        Freeze last frame with subtle zoom effect
        
        Creates natural-looking extension by freezing on last frame
        with subtle zoom-in to maintain visual interest.
        
        Args:
            frames: Video frames (T, H, W, C)
            freeze_duration: How long to freeze in seconds
            fps: Framerate
            zoom_amount: Total zoom (0.1 = 10% zoom)
            zoom_speed: Zoom per frame (0.02 = 2% per frame)
        
        Returns:
            Extended frames with freeze
        """
        if len(frames) == 0:
            return frames
        
        # Get last frame
        last_frame = frames[-1]
        height, width = last_frame.shape[:2]
        
        # Calculate number of freeze frames
        num_freeze_frames = int(freeze_duration * fps)
        
        freeze_frames = []
        for i in range(num_freeze_frames):
            # Calculate current zoom level
            zoom_progress = min(i * zoom_speed, zoom_amount)
            zoom_factor = 1.0 + zoom_progress
            
            # Apply zoom by cropping and resizing
            new_width = int(width / zoom_factor)
            new_height = int(height / zoom_factor)
            
            # Center crop
            x_start = (width - new_width) // 2
            y_start = (height - new_height) // 2
            
            cropped = last_frame[y_start:y_start+new_height, x_start:x_start+new_width]
            zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            
            freeze_frames.append(zoomed)
        
        # Combine original frames with freeze frames
        extended = np.concatenate([frames, np.array(freeze_frames)], axis=0)
        
        self.extension_stats['freeze_count'] += 1
        self.extension_stats['total_extended'] += 1
        
        return extended
    
    # ======================== COMBINED EXTENSION ========================
    
    def extend_to_duration(
        self,
        frames: np.ndarray,
        current_duration: float,
        target_duration: float,
        fps: float = 24.0,
        method: ExtensionMethod = ExtensionMethod.COMBINED
    ) -> Tuple[np.ndarray, float]:
        """
        Extend video to match target duration
        
        Strategy:
        1. If target < 2x current: Use slow motion only
        2. If target >= 2x current: Use slow motion + freeze
        
        Args:
            frames: Video frames (T, H, W, C)
            current_duration: Current duration in seconds
            target_duration: Target duration in seconds
            fps: Framerate
            method: Extension method to use
        
        Returns:
            Tuple of (extended_frames, new_fps)
        """
        if target_duration <= current_duration:
            # No extension needed
            return frames, fps
        
        extension_needed = target_duration - current_duration
        
        if method == ExtensionMethod.SLOW_MOTION:
            # Pure slow motion
            slow_factor = target_duration / current_duration
            return self.apply_slow_motion_blend(frames, slow_factor, fps)
        
        elif method == ExtensionMethod.SMART_FREEZE:
            # Pure freeze
            extended = self.apply_smart_freeze(frames, extension_needed, fps)
            return extended, fps
        
        elif method == ExtensionMethod.COMBINED:
            # Smart combination: SlowMo first, then freeze if needed
            max_slow_factor = 1.5  # Don't slow more than 50%
            
            # Apply slow motion first
            extended_frames, new_fps = self.apply_slow_motion_blend(
                frames, max_slow_factor, fps
            )
            new_duration = current_duration * max_slow_factor
            
            # If still short, add freeze
            if new_duration < target_duration:
                freeze_duration = target_duration - new_duration
                extended_frames = self.apply_smart_freeze(
                    extended_frames, freeze_duration, new_fps
                )
            
            return extended_frames, new_fps
        
        else:
            # Default: return original
            return frames, fps
    
    # ======================== UTILITY METHODS ========================
    
    def calculate_extension_strategy(
        self,
        current_duration: float,
        target_duration: float
    ) -> dict:
        """
        Calculate optimal extension strategy
        
        Args:
            current_duration: Current clip duration
            target_duration: Target duration
        
        Returns:
            Dictionary with strategy details
        """
        extension_ratio = target_duration / current_duration
        
        strategy = {
            'extension_needed': target_duration - current_duration,
            'extension_ratio': extension_ratio,
            'recommended_method': None,
            'slow_motion_duration': 0.0,
            'freeze_duration': 0.0
        }
        
        if extension_ratio <= 1.5:
            # Pure slow motion is enough
            strategy['recommended_method'] = ExtensionMethod.SLOW_MOTION
            strategy['slow_motion_duration'] = target_duration
        elif extension_ratio <= 2.0:
            # Slow motion + small freeze
            strategy['recommended_method'] = ExtensionMethod.COMBINED
            strategy['slow_motion_duration'] = current_duration * 1.5
            strategy['freeze_duration'] = target_duration - (current_duration * 1.5)
        else:
            # Significant freeze needed
            strategy['recommended_method'] = ExtensionMethod.COMBINED
            strategy['slow_motion_duration'] = current_duration * 1.5
            strategy['freeze_duration'] = target_duration - (current_duration * 1.5)
        
        return strategy
    
    def get_stats(self) -> dict:
        """Get extension statistics"""
        return self.extension_stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.extension_stats = {
            'total_extended': 0,
            'slow_motion_count': 0,
            'freeze_count': 0,
            'blend_count': 0
        }


# Singleton instance getter
def get_smart_video_extender() -> SmartVideoExtender:
    """Get singleton instance of SmartVideoExtender"""
    return SmartVideoExtender()
