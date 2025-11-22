"""
Cinematic Transition Core - Day 5 of TTV Studio Intelligence Stack

Purpose:
    Professional transitions between scenes for smooth video flow.
    Enhances production quality with cinematic effects.

Features:
    1. Fade Transitions - Fade to black/white between scenes
    2. Dissolve Transitions - Cross-dissolve/blend between scenes
    3. Cut Transitions - Direct cuts with optional flash
    4. Wipe Transitions - Directional wipes (left, right, up, down)

Integration:
    Works with scene_memory_core to choose transitions based on narrative flow.

Author: TTV Studio Team
Created: November 17, 2025
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2


class TransitionType(Enum):
    """Types of cinematic transitions"""
    CUT = "cut"                      # Direct cut (0 frames)
    FADE_BLACK = "fade_black"        # Fade to black, then in
    FADE_WHITE = "fade_white"        # Fade to white, then in
    DISSOLVE = "dissolve"            # Cross-dissolve/blend
    WIPE_LEFT = "wipe_left"         # Wipe from right to left
    WIPE_RIGHT = "wipe_right"       # Wipe from left to right
    WIPE_UP = "wipe_up"             # Wipe from bottom to top
    WIPE_DOWN = "wipe_down"         # Wipe from top to bottom


@dataclass
class TransitionParams:
    """Parameters for transitions"""
    transition_type: TransitionType
    duration: float = 0.5            # Duration in seconds
    fps: float = 24.0                # Framerate
    easing: str = "linear"           # Easing function
    color: Tuple[int, int, int] = (0, 0, 0)  # Color for fades


class CinematicTransitionCore:
    """
    Professional transitions between video scenes
    
    Provides smooth, cinematic transitions to enhance video flow.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(CinematicTransitionCore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize transition core"""
        if self._initialized:
            return
        
        self.transition_stats = {
            'total_transitions': 0,
            'fade_count': 0,
            'dissolve_count': 0,
            'wipe_count': 0,
            'cut_count': 0
        }
        
        self._initialized = True
        print("🎬 Cinematic Transition Core initialized")
    
    # ======================== EASING FUNCTIONS ========================
    
    def apply_easing(self, t: float, easing: str = "linear") -> float:
        """
        Apply easing function to transition progress
        
        Args:
            t: Progress from 0.0 to 1.0
            easing: Easing type
        
        Returns:
            Eased progress value
        """
        if easing == "linear":
            return t
        elif easing == "ease_in":
            return t * t
        elif easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif easing == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        else:
            return t
    
    # ======================== FADE TRANSITIONS ========================
    
    def create_fade_transition(
        self,
        clip_a_end: np.ndarray,
        clip_b_start: np.ndarray,
        duration: float = 0.5,
        fps: float = 24.0,
        fade_color: Tuple[int, int, int] = (0, 0, 0),
        easing: str = "ease_in_out"
    ) -> np.ndarray:
        """
        Create fade transition (A → Color → B)
        
        Args:
            clip_a_end: Last frames of clip A (T, H, W, C)
            clip_b_start: First frames of clip B (T, H, W, C)
            duration: Transition duration in seconds
            fps: Framerate
            fade_color: Color to fade to (usually black or white)
            easing: Easing function
        
        Returns:
            Transition frames
        """
        num_frames = int(duration * fps)
        if num_frames == 0:
            return np.array([])
        
        # Get last frame of A and first frame of B
        frame_a = clip_a_end[-1] if len(clip_a_end) > 0 else None
        frame_b = clip_b_start[0] if len(clip_b_start) > 0 else None
        
        if frame_a is None or frame_b is None:
            return np.array([])
        
        height, width = frame_a.shape[:2]
        
        # Create solid color frame
        color_frame = np.full((height, width, 3), fade_color, dtype=np.uint8)
        
        transition_frames = []
        half_frames = num_frames // 2
        
        # First half: Fade out A to color
        for i in range(half_frames):
            t = i / half_frames
            t_eased = self.apply_easing(t, easing)
            
            # Blend frame A with color
            blended = cv2.addWeighted(
                frame_a.astype(np.float32), 1 - t_eased,
                color_frame.astype(np.float32), t_eased,
                0
            ).astype(np.uint8)
            transition_frames.append(blended)
        
        # Second half: Fade in B from color
        for i in range(num_frames - half_frames):
            t = i / (num_frames - half_frames)
            t_eased = self.apply_easing(t, easing)
            
            # Blend color with frame B
            blended = cv2.addWeighted(
                color_frame.astype(np.float32), 1 - t_eased,
                frame_b.astype(np.float32), t_eased,
                0
            ).astype(np.uint8)
            transition_frames.append(blended)
        
        self.transition_stats['fade_count'] += 1
        self.transition_stats['total_transitions'] += 1
        
        return np.array(transition_frames)
    
    # ======================== DISSOLVE TRANSITIONS ========================
    
    def create_dissolve_transition(
        self,
        clip_a_end: np.ndarray,
        clip_b_start: np.ndarray,
        duration: float = 0.5,
        fps: float = 24.0,
        easing: str = "linear"
    ) -> np.ndarray:
        """
        Create cross-dissolve transition (A blends into B)
        
        Args:
            clip_a_end: Last frames of clip A
            clip_b_start: First frames of clip B
            duration: Transition duration
            fps: Framerate
            easing: Easing function
        
        Returns:
            Transition frames
        """
        num_frames = int(duration * fps)
        if num_frames == 0:
            return np.array([])
        
        frame_a = clip_a_end[-1] if len(clip_a_end) > 0 else None
        frame_b = clip_b_start[0] if len(clip_b_start) > 0 else None
        
        if frame_a is None or frame_b is None:
            return np.array([])
        
        transition_frames = []
        for i in range(num_frames):
            t = i / num_frames
            t_eased = self.apply_easing(t, easing)
            
            # Blend A and B
            blended = cv2.addWeighted(
                frame_a.astype(np.float32), 1 - t_eased,
                frame_b.astype(np.float32), t_eased,
                0
            ).astype(np.uint8)
            transition_frames.append(blended)
        
        self.transition_stats['dissolve_count'] += 1
        self.transition_stats['total_transitions'] += 1
        
        return np.array(transition_frames)
    
    # ======================== WIPE TRANSITIONS ========================
    
    def create_wipe_transition(
        self,
        clip_a_end: np.ndarray,
        clip_b_start: np.ndarray,
        direction: str = "left",
        duration: float = 0.5,
        fps: float = 24.0,
        easing: str = "linear"
    ) -> np.ndarray:
        """
        Create wipe transition (B wipes over A in specified direction)
        
        Args:
            clip_a_end: Last frames of clip A
            clip_b_start: First frames of clip B
            direction: Wipe direction ('left', 'right', 'up', 'down')
            duration: Transition duration
            fps: Framerate
            easing: Easing function
        
        Returns:
            Transition frames
        """
        num_frames = int(duration * fps)
        if num_frames == 0:
            return np.array([])
        
        frame_a = clip_a_end[-1] if len(clip_a_end) > 0 else None
        frame_b = clip_b_start[0] if len(clip_b_start) > 0 else None
        
        if frame_a is None or frame_b is None:
            return np.array([])
        
        height, width = frame_a.shape[:2]
        
        transition_frames = []
        for i in range(num_frames):
            t = i / num_frames
            t_eased = self.apply_easing(t, easing)
            
            # Create composite frame
            frame = frame_a.copy()
            
            if direction == "left":
                # Wipe from right to left
                wipe_x = int(width * (1 - t_eased))
                frame[:, :wipe_x] = frame_b[:, :wipe_x]
            elif direction == "right":
                # Wipe from left to right
                wipe_x = int(width * t_eased)
                frame[:, :wipe_x] = frame_b[:, :wipe_x]
            elif direction == "up":
                # Wipe from bottom to top
                wipe_y = int(height * (1 - t_eased))
                frame[:wipe_y, :] = frame_b[:wipe_y, :]
            elif direction == "down":
                # Wipe from top to bottom
                wipe_y = int(height * t_eased)
                frame[:wipe_y, :] = frame_b[:wipe_y, :]
            
            transition_frames.append(frame)
        
        self.transition_stats['wipe_count'] += 1
        self.transition_stats['total_transitions'] += 1
        
        return np.array(transition_frames)
    
    # ======================== SMART TRANSITION SELECTION ========================
    
    def choose_transition_for_scenes(
        self,
        scene_a_type: str,
        scene_b_type: str,
        narrative_beat_a: Optional[str] = None,
        narrative_beat_b: Optional[str] = None
    ) -> TransitionType:
        """
        Choose appropriate transition based on scene context
        
        Args:
            scene_a_type: Type of scene A ('temple', 'forest', etc.)
            scene_b_type: Type of scene B
            narrative_beat_a: Narrative beat of scene A
            narrative_beat_b: Narrative beat of scene B
        
        Returns:
            Recommended transition type
        """
        # Same location: Use dissolve or cut
        if scene_a_type == scene_b_type:
            return TransitionType.DISSOLVE
        
        # Dramatic beats: Use fade to black
        dramatic_beats = ['CLIMAX', 'TWIST', 'RESOLUTION']
        if narrative_beat_a in dramatic_beats or narrative_beat_b in dramatic_beats:
            return TransitionType.FADE_BLACK
        
        # Time passage: Use fade to white
        if narrative_beat_a == 'SETUP' and narrative_beat_b == 'RISING_ACTION':
            return TransitionType.FADE_WHITE
        
        # Default: Dissolve for smooth flow
        return TransitionType.DISSOLVE
    
    def apply_transition(
        self,
        clip_a_end: np.ndarray,
        clip_b_start: np.ndarray,
        params: TransitionParams
    ) -> np.ndarray:
        """
        Apply transition with specified parameters
        
        Args:
            clip_a_end: End frames of clip A
            clip_b_start: Start frames of clip B
            params: Transition parameters
        
        Returns:
            Transition frames
        """
        if params.transition_type == TransitionType.CUT:
            self.transition_stats['cut_count'] += 1
            self.transition_stats['total_transitions'] += 1
            return np.array([])  # No transition frames for cut
        
        elif params.transition_type == TransitionType.FADE_BLACK:
            return self.create_fade_transition(
                clip_a_end, clip_b_start,
                params.duration, params.fps,
                (0, 0, 0), params.easing
            )
        
        elif params.transition_type == TransitionType.FADE_WHITE:
            return self.create_fade_transition(
                clip_a_end, clip_b_start,
                params.duration, params.fps,
                (255, 255, 255), params.easing
            )
        
        elif params.transition_type == TransitionType.DISSOLVE:
            return self.create_dissolve_transition(
                clip_a_end, clip_b_start,
                params.duration, params.fps,
                params.easing
            )
        
        elif params.transition_type in [TransitionType.WIPE_LEFT, TransitionType.WIPE_RIGHT,
                                        TransitionType.WIPE_UP, TransitionType.WIPE_DOWN]:
            direction = params.transition_type.value.split('_')[1]
            return self.create_wipe_transition(
                clip_a_end, clip_b_start,
                direction, params.duration, params.fps,
                params.easing
            )
        
        else:
            # Default: dissolve
            return self.create_dissolve_transition(
                clip_a_end, clip_b_start,
                params.duration, params.fps,
                params.easing
            )
    
    # ======================== UTILITY METHODS ========================
    
    def get_stats(self) -> dict:
        """Get transition statistics"""
        return self.transition_stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.transition_stats = {
            'total_transitions': 0,
            'fade_count': 0,
            'dissolve_count': 0,
            'wipe_count': 0,
            'cut_count': 0
        }


# Singleton instance getter
def get_cinematic_transition_core() -> CinematicTransitionCore:
    """Get singleton instance of CinematicTransitionCore"""
    return CinematicTransitionCore()
