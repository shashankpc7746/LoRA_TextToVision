#!/usr/bin/env python3
"""
Advanced Cinematic Flow Engine
Implements smooth scene transitions, camera movements, and flow control
For professional video production with AnimateDiff + ControlNet
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import resize, crop
import logging

logger = logging.getLogger(__name__)

class CinematicFlowEngine:
    """Advanced cinematic flow control for professional video transitions"""
    
    def __init__(self):
        self.transition_types = {
            'fade': self._create_fade_transition,
            'dissolve': self._create_dissolve_transition,
            'slide': self._create_slide_transition,
            'zoom': self._create_zoom_transition,
            'depth_morph': self._create_depth_morph_transition,
            'scene_blend': self._create_scene_blend_transition
        }
        
        self.camera_movements = {
            'pan_left': self._apply_pan_left,
            'pan_right': self._apply_pan_right,
            'zoom_in': self._apply_zoom_in,
            'zoom_out': self._apply_zoom_out,
            'tilt_up': self._apply_tilt_up,
            'tilt_down': self._apply_tilt_down,
            'orbit': self._apply_orbit_movement,
            'dolly': self._apply_dolly_movement
        }
        
        self.scene_contexts = {
            'temple': {'mood': 'serene', 'lighting': 'warm', 'movement': 'slow'},
            'forest': {'mood': 'mystical', 'lighting': 'dappled', 'movement': 'organic'},
            'cosmic': {'mood': 'ethereal', 'lighting': 'cool', 'movement': 'floating'},
            'mountain': {'mood': 'majestic', 'lighting': 'bright', 'movement': 'steady'},
            'river': {'mood': 'flowing', 'lighting': 'reflective', 'movement': 'fluid'},
            'palace': {'mood': 'grand', 'lighting': 'golden', 'movement': 'regal'}
        }
    
    def create_cinematic_sequence(self, video_clips: List[VideoFileClip], 
                                scene_contexts: List[str],
                                flow_instructions: List[Dict]) -> VideoFileClip:
        """
        Create a cinematic sequence with advanced flow control
        
        Args:
            video_clips: List of video clips to process
            scene_contexts: Scene types for each clip (temple, forest, etc.)
            flow_instructions: Detailed flow control instructions
        
        Returns:
            Final cinematic video with transitions and flow
        """
        
        try:
            logger.info(f"Creating cinematic sequence with {len(video_clips)} clips")
            
            if len(video_clips) == 0:
                raise ValueError("No video clips provided")
            
            if len(video_clips) == 1:
                # Single clip - apply cinematic enhancement
                return self._enhance_single_clip(video_clips[0], scene_contexts[0] if scene_contexts else 'temple')
            
            # Multi-clip sequence with transitions
            enhanced_clips = []
            
            for i, (clip, scene) in enumerate(zip(video_clips, scene_contexts)):
                # Get flow instruction for this clip
                flow_instruction = flow_instructions[i] if i < len(flow_instructions) else {}
                
                # Enhance individual clip
                enhanced_clip = self._enhance_clip_with_flow(clip, scene, flow_instruction, i)
                enhanced_clips.append(enhanced_clip)
            
            # Create transitions between clips
            final_clips = []
            
            for i in range(len(enhanced_clips)):
                final_clips.append(enhanced_clips[i])
                
                # Add transition to next clip (except for last clip)
                if i < len(enhanced_clips) - 1:
                    current_scene = scene_contexts[i]
                    next_scene = scene_contexts[i + 1]
                    
                    transition = self._create_scene_transition(
                        enhanced_clips[i], 
                        enhanced_clips[i + 1],
                        current_scene,
                        next_scene
                    )
                    
                    if transition:
                        final_clips.append(transition)
            
            # Concatenate all clips
            final_video = concatenate_videoclips(final_clips, method="compose")
            
            logger.info(f"Cinematic sequence created: {final_video.duration:.1f}s")
            return final_video
            
        except Exception as e:
            logger.error(f"Cinematic sequence creation failed: {e}")
            # Fallback: return simple concatenation
            return concatenate_videoclips(video_clips, method="compose")
    
    def _enhance_clip_with_flow(self, clip: VideoFileClip, scene: str,
                              flow_instruction: Dict, clip_index: int) -> VideoFileClip:
        """Enhance individual clip with cinematic flow"""

        try:
            # SAFETY: Disable cinematic effects temporarily to fix black screen issue
            logger.info(f"Applying safe cinematic enhancement for clip {clip_index}")

            # For now, return original clip to prevent black screen issues
            # TODO: Re-enable after fixing frame boundary issues
            return clip

            # Get scene context
            context = self.scene_contexts.get(scene, self.scene_contexts['temple'])

            # Apply camera movement based on flow instruction
            movement_type = flow_instruction.get('movement', 'subtle_pan')
            intensity = flow_instruction.get('intensity', 0.2)  # Reduced intensity

            # Validate clip before applying effects
            if not self._validate_clip(clip):
                logger.warning(f"Invalid clip detected, skipping effects")
                return clip

            if movement_type in self.camera_movements:
                enhanced_clip = self.camera_movements[movement_type](clip, intensity, context)

                # Validate enhanced clip
                if not self._validate_clip(enhanced_clip):
                    logger.warning(f"Enhanced clip validation failed, using original")
                    return clip

                return enhanced_clip
            else:
                return clip  # Skip unknown movements

        except Exception as e:
            logger.error(f"Clip enhancement failed: {e}")
            return clip

    def _validate_clip(self, clip: VideoFileClip) -> bool:
        """Validate that clip is not corrupted"""
        try:
            if clip is None:
                return False
            if clip.duration <= 0:
                return False
            # Try to get a frame to ensure clip is valid
            test_frame = clip.get_frame(0.1)
            if test_frame is None or test_frame.size == 0:
                return False
            return True
        except Exception:
            return False
    
    def _apply_default_movement(self, clip: VideoFileClip, scene: str, clip_index: int) -> VideoFileClip:
        """Apply default cinematic movement based on scene and position"""
        
        movement_patterns = {
            'temple': ['pan_right', 'zoom_in', 'tilt_up'],
            'forest': ['pan_left', 'orbit', 'dolly'],
            'cosmic': ['zoom_out', 'orbit', 'pan_right'],
            'mountain': ['pan_right', 'tilt_up', 'zoom_in'],
            'river': ['pan_left', 'dolly', 'pan_right'],
            'palace': ['zoom_in', 'pan_right', 'tilt_up']
        }
        
        patterns = movement_patterns.get(scene, ['pan_right', 'zoom_in', 'pan_left'])
        movement = patterns[clip_index % len(patterns)]
        
        return self.camera_movements[movement](clip, 0.3, self.scene_contexts.get(scene, {}))
    
    def _apply_pan_left(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply smooth left pan movement with safe boundaries"""

        def pan_effect(get_frame, t):
            frame = get_frame(t)
            if frame is None or frame.size == 0:
                return frame

            progress = t / clip.duration
            h, w = frame.shape[:2]

            # Safe pan motion with boundary checks
            max_pan = min(50, w // 10)  # Limit pan distance
            pan_distance = int(max_pan * intensity)
            x_offset = int(pan_distance * progress)

            # Ensure safe boundaries
            x_start = max(0, min(x_offset, w - 100))
            x_end = min(w, x_start + w - pan_distance)

            # Additional safety check
            if x_end <= x_start or x_start < 0 or x_end > w:
                return frame  # Return original frame if bounds are invalid

            return frame[:, x_start:x_end]

        try:
            return clip.fl(pan_effect)
        except Exception as e:
            logger.error(f"Pan left effect failed: {e}")
            return clip  # Return original clip if effect fails
    
    def _apply_pan_right(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply smooth right pan movement"""
        
        def pan_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / clip.duration
            
            # Smooth pan motion (reverse direction)
            pan_distance = int(frame.shape[1] * 0.1 * intensity)
            x_offset = int(pan_distance * (1 - progress))
            
            # Ensure we don't go out of bounds
            x_start = max(0, x_offset)
            x_end = min(frame.shape[1], x_start + frame.shape[1] - pan_distance)
            
            return frame[:, x_start:x_end]
        
        return clip.fl(pan_effect)
    
    def _apply_zoom_in(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply smooth zoom in effect"""
        
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / clip.duration
            
            # Calculate zoom factor
            zoom_factor = 1.0 + (0.2 * intensity * progress)
            
            # Calculate crop area for zoom
            h, w = frame.shape[:2]
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            
            # Center crop
            y_start = (h - new_h) // 2
            x_start = (w - new_w) // 2
            
            cropped = frame[y_start:y_start + new_h, x_start:x_start + new_w]
            
            # Resize back to original size
            return cv2.resize(cropped, (w, h))
        
        return clip.fl(zoom_effect)
    
    def _apply_zoom_out(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply smooth zoom out effect"""
        
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / clip.duration
            
            # Calculate zoom factor (reverse)
            zoom_factor = 1.2 - (0.2 * intensity * progress)
            
            # Calculate crop area for zoom
            h, w = frame.shape[:2]
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            
            # Center crop
            y_start = max(0, (h - new_h) // 2)
            x_start = max(0, (w - new_w) // 2)
            
            if new_h > 0 and new_w > 0:
                cropped = frame[y_start:y_start + new_h, x_start:x_start + new_w]
                return cv2.resize(cropped, (w, h))
            else:
                return frame
        
        return clip.fl(zoom_effect)
    
    def _apply_tilt_up(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply smooth upward tilt movement"""
        
        def tilt_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / clip.duration
            
            # Smooth tilt motion
            tilt_distance = int(frame.shape[0] * 0.1 * intensity)
            y_offset = int(tilt_distance * progress)
            
            # Ensure we don't go out of bounds
            y_start = min(y_offset, frame.shape[0] - 100)
            y_end = min(frame.shape[0], y_start + frame.shape[0] - tilt_distance)
            
            return frame[y_start:y_end, :]
        
        return clip.fl(tilt_effect)
    
    def _apply_tilt_down(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply smooth downward tilt movement"""
        
        def tilt_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / clip.duration
            
            # Smooth tilt motion (reverse direction)
            tilt_distance = int(frame.shape[0] * 0.1 * intensity)
            y_offset = int(tilt_distance * (1 - progress))
            
            # Ensure we don't go out of bounds
            y_start = max(0, y_offset)
            y_end = min(frame.shape[0], y_start + frame.shape[0] - tilt_distance)
            
            return frame[y_start:y_end, :]
        
        return clip.fl(tilt_effect)
    
    def _apply_orbit_movement(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply orbital camera movement"""
        
        def orbit_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / clip.duration
            
            import math
            
            # Orbital motion parameters
            orbit_radius = int(30 * intensity)
            angle = progress * 2 * math.pi
            
            # Calculate position
            x_offset = int(orbit_radius * math.cos(angle)) + orbit_radius
            y_offset = int(orbit_radius * math.sin(angle) * 0.6) + orbit_radius  # Elliptical
            
            # Ensure bounds
            h, w = frame.shape[:2]
            x_start = max(0, min(x_offset, w - 100))
            y_start = max(0, min(y_offset, h - 100))
            x_end = min(w, x_start + w - 2 * orbit_radius)
            y_end = min(h, y_start + h - 2 * orbit_radius)
            
            return frame[y_start:y_end, x_start:x_end]
        
        return clip.fl(orbit_effect)
    
    def _apply_dolly_movement(self, clip: VideoFileClip, intensity: float, context: Dict) -> VideoFileClip:
        """Apply dolly camera movement (forward/backward)"""
        
        def dolly_effect(get_frame, t):
            frame = get_frame(t)
            progress = t / clip.duration
            
            # Dolly motion with perspective change
            scale_factor = 1.0 + (0.15 * intensity * math.sin(progress * math.pi))
            
            h, w = frame.shape[:2]
            new_h, new_w = int(h / scale_factor), int(w / scale_factor)
            
            # Center crop
            y_start = (h - new_h) // 2
            x_start = (w - new_w) // 2
            
            if new_h > 0 and new_w > 0:
                cropped = frame[y_start:y_start + new_h, x_start:x_start + new_w]
                return cv2.resize(cropped, (w, h))
            else:
                return frame
        
        return clip.fl(dolly_effect)
    
    def _apply_scene_enhancement(self, clip: VideoFileClip, context: Dict) -> VideoFileClip:
        """Apply scene-specific visual enhancements"""
        
        # This would apply color grading, lighting adjustments, etc.
        # For now, return the clip as-is
        return clip
    
    def _enhance_single_clip(self, clip: VideoFileClip, scene: str) -> VideoFileClip:
        """Enhance a single clip with cinematic effects"""
        
        context = self.scene_contexts.get(scene, self.scene_contexts['temple'])
        
        # Apply subtle movement for single clips
        enhanced_clip = self._apply_pan_right(clip, 0.2, context)
        enhanced_clip = self._apply_scene_enhancement(enhanced_clip, context)
        
        return enhanced_clip
    
    def _create_scene_transition(self, clip1: VideoFileClip, clip2: VideoFileClip,
                               scene1: str, scene2: str) -> Optional[VideoFileClip]:
        """Create transition between two different scenes"""
        
        try:
            # Determine transition type based on scene change
            transition_type = self._get_transition_type(scene1, scene2)
            
            if transition_type in self.transition_types:
                return self.transition_types[transition_type](clip1, clip2, 0.5)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Scene transition creation failed: {e}")
            return None
    
    def _get_transition_type(self, scene1: str, scene2: str) -> str:
        """Determine appropriate transition type between scenes"""
        
        # Define transition rules
        transition_rules = {
            ('temple', 'forest'): 'dissolve',
            ('forest', 'cosmic'): 'fade',
            ('cosmic', 'temple'): 'depth_morph',
            ('temple', 'mountain'): 'slide',
            ('mountain', 'river'): 'dissolve',
            ('river', 'palace'): 'zoom'
        }
        
        return transition_rules.get((scene1, scene2), 'dissolve')
    
    def _create_dissolve_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                                  duration: float) -> VideoFileClip:
        """Create dissolve transition between clips"""
        
        try:
            # Get last frame of clip1 and first frame of clip2
            last_frame = clip1.get_frame(clip1.duration - 0.1)
            first_frame = clip2.get_frame(0.1)
            
            # Create transition frames
            transition_frames = []
            fps = 8  # Match current FPS
            num_frames = int(duration * fps)
            
            for i in range(num_frames):
                alpha = i / (num_frames - 1)
                blended_frame = cv2.addWeighted(last_frame, 1 - alpha, first_frame, alpha, 0)
                transition_frames.append(blended_frame)
            
            # Create transition clip
            def make_frame(t):
                frame_index = min(int(t * fps), len(transition_frames) - 1)
                return transition_frames[frame_index]
            
            transition_clip = VideoFileClip(make_frame, duration=duration)
            return transition_clip
            
        except Exception as e:
            logger.error(f"Dissolve transition creation failed: {e}")
            return None
    
    def _create_fade_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                              duration: float) -> VideoFileClip:
        """Create fade transition (fade to black, then fade in)"""
        
        # For now, return None to skip fade transitions (as per user preference)
        return None
    
    def _create_slide_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                               duration: float) -> VideoFileClip:
        """Create slide transition between clips"""
        
        # Implementation would create sliding effect
        return None
    
    def _create_zoom_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                              duration: float) -> VideoFileClip:
        """Create zoom transition between clips"""
        
        # Implementation would create zoom effect
        return None
    
    def _create_depth_morph_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                                     duration: float) -> VideoFileClip:
        """Create depth-based morphing transition"""
        
        # Implementation would use depth maps for morphing
        return None
    
    def _create_scene_blend_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                                     duration: float) -> VideoFileClip:
        """Create scene blending transition"""
        
        # Implementation would blend scenes based on content
        return None

if __name__ == "__main__":
    # Test cinematic flow engine
    engine = CinematicFlowEngine()
    
    print("🎬 Cinematic Flow Engine initialized")
    print(f"Available movements: {list(engine.camera_movements.keys())}")
    print(f"Available transitions: {list(engine.transition_types.keys())}")
    print(f"Scene contexts: {list(engine.scene_contexts.keys())}")
