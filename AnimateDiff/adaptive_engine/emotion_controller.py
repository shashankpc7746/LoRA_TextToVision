"""
Emotion Controller - Day 4 of TTV Studio Intelligence Stack

Purpose:
    Tracks character emotions across scenes and couples them with motion parameters
    for emotionally intelligent video generation.

Features:
    1. Emotion State Tracking - Track per-character emotions across scenes
    2. Motion-Emotion Coupling - Map emotions to motion parameters
    3. Cross-Scene Emotional Continuity - Smooth emotion transitions
    4. Micro-Expression Timing - Subtle emotion changes

Author: TTV Studio Team
Created: November 17, 2025
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle
from pathlib import Path


# ======================== EMOTION TYPES ========================

class EmotionType(Enum):
    """Primary emotion categories for character expression"""
    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPLATION = "contemplation"
    DETERMINATION = "determination"
    PEACE = "peace"
    CONFUSION = "confusion"
    AWE = "awe"


class EmotionIntensity(Enum):
    """Intensity levels for emotions"""
    SUBTLE = 0.3      # Barely noticeable
    MILD = 0.5        # Noticeable but controlled
    MODERATE = 0.7    # Clear and evident
    STRONG = 0.9      # Highly expressive
    EXTREME = 1.0     # Maximum expression


# ======================== DATA STRUCTURES ========================

@dataclass
class EmotionState:
    """Represents an emotion state at a specific point in time"""
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    scene_index: int
    timestamp: float  # Frame timestamp within scene
    duration: float   # How long this emotion lasts (in frames)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'emotion': self.emotion.value,
            'intensity': self.intensity,
            'scene_index': self.scene_index,
            'timestamp': self.timestamp,
            'duration': self.duration
        }


@dataclass
class EmotionTransition:
    """Represents a transition between two emotion states"""
    from_emotion: EmotionType
    to_emotion: EmotionType
    from_intensity: float
    to_intensity: float
    start_scene: int
    end_scene: int
    transition_frames: int  # Number of frames for smooth transition
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'from_emotion': self.from_emotion.value,
            'to_emotion': self.to_emotion.value,
            'from_intensity': self.from_intensity,
            'to_intensity': self.to_intensity,
            'start_scene': self.start_scene,
            'end_scene': self.end_scene,
            'transition_frames': self.transition_frames
        }


@dataclass
class MicroExpression:
    """Represents a brief, subtle emotional expression"""
    emotion: EmotionType
    intensity: float
    scene_index: int
    start_frame: int
    end_frame: int
    peak_frame: int  # Frame where expression is most intense
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'emotion': self.emotion.value,
            'intensity': self.intensity,
            'scene_index': self.scene_index,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'peak_frame': self.peak_frame
        }


@dataclass
class MotionParameters:
    """Motion parameters derived from emotional state"""
    speed_multiplier: float      # 0.5 (slow) to 2.0 (fast)
    gesture_amplitude: float     # 0.0 (minimal) to 1.0 (large)
    gesture_frequency: float     # Gestures per second
    body_tension: float          # 0.0 (relaxed) to 1.0 (tense)
    movement_smoothness: float   # 0.0 (jerky) to 1.0 (smooth)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'speed_multiplier': self.speed_multiplier,
            'gesture_amplitude': self.gesture_amplitude,
            'gesture_frequency': self.gesture_frequency,
            'body_tension': self.body_tension,
            'movement_smoothness': self.movement_smoothness
        }


# ======================== MAIN CONTROLLER ========================

class EmotionController:
    """
    Emotion Controller - Manages character emotions across scenes
    
    Features:
        - Emotion state tracking per character
        - Motion-emotion coupling
        - Cross-scene emotional continuity
        - Micro-expression scheduling
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        """Singleton pattern - only one instance exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the Emotion Controller"""
        if self._initialized:
            return
        
        # Character emotion tracking
        self.character_emotions: Dict[str, List[EmotionState]] = {}
        
        # Emotion transitions between scenes
        self.emotion_transitions: Dict[str, List[EmotionTransition]] = {}
        
        # Scheduled micro-expressions
        self.micro_expressions: Dict[str, List[MicroExpression]] = {}
        
        # Cache directory
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Default motion parameters for each emotion type
        self.emotion_motion_map = self._build_emotion_motion_map()
        
        self._initialized = True
        print("🎭 Emotion Controller initialized")
    
    def _build_emotion_motion_map(self) -> Dict[EmotionType, MotionParameters]:
        """Build default motion parameters for each emotion type"""
        return {
            EmotionType.NEUTRAL: MotionParameters(
                speed_multiplier=1.0,
                gesture_amplitude=0.4,
                gesture_frequency=0.5,
                body_tension=0.3,
                movement_smoothness=0.8
            ),
            EmotionType.JOY: MotionParameters(
                speed_multiplier=1.3,
                gesture_amplitude=0.7,
                gesture_frequency=1.0,
                body_tension=0.2,
                movement_smoothness=0.9
            ),
            EmotionType.SADNESS: MotionParameters(
                speed_multiplier=0.6,
                gesture_amplitude=0.3,
                gesture_frequency=0.2,
                body_tension=0.4,
                movement_smoothness=0.6
            ),
            EmotionType.ANGER: MotionParameters(
                speed_multiplier=1.5,
                gesture_amplitude=0.9,
                gesture_frequency=1.2,
                body_tension=0.9,
                movement_smoothness=0.3
            ),
            EmotionType.FEAR: MotionParameters(
                speed_multiplier=1.4,
                gesture_amplitude=0.6,
                gesture_frequency=0.8,
                body_tension=0.8,
                movement_smoothness=0.4
            ),
            EmotionType.SURPRISE: MotionParameters(
                speed_multiplier=1.8,
                gesture_amplitude=0.8,
                gesture_frequency=0.9,
                body_tension=0.6,
                movement_smoothness=0.5
            ),
            EmotionType.DISGUST: MotionParameters(
                speed_multiplier=0.8,
                gesture_amplitude=0.5,
                gesture_frequency=0.4,
                body_tension=0.7,
                movement_smoothness=0.6
            ),
            EmotionType.CONTEMPLATION: MotionParameters(
                speed_multiplier=0.7,
                gesture_amplitude=0.3,
                gesture_frequency=0.3,
                body_tension=0.3,
                movement_smoothness=0.9
            ),
            EmotionType.DETERMINATION: MotionParameters(
                speed_multiplier=1.2,
                gesture_amplitude=0.6,
                gesture_frequency=0.7,
                body_tension=0.6,
                movement_smoothness=0.8
            ),
            EmotionType.PEACE: MotionParameters(
                speed_multiplier=0.8,
                gesture_amplitude=0.4,
                gesture_frequency=0.3,
                body_tension=0.2,
                movement_smoothness=1.0
            ),
            EmotionType.CONFUSION: MotionParameters(
                speed_multiplier=0.9,
                gesture_amplitude=0.5,
                gesture_frequency=0.6,
                body_tension=0.5,
                movement_smoothness=0.5
            ),
            EmotionType.AWE: MotionParameters(
                speed_multiplier=0.6,
                gesture_amplitude=0.6,
                gesture_frequency=0.4,
                body_tension=0.3,
                movement_smoothness=0.9
            )
        }
    
    # ======================== EMOTION STATE TRACKING ========================
    
    def set_emotion(
        self,
        character_name: str,
        emotion: Union[EmotionType, str],
        intensity: float,
        scene_index: int,
        timestamp: float = 0.0,
        duration: float = -1.0
    ) -> EmotionState:
        """
        Set emotion state for a character in a specific scene
        
        Args:
            character_name: Name of the character
            emotion: Emotion type (EmotionType enum or string)
            intensity: Emotion intensity (0.0 to 1.0)
            scene_index: Scene number
            timestamp: Frame timestamp within scene
            duration: How long emotion lasts (-1 for entire scene)
        
        Returns:
            EmotionState object
        """
        # Convert string to EmotionType if needed
        if isinstance(emotion, str):
            emotion_str = emotion.upper()
            try:
                emotion = EmotionType[emotion_str]
            except KeyError:
                # Default to NEUTRAL if emotion not found
                emotion = EmotionType.NEUTRAL
        
        # Clamp intensity to valid range
        intensity = max(0.0, min(1.0, intensity))
        
        # Create emotion state
        emotion_state = EmotionState(
            emotion=emotion,
            intensity=intensity,
            scene_index=scene_index,
            timestamp=timestamp,
            duration=duration
        )
        
        # Store in character's emotion history
        if character_name not in self.character_emotions:
            self.character_emotions[character_name] = []
        
        self.character_emotions[character_name].append(emotion_state)
        
        # Sort by scene_index and timestamp
        self.character_emotions[character_name].sort(
            key=lambda e: (e.scene_index, e.timestamp)
        )
        
        return emotion_state
    
    def get_current_emotion(
        self,
        character_name: str,
        scene_index: int,
        timestamp: float = 0.0
    ) -> Optional[EmotionState]:
        """
        Get current emotion for character at specific scene/timestamp
        
        Args:
            character_name: Name of the character
            scene_index: Scene number
            timestamp: Frame timestamp within scene
        
        Returns:
            EmotionState or None if no emotion set
        """
        if character_name not in self.character_emotions:
            return None
        
        emotions = self.character_emotions[character_name]
        
        # Find most recent emotion at or before this point
        current_emotion = None
        for emotion in emotions:
            if emotion.scene_index > scene_index:
                break
            if emotion.scene_index == scene_index and emotion.timestamp > timestamp:
                break
            current_emotion = emotion
        
        return current_emotion
    
    def get_emotion_history(
        self,
        character_name: str,
        scene_range: Optional[Tuple[int, int]] = None
    ) -> List[EmotionState]:
        """
        Get emotion history for a character
        
        Args:
            character_name: Name of the character
            scene_range: Optional (start_scene, end_scene) range
        
        Returns:
            List of EmotionState objects
        """
        if character_name not in self.character_emotions:
            return []
        
        emotions = self.character_emotions[character_name]
        
        if scene_range is None:
            return emotions.copy()
        
        start_scene, end_scene = scene_range
        return [
            e for e in emotions
            if start_scene <= e.scene_index <= end_scene
        ]
    
    # ======================== MOTION-EMOTION COUPLING ========================
    
    def get_motion_intensity(self, emotion: Union[EmotionType, str], intensity: float) -> float:
        """
        Get motion intensity multiplier for an emotion
        
        Args:
            emotion: Emotion type (EmotionType enum or string)
            intensity: Emotion intensity (0.0 to 1.0)
        
        Returns:
            Motion intensity multiplier
        """
        # Convert string to EmotionType if needed
        if isinstance(emotion, str):
            emotion_str = emotion.upper()
            try:
                emotion = EmotionType[emotion_str]
            except KeyError:
                emotion = EmotionType.NEUTRAL
        
        base_params = self.emotion_motion_map[emotion]
        # Scale speed multiplier by emotion intensity
        return base_params.speed_multiplier * (0.5 + 0.5 * intensity)
    
    def get_gesture_style(self, emotion: EmotionType) -> Dict[str, float]:
        """
        Get gesture style parameters for an emotion
        
        Args:
            emotion: Emotion type
        
        Returns:
            Dictionary with gesture parameters
        """
        params = self.emotion_motion_map[emotion]
        return {
            'amplitude': params.gesture_amplitude,
            'frequency': params.gesture_frequency,
            'tension': params.body_tension,
            'smoothness': params.movement_smoothness
        }
    
    def calculate_emotional_motion(
        self,
        character_name: str,
        scene_index: int,
        timestamp: float = 0.0
    ) -> Optional[MotionParameters]:
        """
        Calculate motion parameters based on character's current emotion
        
        Args:
            character_name: Name of the character
            scene_index: Scene number
            timestamp: Frame timestamp within scene
        
        Returns:
            MotionParameters or None if no emotion set
        """
        emotion_state = self.get_current_emotion(character_name, scene_index, timestamp)
        
        if emotion_state is None:
            return None
        
        # Get base parameters for this emotion
        base_params = self.emotion_motion_map[emotion_state.emotion]
        
        # Scale by intensity
        intensity = emotion_state.intensity
        
        return MotionParameters(
            speed_multiplier=base_params.speed_multiplier * (0.5 + 0.5 * intensity),
            gesture_amplitude=base_params.gesture_amplitude * intensity,
            gesture_frequency=base_params.gesture_frequency * (0.7 + 0.3 * intensity),
            body_tension=base_params.body_tension * intensity,
            movement_smoothness=base_params.movement_smoothness
        )
    
    # ======================== CROSS-SCENE EMOTIONAL CONTINUITY ========================
    
    def transition_emotion(
        self,
        character_name: str,
        from_scene: int,
        to_scene: int,
        to_emotion: EmotionType,
        to_intensity: float,
        transition_frames: int = 15
    ) -> EmotionTransition:
        """
        Create smooth emotion transition between scenes
        
        Args:
            character_name: Name of the character
            from_scene: Starting scene index
            to_scene: Ending scene index
            to_emotion: Target emotion
            to_intensity: Target intensity
            transition_frames: Number of frames for transition
        
        Returns:
            EmotionTransition object
        """
        # Get current emotion in from_scene
        from_emotion_state = self.get_current_emotion(character_name, from_scene)
        
        if from_emotion_state is None:
            # Default to neutral if no prior emotion
            from_emotion = EmotionType.NEUTRAL
            from_intensity = 0.5
        else:
            from_emotion = from_emotion_state.emotion
            from_intensity = from_emotion_state.intensity
        
        # Create transition object
        transition = EmotionTransition(
            from_emotion=from_emotion,
            to_emotion=to_emotion,
            from_intensity=from_intensity,
            to_intensity=to_intensity,
            start_scene=from_scene,
            end_scene=to_scene,
            transition_frames=transition_frames
        )
        
        # Store transition
        if character_name not in self.emotion_transitions:
            self.emotion_transitions[character_name] = []
        
        self.emotion_transitions[character_name].append(transition)
        
        # Set the target emotion in to_scene
        self.set_emotion(character_name, to_emotion, to_intensity, to_scene)
        
        return transition
    
    def get_transition_frames(
        self,
        character_name: str,
        from_scene: int,
        to_scene: int
    ) -> Optional[List[Tuple[EmotionType, float]]]:
        """
        Get interpolated emotion frames for a transition
        
        Args:
            character_name: Name of the character
            from_scene: Starting scene
            to_scene: Ending scene
        
        Returns:
            List of (emotion, intensity) tuples for each frame, or None
        """
        if character_name not in self.emotion_transitions:
            return None
        
        # Find matching transition
        transition = None
        for trans in self.emotion_transitions[character_name]:
            if trans.start_scene == from_scene and trans.end_scene == to_scene:
                transition = trans
                break
        
        if transition is None:
            return None
        
        # Generate interpolated frames
        frames = []
        num_frames = transition.transition_frames
        
        for i in range(num_frames):
            # Linear interpolation
            alpha = i / max(1, num_frames - 1)
            
            # For simplicity, switch emotion at midpoint
            if alpha < 0.5:
                emotion = transition.from_emotion
                intensity = transition.from_intensity * (1 - alpha * 2)
            else:
                emotion = transition.to_emotion
                intensity = transition.to_intensity * ((alpha - 0.5) * 2)
            
            frames.append((emotion, intensity))
        
        return frames
    
    def validate_emotional_arc(self, character_name: str) -> Dict[str, any]:
        """
        Validate emotional arc for character across all scenes
        
        Args:
            character_name: Name of the character
        
        Returns:
            Validation report dictionary
        """
        if character_name not in self.character_emotions:
            return {
                'valid': False,
                'issues': ['No emotions set for character'],
                'recommendations': []
            }
        
        emotions = self.character_emotions[character_name]
        issues = []
        recommendations = []
        
        # Check for abrupt changes
        for i in range(len(emotions) - 1):
            curr = emotions[i]
            next_em = emotions[i + 1]
            
            # Check if scene transition is too abrupt
            if next_em.scene_index == curr.scene_index + 1:
                intensity_change = abs(next_em.intensity - curr.intensity)
                if intensity_change > 0.5:
                    issues.append(
                        f"Abrupt intensity change ({intensity_change:.2f}) "
                        f"between scenes {curr.scene_index} and {next_em.scene_index}"
                    )
                    recommendations.append(
                        f"Add transition between scenes {curr.scene_index} and {next_em.scene_index}"
                    )
        
        # Check for emotion variety
        unique_emotions = set(e.emotion for e in emotions)
        if len(unique_emotions) == 1 and len(emotions) > 3:
            issues.append("Limited emotional range - same emotion throughout")
            recommendations.append("Consider varying emotions to create a richer arc")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'total_emotions': len(emotions),
            'unique_emotions': len(unique_emotions),
            'emotion_variety': list(unique_emotions)
        }
    
    # ======================== MICRO-EXPRESSION TIMING ========================
    
    def schedule_micro_expression(
        self,
        character_name: str,
        emotion: EmotionType,
        intensity: float,
        scene_index: int,
        start_frame: int,
        duration_frames: int = 10
    ) -> MicroExpression:
        """
        Schedule a brief micro-expression within a scene
        
        Args:
            character_name: Name of the character
            emotion: Emotion type for micro-expression
            intensity: Expression intensity (typically 0.3-0.6)
            scene_index: Scene number
            start_frame: Frame where expression starts
            duration_frames: How many frames the expression lasts
        
        Returns:
            MicroExpression object
        """
        peak_frame = start_frame + duration_frames // 2
        end_frame = start_frame + duration_frames
        
        micro_expr = MicroExpression(
            emotion=emotion,
            intensity=intensity,
            scene_index=scene_index,
            start_frame=start_frame,
            end_frame=end_frame,
            peak_frame=peak_frame
        )
        
        # Store micro-expression
        if character_name not in self.micro_expressions:
            self.micro_expressions[character_name] = []
        
        self.micro_expressions[character_name].append(micro_expr)
        
        # Sort by scene and start frame
        self.micro_expressions[character_name].sort(
            key=lambda m: (m.scene_index, m.start_frame)
        )
        
        return micro_expr
    
    def get_expression_keyframes(
        self,
        character_name: str,
        scene_index: int
    ) -> List[MicroExpression]:
        """
        Get all micro-expressions scheduled for a character in a scene
        
        Args:
            character_name: Name of the character
            scene_index: Scene number
        
        Returns:
            List of MicroExpression objects
        """
        if character_name not in self.micro_expressions:
            return []
        
        return [
            expr for expr in self.micro_expressions[character_name]
            if expr.scene_index == scene_index
        ]
    
    def blend_expressions(
        self,
        base_emotion: EmotionState,
        micro_expression: MicroExpression,
        current_frame: int
    ) -> Tuple[EmotionType, float]:
        """
        Blend base emotion with micro-expression based on frame position
        
        Args:
            base_emotion: Base emotion state
            micro_expression: Micro-expression to blend
            current_frame: Current frame number
        
        Returns:
            (emotion, intensity) tuple for blended result
        """
        # Check if we're within micro-expression range
        if not (micro_expression.start_frame <= current_frame <= micro_expression.end_frame):
            return (base_emotion.emotion, base_emotion.intensity)
        
        # Calculate blend factor based on position in micro-expression timeline
        total_frames = micro_expression.end_frame - micro_expression.start_frame
        frame_position = current_frame - micro_expression.start_frame
        
        # Bell curve blend (peak at middle)
        if frame_position <= (total_frames // 2):
            # Rising phase
            blend_factor = frame_position / (total_frames // 2)
        else:
            # Falling phase
            blend_factor = (total_frames - frame_position) / (total_frames // 2)
        
        # Blend intensity
        base_intensity = base_emotion.intensity
        micro_intensity = micro_expression.intensity
        blended_intensity = base_intensity * (1 - blend_factor) + micro_intensity * blend_factor
        
        # Use micro-expression emotion at peak, base otherwise
        if current_frame == micro_expression.peak_frame:
            return (micro_expression.emotion, blended_intensity)
        else:
            return (base_emotion.emotion, blended_intensity)
    
    # ======================== PERSISTENCE & EXPORT ========================
    
    def save_to_cache(self, filename: str = "emotion_controller.pkl"):
        """Save emotion controller state to cache"""
        cache_path = self.cache_dir / filename
        
        data = {
            'character_emotions': {
                char: [e.to_dict() for e in emotions]
                for char, emotions in self.character_emotions.items()
            },
            'emotion_transitions': {
                char: [t.to_dict() for t in transitions]
                for char, transitions in self.emotion_transitions.items()
            },
            'micro_expressions': {
                char: [m.to_dict() for m in expressions]
                for char, expressions in self.micro_expressions.items()
            }
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Emotion controller saved to {cache_path}")
    
    def export_to_json(self, filename: str = "emotion_controller.json") -> str:
        """Export emotion controller state to JSON"""
        output_path = self.cache_dir / filename
        
        data = {
            'character_emotions': {
                char: [e.to_dict() for e in emotions]
                for char, emotions in self.character_emotions.items()
            },
            'emotion_transitions': {
                char: [t.to_dict() for t in transitions]
                for char, transitions in self.emotion_transitions.items()
            },
            'micro_expressions': {
                char: [m.to_dict() for m in expressions]
                for char, expressions in self.micro_expressions.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📄 Emotion controller exported to {output_path}")
        return str(output_path)
    
    def reset(self):
        """Reset emotion controller state"""
        self.character_emotions.clear()
        self.emotion_transitions.clear()
        self.micro_expressions.clear()
        print("🔄 Emotion controller reset")


# ======================== SINGLETON ACCESS ========================

def get_emotion_controller() -> EmotionController:
    """Get the singleton instance of EmotionController"""
    return EmotionController()
