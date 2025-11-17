"""
Comprehensive Tests for Emotion Controller (Task 11 Day 4)

Test Coverage:
    1. Emotion state tracking
    2. Motion-emotion coupling
    3. Cross-scene emotional continuity
    4. Micro-expression timing
    5. Emotional arc validation
    6. Singleton pattern
    7. Persistence & export

Author: TTV Studio Team
Created: November 17, 2025
"""

import pytest
from pathlib import Path
import json

from adaptive_engine.emotion_controller import (
    EmotionController,
    get_emotion_controller,
    EmotionType,
    EmotionIntensity,
    EmotionState,
    EmotionTransition,
    MicroExpression,
    MotionParameters
)


# ======================== FIXTURES ========================

@pytest.fixture
def controller():
    """Create a fresh emotion controller for each test"""
    ctrl = get_emotion_controller()
    ctrl.reset()
    return ctrl


@pytest.fixture
def sample_character():
    """Sample character name"""
    return "spiritual_seeker"


# ======================== INITIALIZATION TESTS ========================

def test_emotion_controller_initialization():
    """Test emotion controller initializes correctly"""
    controller = get_emotion_controller()
    
    assert controller is not None
    assert hasattr(controller, 'character_emotions')
    assert hasattr(controller, 'emotion_transitions')
    assert hasattr(controller, 'micro_expressions')
    assert hasattr(controller, 'emotion_motion_map')
    
    # Check emotion-motion map is populated
    assert len(controller.emotion_motion_map) == 12  # 12 emotion types
    assert EmotionType.NEUTRAL in controller.emotion_motion_map
    assert EmotionType.JOY in controller.emotion_motion_map


def test_singleton_pattern():
    """Test that emotion controller follows singleton pattern"""
    controller1 = get_emotion_controller()
    controller2 = get_emotion_controller()
    controller3 = EmotionController()
    
    assert controller1 is controller2
    assert controller2 is controller3


# ======================== EMOTION STATE TRACKING TESTS ========================

def test_set_emotion(controller, sample_character):
    """Test setting emotion for a character"""
    emotion_state = controller.set_emotion(
        character_name=sample_character,
        emotion=EmotionType.NEUTRAL,
        intensity=0.5,
        scene_index=0,
        timestamp=0.0,
        duration=30.0
    )
    
    assert emotion_state is not None
    assert emotion_state.emotion == EmotionType.NEUTRAL
    assert emotion_state.intensity == 0.5
    assert emotion_state.scene_index == 0
    assert emotion_state.timestamp == 0.0
    assert emotion_state.duration == 30.0


def test_set_emotion_clamps_intensity(controller, sample_character):
    """Test that intensity is clamped to [0.0, 1.0]"""
    # Test upper bound
    emotion1 = controller.set_emotion(
        sample_character, EmotionType.JOY, 1.5, 0
    )
    assert emotion1.intensity == 1.0
    
    # Test lower bound
    emotion2 = controller.set_emotion(
        sample_character, EmotionType.SADNESS, -0.5, 1
    )
    assert emotion2.intensity == 0.0


def test_get_current_emotion(controller, sample_character):
    """Test retrieving current emotion for a character"""
    # Set emotions in multiple scenes
    controller.set_emotion(sample_character, EmotionType.NEUTRAL, 0.5, 0)
    controller.set_emotion(sample_character, EmotionType.JOY, 0.7, 1)
    controller.set_emotion(sample_character, EmotionType.CONTEMPLATION, 0.8, 2)
    
    # Get current emotion at scene 1
    current = controller.get_current_emotion(sample_character, 1)
    assert current is not None
    assert current.emotion == EmotionType.JOY
    assert current.intensity == 0.7
    
    # Get current emotion at scene 2
    current = controller.get_current_emotion(sample_character, 2)
    assert current.emotion == EmotionType.CONTEMPLATION
    
    # Get emotion for non-existent character
    current = controller.get_current_emotion("unknown_character", 0)
    assert current is None


def test_get_emotion_history(controller, sample_character):
    """Test retrieving emotion history for a character"""
    # Set multiple emotions
    controller.set_emotion(sample_character, EmotionType.NEUTRAL, 0.5, 0)
    controller.set_emotion(sample_character, EmotionType.JOY, 0.7, 1)
    controller.set_emotion(sample_character, EmotionType.SADNESS, 0.6, 2)
    controller.set_emotion(sample_character, EmotionType.PEACE, 0.8, 3)
    
    # Get full history
    history = controller.get_emotion_history(sample_character)
    assert len(history) == 4
    assert history[0].emotion == EmotionType.NEUTRAL
    assert history[3].emotion == EmotionType.PEACE
    
    # Get history for specific range
    history_range = controller.get_emotion_history(sample_character, scene_range=(1, 2))
    assert len(history_range) == 2
    assert history_range[0].emotion == EmotionType.JOY
    assert history_range[1].emotion == EmotionType.SADNESS


def test_emotion_history_sorting(controller, sample_character):
    """Test that emotion history is sorted by scene and timestamp"""
    # Add emotions out of order
    controller.set_emotion(sample_character, EmotionType.JOY, 0.7, 2, timestamp=5.0)
    controller.set_emotion(sample_character, EmotionType.NEUTRAL, 0.5, 0, timestamp=0.0)
    controller.set_emotion(sample_character, EmotionType.SADNESS, 0.6, 1, timestamp=10.0)
    controller.set_emotion(sample_character, EmotionType.PEACE, 0.8, 2, timestamp=0.0)
    
    history = controller.get_emotion_history(sample_character)
    
    # Should be sorted by scene_index, then timestamp
    assert history[0].scene_index == 0  # Scene 0, timestamp 0
    assert history[1].scene_index == 1  # Scene 1, timestamp 10
    assert history[2].scene_index == 2  # Scene 2, timestamp 0
    assert history[2].emotion == EmotionType.PEACE
    assert history[3].scene_index == 2  # Scene 2, timestamp 5
    assert history[3].emotion == EmotionType.JOY


# ======================== MOTION-EMOTION COUPLING TESTS ========================

def test_get_motion_intensity(controller):
    """Test getting motion intensity for emotions"""
    # Test neutral emotion
    intensity = controller.get_motion_intensity(EmotionType.NEUTRAL, 0.5)
    assert 0.5 <= intensity <= 1.5  # Should be around 1.0 * (0.5 + 0.5*0.5) = 0.75
    
    # Test high-energy emotion (joy)
    intensity_joy = controller.get_motion_intensity(EmotionType.JOY, 1.0)
    # Joy has speed_multiplier = 1.3, so 1.3 * (0.5 + 0.5*1.0) = 1.3
    assert intensity_joy > 1.0
    
    # Test low-energy emotion (sadness)
    intensity_sad = controller.get_motion_intensity(EmotionType.SADNESS, 1.0)
    # Sadness has speed_multiplier = 0.6, so 0.6 * (0.5 + 0.5*1.0) = 0.6
    assert intensity_sad < 1.0


def test_get_gesture_style(controller):
    """Test getting gesture style parameters"""
    # Test joy (high energy, expressive)
    joy_style = controller.get_gesture_style(EmotionType.JOY)
    assert 'amplitude' in joy_style
    assert 'frequency' in joy_style
    assert 'tension' in joy_style
    assert 'smoothness' in joy_style
    assert joy_style['amplitude'] > 0.5  # Joy should be expressive
    
    # Test sadness (low energy, subtle)
    sad_style = controller.get_gesture_style(EmotionType.SADNESS)
    assert sad_style['amplitude'] < 0.5  # Sadness should be subtle
    assert sad_style['frequency'] < 0.5  # Sadness should be slow
    
    # Test anger (high tension)
    anger_style = controller.get_gesture_style(EmotionType.ANGER)
    assert anger_style['tension'] > 0.7  # Anger should be tense


def test_calculate_emotional_motion(controller, sample_character):
    """Test calculating motion parameters from character's emotion"""
    # Set emotion for character
    controller.set_emotion(sample_character, EmotionType.JOY, 0.8, 0)
    
    # Calculate motion parameters
    motion = controller.calculate_emotional_motion(sample_character, 0, 0.0)
    
    assert motion is not None
    assert isinstance(motion, MotionParameters)
    assert motion.speed_multiplier > 1.0  # Joy should increase speed
    assert motion.gesture_amplitude > 0.5  # Joy should have expressive gestures
    assert 0.0 <= motion.body_tension <= 1.0
    
    # Test with no emotion set
    motion_none = controller.calculate_emotional_motion("unknown_character", 0, 0.0)
    assert motion_none is None


def test_motion_parameters_scale_with_intensity(controller, sample_character):
    """Test that motion parameters scale with emotion intensity"""
    # Set low intensity joy
    controller.set_emotion(sample_character, EmotionType.JOY, 0.3, 0)
    motion_low = controller.calculate_emotional_motion(sample_character, 0)
    
    # Set high intensity joy
    controller.set_emotion(sample_character, EmotionType.JOY, 0.9, 1)
    motion_high = controller.calculate_emotional_motion(sample_character, 1)
    
    # High intensity should have larger values
    assert motion_high.gesture_amplitude > motion_low.gesture_amplitude
    assert motion_high.gesture_frequency > motion_low.gesture_frequency


# ======================== CROSS-SCENE EMOTIONAL CONTINUITY TESTS ========================

def test_transition_emotion(controller, sample_character):
    """Test creating emotion transition between scenes"""
    # Set initial emotion
    controller.set_emotion(sample_character, EmotionType.NEUTRAL, 0.5, 0)
    
    # Create transition
    transition = controller.transition_emotion(
        character_name=sample_character,
        from_scene=0,
        to_scene=1,
        to_emotion=EmotionType.JOY,
        to_intensity=0.8,
        transition_frames=15
    )
    
    assert transition is not None
    assert isinstance(transition, EmotionTransition)
    assert transition.from_emotion == EmotionType.NEUTRAL
    assert transition.to_emotion == EmotionType.JOY
    assert transition.from_intensity == 0.5
    assert transition.to_intensity == 0.8
    assert transition.start_scene == 0
    assert transition.end_scene == 1
    assert transition.transition_frames == 15


def test_transition_emotion_defaults_to_neutral(controller, sample_character):
    """Test that transition defaults to neutral if no prior emotion exists"""
    # Create transition without setting initial emotion
    transition = controller.transition_emotion(
        character_name=sample_character,
        from_scene=0,
        to_scene=1,
        to_emotion=EmotionType.JOY,
        to_intensity=0.8
    )
    
    assert transition.from_emotion == EmotionType.NEUTRAL
    assert transition.from_intensity == 0.5


def test_get_transition_frames(controller, sample_character):
    """Test getting interpolated transition frames"""
    # Set up transition
    controller.set_emotion(sample_character, EmotionType.SADNESS, 0.7, 0)
    controller.transition_emotion(
        sample_character, 0, 1, EmotionType.JOY, 0.9, transition_frames=10
    )
    
    # Get transition frames
    frames = controller.get_transition_frames(sample_character, 0, 1)
    
    assert frames is not None
    assert len(frames) == 10
    
    # Each frame should be a tuple of (emotion, intensity)
    for frame in frames:
        assert len(frame) == 2
        emotion, intensity = frame
        assert isinstance(emotion, EmotionType)
        assert 0.0 <= intensity <= 1.0
    
    # First half should be from_emotion, second half should be to_emotion
    first_half = frames[:5]
    second_half = frames[5:]
    
    assert all(emotion == EmotionType.SADNESS for emotion, _ in first_half)
    assert all(emotion == EmotionType.JOY for emotion, _ in second_half)


def test_validate_emotional_arc(controller, sample_character):
    """Test validating emotional arc for continuity"""
    # Create smooth arc
    controller.set_emotion(sample_character, EmotionType.NEUTRAL, 0.5, 0)
    controller.set_emotion(sample_character, EmotionType.JOY, 0.6, 1)
    controller.set_emotion(sample_character, EmotionType.JOY, 0.7, 2)
    controller.set_emotion(sample_character, EmotionType.PEACE, 0.8, 3)
    
    report = controller.validate_emotional_arc(sample_character)
    
    assert 'valid' in report
    assert 'issues' in report
    assert 'recommendations' in report
    assert 'total_emotions' in report
    assert 'unique_emotions' in report
    
    assert report['total_emotions'] == 4
    assert report['unique_emotions'] > 1
    assert report['valid'] == True  # Smooth arc should be valid


def test_validate_emotional_arc_detects_abrupt_changes(controller, sample_character):
    """Test that validation detects abrupt emotional changes"""
    # Create abrupt change
    controller.set_emotion(sample_character, EmotionType.PEACE, 0.9, 0)
    controller.set_emotion(sample_character, EmotionType.ANGER, 0.2, 1)  # Big drop
    
    report = controller.validate_emotional_arc(sample_character)
    
    assert len(report['issues']) > 0
    assert len(report['recommendations']) > 0
    assert 'abrupt' in report['issues'][0].lower() or 'intensity' in report['issues'][0].lower()


def test_validate_emotional_arc_detects_monotony(controller, sample_character):
    """Test that validation detects lack of emotional variety"""
    # Set same emotion for many scenes
    for i in range(5):
        controller.set_emotion(sample_character, EmotionType.NEUTRAL, 0.5, i)
    
    report = controller.validate_emotional_arc(sample_character)
    
    assert report['unique_emotions'] == 1
    # Should suggest more variety
    assert len(report['recommendations']) > 0


# ======================== MICRO-EXPRESSION TIMING TESTS ========================

def test_schedule_micro_expression(controller, sample_character):
    """Test scheduling a micro-expression"""
    micro_expr = controller.schedule_micro_expression(
        character_name=sample_character,
        emotion=EmotionType.SURPRISE,
        intensity=0.4,
        scene_index=0,
        start_frame=10,
        duration_frames=8
    )
    
    assert micro_expr is not None
    assert isinstance(micro_expr, MicroExpression)
    assert micro_expr.emotion == EmotionType.SURPRISE
    assert micro_expr.intensity == 0.4
    assert micro_expr.scene_index == 0
    assert micro_expr.start_frame == 10
    assert micro_expr.end_frame == 18  # start + duration
    assert micro_expr.peak_frame == 14  # start + duration//2


def test_get_expression_keyframes(controller, sample_character):
    """Test retrieving micro-expressions for a scene"""
    # Schedule multiple micro-expressions
    controller.schedule_micro_expression(sample_character, EmotionType.SURPRISE, 0.4, 0, 10, 8)
    controller.schedule_micro_expression(sample_character, EmotionType.CONFUSION, 0.3, 0, 30, 6)
    controller.schedule_micro_expression(sample_character, EmotionType.JOY, 0.5, 1, 5, 10)
    
    # Get keyframes for scene 0
    keyframes_scene0 = controller.get_expression_keyframes(sample_character, 0)
    assert len(keyframes_scene0) == 2
    assert keyframes_scene0[0].emotion == EmotionType.SURPRISE
    assert keyframes_scene0[1].emotion == EmotionType.CONFUSION
    
    # Get keyframes for scene 1
    keyframes_scene1 = controller.get_expression_keyframes(sample_character, 1)
    assert len(keyframes_scene1) == 1
    assert keyframes_scene1[0].emotion == EmotionType.JOY


def test_blend_expressions(controller, sample_character):
    """Test blending base emotion with micro-expression"""
    # Set base emotion
    base_emotion = EmotionState(
        emotion=EmotionType.NEUTRAL,
        intensity=0.5,
        scene_index=0,
        timestamp=0.0,
        duration=100.0
    )
    
    # Create micro-expression
    micro_expr = MicroExpression(
        emotion=EmotionType.SURPRISE,
        intensity=0.8,
        scene_index=0,
        start_frame=20,
        end_frame=30,
        peak_frame=25
    )
    
    # Test before micro-expression
    emotion, intensity = controller.blend_expressions(base_emotion, micro_expr, 15)
    assert emotion == EmotionType.NEUTRAL  # Base emotion
    assert intensity == 0.5  # Base intensity
    
    # Test at peak of micro-expression
    emotion, intensity = controller.blend_expressions(base_emotion, micro_expr, 25)
    assert emotion == EmotionType.SURPRISE  # Micro-expression at peak
    assert intensity > 0.5  # Blended intensity
    
    # Test after micro-expression
    emotion, intensity = controller.blend_expressions(base_emotion, micro_expr, 35)
    assert emotion == EmotionType.NEUTRAL  # Back to base


def test_blend_expressions_bell_curve(controller, sample_character):
    """Test that blending follows bell curve (rise and fall)"""
    base_emotion = EmotionState(EmotionType.NEUTRAL, 0.5, 0, 0.0, 100.0)
    micro_expr = MicroExpression(EmotionType.SURPRISE, 0.8, 0, 10, 20, 15)
    
    # Test rising phase
    _, intensity_early = controller.blend_expressions(base_emotion, micro_expr, 12)
    _, intensity_mid = controller.blend_expressions(base_emotion, micro_expr, 15)
    _, intensity_late = controller.blend_expressions(base_emotion, micro_expr, 18)
    
    # Intensity should rise then fall (bell curve)
    # At mid (peak) should be highest
    assert intensity_mid >= intensity_early
    assert intensity_mid >= intensity_late


# ======================== PERSISTENCE & EXPORT TESTS ========================

def test_export_to_json(controller, sample_character):
    """Test exporting emotion controller state to JSON"""
    # Add some data
    controller.set_emotion(sample_character, EmotionType.JOY, 0.7, 0)
    controller.transition_emotion(sample_character, 0, 1, EmotionType.PEACE, 0.8)
    controller.schedule_micro_expression(sample_character, EmotionType.SURPRISE, 0.4, 0, 10, 8)
    
    # Export to JSON
    output_path = controller.export_to_json("test_emotion_controller.json")
    
    assert Path(output_path).exists()
    
    # Read and validate JSON
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    assert 'character_emotions' in data
    assert 'emotion_transitions' in data
    assert 'micro_expressions' in data
    
    assert sample_character in data['character_emotions']
    assert len(data['character_emotions'][sample_character]) >= 1
    
    # Clean up
    Path(output_path).unlink()


def test_reset(controller, sample_character):
    """Test resetting emotion controller state"""
    # Add data
    controller.set_emotion(sample_character, EmotionType.JOY, 0.7, 0)
    controller.transition_emotion(sample_character, 0, 1, EmotionType.PEACE, 0.8)
    controller.schedule_micro_expression(sample_character, EmotionType.SURPRISE, 0.4, 0, 10, 8)
    
    # Verify data exists
    assert len(controller.character_emotions) > 0
    
    # Reset
    controller.reset()
    
    # Verify data cleared
    assert len(controller.character_emotions) == 0
    assert len(controller.emotion_transitions) == 0
    assert len(controller.micro_expressions) == 0


# ======================== RUN TESTS ========================

if __name__ == "__main__":
    print("Running Emotion Controller Tests...")
    pytest.main([__file__, "-v", "--tb=short"])
