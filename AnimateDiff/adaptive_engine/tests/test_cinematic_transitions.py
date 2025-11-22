"""
Tests for Cinematic Transition Core - Day 5
"""

import pytest
import numpy as np
import sys
import os

# Add AnimateDiff to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from adaptive_engine import (
    get_cinematic_transition_core,
    CinematicTransitionCore,
    TransitionType,
    TransitionParams
)


@pytest.fixture
def transition_core():
    """Get fresh transition core instance"""
    core = get_cinematic_transition_core()
    core.reset_stats()
    return core


@pytest.fixture
def sample_clips():
    """Create sample video clips"""
    # Create two 12-frame clips (0.5s at 24fps)
    clip_a = np.random.randint(50, 100, (12, 720, 1280, 3), dtype=np.uint8)
    clip_b = np.random.randint(150, 200, (12, 720, 1280, 3), dtype=np.uint8)
    return clip_a, clip_b


def test_singleton_pattern():
    """Test that CinematicTransitionCore is a singleton"""
    core1 = get_cinematic_transition_core()
    core2 = get_cinematic_transition_core()
    assert core1 is core2


def test_fade_black_transition(transition_core, sample_clips):
    """Test fade to black transition"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_fade_transition(
        clip_a, clip_b,
        duration=0.5, fps=24.0,
        fade_color=(0, 0, 0)
    )
    
    # Should have frames for 0.5 seconds at 24fps
    expected_frames = int(0.5 * 24.0)
    assert len(transition_frames) == expected_frames
    
    # Stats should update
    stats = transition_core.get_stats()
    assert stats['fade_count'] == 1


def test_fade_white_transition(transition_core, sample_clips):
    """Test fade to white transition"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_fade_transition(
        clip_a, clip_b,
        duration=0.5, fps=24.0,
        fade_color=(255, 255, 255)
    )
    
    assert len(transition_frames) > 0
    
    # Middle frame should be mostly white
    mid_frame = transition_frames[len(transition_frames) // 2]
    avg_brightness = np.mean(mid_frame)
    assert avg_brightness > 200  # Should be close to white


def test_dissolve_transition(transition_core, sample_clips):
    """Test cross-dissolve transition"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_dissolve_transition(
        clip_a, clip_b,
        duration=0.5, fps=24.0
    )
    
    assert len(transition_frames) == int(0.5 * 24.0)
    
    # Mid frame should be a blend (not identical to either source)
    mid_frame = transition_frames[len(transition_frames) // 2]
    
    # Check that mid frame is a blend (not identical to either source)
    assert not np.array_equal(mid_frame, clip_a[-1])
    assert not np.array_equal(mid_frame, clip_b[0])
    
    stats = transition_core.get_stats()
    assert stats['dissolve_count'] == 1


def test_wipe_left_transition(transition_core, sample_clips):
    """Test wipe left transition"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_wipe_transition(
        clip_a, clip_b,
        direction="left", duration=0.5, fps=24.0
    )
    
    assert len(transition_frames) > 0
    
    # First frame should have B on right, A on left
    # Last frame should be mostly B
    first_frame = transition_frames[0]
    last_frame = transition_frames[-1]
    
    # Frames should be different
    assert not np.array_equal(first_frame, last_frame)
    
    stats = transition_core.get_stats()
    assert stats['wipe_count'] == 1


def test_wipe_right_transition(transition_core, sample_clips):
    """Test wipe right transition"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_wipe_transition(
        clip_a, clip_b,
        direction="right", duration=0.5, fps=24.0
    )
    
    assert len(transition_frames) == int(0.5 * 24.0)


def test_wipe_up_transition(transition_core, sample_clips):
    """Test wipe up transition"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_wipe_transition(
        clip_a, clip_b,
        direction="up", duration=0.5, fps=24.0
    )
    
    assert len(transition_frames) > 0


def test_wipe_down_transition(transition_core, sample_clips):
    """Test wipe down transition"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_wipe_transition(
        clip_a, clip_b,
        direction="down", duration=0.5, fps=24.0
    )
    
    assert len(transition_frames) > 0


def test_easing_functions(transition_core):
    """Test easing functions"""
    # Linear
    assert transition_core.apply_easing(0.5, "linear") == 0.5
    
    # Ease in (should be < 0.5 at t=0.5)
    assert transition_core.apply_easing(0.5, "ease_in") < 0.5
    
    # Ease out (should be > 0.5 at t=0.5)
    assert transition_core.apply_easing(0.5, "ease_out") > 0.5
    
    # Boundaries
    assert transition_core.apply_easing(0.0, "linear") == 0.0
    assert transition_core.apply_easing(1.0, "linear") == 1.0


def test_transition_with_easing(transition_core, sample_clips):
    """Test transition with different easing"""
    clip_a, clip_b = sample_clips
    
    # Linear easing
    linear_frames = transition_core.create_dissolve_transition(
        clip_a, clip_b, duration=0.5, fps=24.0, easing="linear"
    )
    
    # Ease in/out
    eased_frames = transition_core.create_dissolve_transition(
        clip_a, clip_b, duration=0.5, fps=24.0, easing="ease_in_out"
    )
    
    # Both should have same length
    assert len(linear_frames) == len(eased_frames)
    
    # But should be different (due to easing)
    assert not np.array_equal(linear_frames[5], eased_frames[5])


def test_choose_transition_same_location(transition_core):
    """Test transition selection for same location"""
    transition = transition_core.choose_transition_for_scenes(
        scene_a_type="temple",
        scene_b_type="temple"
    )
    
    # Same location should use dissolve
    assert transition == TransitionType.DISSOLVE


def test_choose_transition_dramatic_beat(transition_core):
    """Test transition selection for dramatic beats"""
    transition = transition_core.choose_transition_for_scenes(
        scene_a_type="temple",
        scene_b_type="forest",
        narrative_beat_a="CLIMAX"
    )
    
    # Dramatic beat should use fade to black
    assert transition == TransitionType.FADE_BLACK


def test_choose_transition_time_passage(transition_core):
    """Test transition selection for time passage"""
    transition = transition_core.choose_transition_for_scenes(
        scene_a_type="temple",
        scene_b_type="forest",
        narrative_beat_a="SETUP",
        narrative_beat_b="RISING_ACTION"
    )
    
    # Time passage should use fade to white
    assert transition == TransitionType.FADE_WHITE


def test_choose_transition_default(transition_core):
    """Test default transition selection"""
    transition = transition_core.choose_transition_for_scenes(
        scene_a_type="temple",
        scene_b_type="forest"
    )
    
    # Default should be dissolve
    assert transition == TransitionType.DISSOLVE


def test_apply_transition_cut(transition_core, sample_clips):
    """Test applying cut transition (no frames)"""
    clip_a, clip_b = sample_clips
    
    params = TransitionParams(
        transition_type=TransitionType.CUT,
        duration=0.0, fps=24.0
    )
    
    transition_frames = transition_core.apply_transition(
        clip_a, clip_b, params
    )
    
    # Cut should have no transition frames
    assert len(transition_frames) == 0
    
    stats = transition_core.get_stats()
    assert stats['cut_count'] == 1


def test_apply_transition_fade_black(transition_core, sample_clips):
    """Test applying fade to black via apply_transition"""
    clip_a, clip_b = sample_clips
    
    params = TransitionParams(
        transition_type=TransitionType.FADE_BLACK,
        duration=0.5, fps=24.0
    )
    
    transition_frames = transition_core.apply_transition(
        clip_a, clip_b, params
    )
    
    assert len(transition_frames) > 0
    stats = transition_core.get_stats()
    assert stats['fade_count'] == 1


def test_apply_transition_dissolve(transition_core, sample_clips):
    """Test applying dissolve via apply_transition"""
    clip_a, clip_b = sample_clips
    
    params = TransitionParams(
        transition_type=TransitionType.DISSOLVE,
        duration=0.5, fps=24.0, easing="ease_in_out"
    )
    
    transition_frames = transition_core.apply_transition(
        clip_a, clip_b, params
    )
    
    assert len(transition_frames) > 0
    stats = transition_core.get_stats()
    assert stats['dissolve_count'] == 1


def test_apply_transition_wipe(transition_core, sample_clips):
    """Test applying wipe transitions via apply_transition"""
    clip_a, clip_b = sample_clips
    
    for wipe_type in [TransitionType.WIPE_LEFT, TransitionType.WIPE_RIGHT,
                      TransitionType.WIPE_UP, TransitionType.WIPE_DOWN]:
        params = TransitionParams(
            transition_type=wipe_type,
            duration=0.5, fps=24.0
        )
        
        transition_frames = transition_core.apply_transition(
            clip_a, clip_b, params
        )
        
        assert len(transition_frames) > 0


def test_empty_clips(transition_core):
    """Test handling of empty clips"""
    empty_clip = np.array([])
    sample_clip = np.random.randint(0, 255, (12, 720, 1280, 3), dtype=np.uint8)
    
    # Empty clip A
    frames = transition_core.create_dissolve_transition(
        empty_clip, sample_clip, duration=0.5, fps=24.0
    )
    assert len(frames) == 0
    
    # Empty clip B
    frames = transition_core.create_dissolve_transition(
        sample_clip, empty_clip, duration=0.5, fps=24.0
    )
    assert len(frames) == 0


def test_stats_tracking(transition_core, sample_clips):
    """Test statistics tracking"""
    clip_a, clip_b = sample_clips
    
    transition_core.reset_stats()
    
    # Create various transitions
    transition_core.create_fade_transition(clip_a, clip_b, 0.5, 24.0, (0, 0, 0))
    transition_core.create_fade_transition(clip_a, clip_b, 0.5, 24.0, (255, 255, 255))
    transition_core.create_dissolve_transition(clip_a, clip_b, 0.5, 24.0)
    transition_core.create_wipe_transition(clip_a, clip_b, "left", 0.5, 24.0)
    
    stats = transition_core.get_stats()
    
    assert stats['fade_count'] == 2
    assert stats['dissolve_count'] == 1
    assert stats['wipe_count'] == 1
    assert stats['total_transitions'] == 4


def test_zero_duration_transition(transition_core, sample_clips):
    """Test transition with zero duration"""
    clip_a, clip_b = sample_clips
    
    transition_frames = transition_core.create_dissolve_transition(
        clip_a, clip_b, duration=0.0, fps=24.0
    )
    
    # Should have no frames
    assert len(transition_frames) == 0


def test_production_scenario_scene_change(transition_core):
    """Test realistic production scenario: temple → forest transition"""
    # Create clips with distinct visual characteristics
    temple_clip = np.random.randint(100, 150, (12, 720, 1280, 3), dtype=np.uint8)
    forest_clip = np.random.randint(50, 100, (12, 720, 1280, 3), dtype=np.uint8)
    
    # Choose transition
    transition_type = transition_core.choose_transition_for_scenes(
        scene_a_type="temple",
        scene_b_type="forest",
        narrative_beat_a="RISING_ACTION",
        narrative_beat_b="CLIMAX"
    )
    
    # Apply transition
    params = TransitionParams(
        transition_type=transition_type,
        duration=0.5, fps=24.0, easing="ease_in_out"
    )
    
    transition_frames = transition_core.apply_transition(
        temple_clip, forest_clip, params
    )
    
    # Should have smooth transition
    assert len(transition_frames) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
