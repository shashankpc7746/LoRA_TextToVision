"""
Tests for Smart Video Extender - Day 5
"""

import pytest
import numpy as np
import sys
import os

# Add AnimateDiff to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from adaptive_engine import (
    get_smart_video_extender,
    SmartVideoExtender,
    ExtensionMethod,
    ExtensionParams
)


@pytest.fixture
def video_extender():
    """Get fresh video extender instance"""
    extender = get_smart_video_extender()
    extender.reset_stats()
    return extender


@pytest.fixture
def sample_frames():
    """Create sample video frames (24 frames, 1 second at 24fps)"""
    # Create 24 frames of 720x1280 RGB video
    frames = np.random.randint(0, 255, (24, 720, 1280, 3), dtype=np.uint8)
    return frames


def test_singleton_pattern():
    """Test that SmartVideoExtender is a singleton"""
    extender1 = get_smart_video_extender()
    extender2 = get_smart_video_extender()
    assert extender1 is extender2


def test_slow_motion_basic(video_extender, sample_frames):
    """Test basic slow motion extension"""
    extended, new_fps = video_extender.apply_slow_motion(
        sample_frames, slow_factor=1.5, fps=24.0
    )
    
    # Should have 1.5x more frames
    assert len(extended) == int(len(sample_frames) * 1.5)
    assert new_fps == 24.0 / 1.5
    
    # Stats should update
    stats = video_extender.get_stats()
    assert stats['slow_motion_count'] == 1


def test_slow_motion_with_blend(video_extender, sample_frames):
    """Test slow motion with frame blending"""
    extended, new_fps = video_extender.apply_slow_motion_blend(
        sample_frames, slow_factor=1.5, fps=24.0, blend_weight=0.3
    )
    
    # Should have extended frames
    assert len(extended) > len(sample_frames)
    
    # Stats should show both slow motion and blend
    stats = video_extender.get_stats()
    assert stats['slow_motion_count'] >= 1
    assert stats['blend_count'] == 1


def test_smart_freeze(video_extender, sample_frames):
    """Test smart freeze with zoom"""
    freeze_duration = 2.0  # 2 seconds
    fps = 24.0
    
    extended = video_extender.apply_smart_freeze(
        sample_frames, freeze_duration, fps,
        zoom_amount=0.1, zoom_speed=0.02
    )
    
    # Should have original + freeze frames
    expected_freeze_frames = int(freeze_duration * fps)
    assert len(extended) == len(sample_frames) + expected_freeze_frames
    
    # Stats should update
    stats = video_extender.get_stats()
    assert stats['freeze_count'] == 1


def test_smart_freeze_zoom_effect(video_extender, sample_frames):
    """Test that zoom effect is actually applied"""
    extended = video_extender.apply_smart_freeze(
        sample_frames, freeze_duration=1.0, fps=24.0,
        zoom_amount=0.2, zoom_speed=0.05
    )
    
    # Get last frame of original and a later freeze frame (not first, which has minimal zoom)
    last_original = sample_frames[-1]
    mid_freeze = extended[len(sample_frames) + 10]  # 10 frames into freeze
    
    # Frames should be different due to zoom
    assert not np.array_equal(last_original, mid_freeze)


def test_extend_to_duration_slow_only(video_extender, sample_frames):
    """Test extension using slow motion only"""
    current_duration = 1.0  # 24 frames at 24fps
    target_duration = 1.4   # 40% increase - can be done with slow motion
    
    extended, new_fps = video_extender.extend_to_duration(
        sample_frames, current_duration, target_duration,
        fps=24.0, method=ExtensionMethod.SLOW_MOTION
    )
    
    # Should be extended
    assert len(extended) > len(sample_frames)


def test_extend_to_duration_combined(video_extender, sample_frames):
    """Test extension using combined method"""
    current_duration = 1.0
    target_duration = 3.0  # 3x - needs both slow motion and freeze
    
    extended, new_fps = video_extender.extend_to_duration(
        sample_frames, current_duration, target_duration,
        fps=24.0, method=ExtensionMethod.COMBINED
    )
    
    # Should use both methods
    stats = video_extender.get_stats()
    assert stats['slow_motion_count'] >= 1
    assert stats['freeze_count'] >= 1


def test_extend_to_duration_freeze_only(video_extender, sample_frames):
    """Test extension using freeze only"""
    current_duration = 1.0
    target_duration = 2.0
    
    extended, new_fps = video_extender.extend_to_duration(
        sample_frames, current_duration, target_duration,
        fps=24.0, method=ExtensionMethod.SMART_FREEZE
    )
    
    # Should only use freeze
    stats = video_extender.get_stats()
    assert stats['freeze_count'] == 1


def test_extension_strategy_calculation(video_extender):
    """Test extension strategy calculator"""
    # Test 1: Small extension (slow motion only)
    strategy = video_extender.calculate_extension_strategy(
        current_duration=2.0,
        target_duration=2.8  # 40% increase
    )
    
    assert strategy['recommended_method'] == ExtensionMethod.SLOW_MOTION
    assert strategy['extension_ratio'] == 1.4
    
    # Test 2: Large extension (combined)
    strategy = video_extender.calculate_extension_strategy(
        current_duration=2.0,
        target_duration=4.5  # 125% increase
    )
    
    assert strategy['recommended_method'] == ExtensionMethod.COMBINED
    assert strategy['freeze_duration'] > 0


def test_no_extension_needed(video_extender, sample_frames):
    """Test that no extension occurs when target <= current"""
    current_duration = 2.0
    target_duration = 1.5  # Less than current
    
    extended, new_fps = video_extender.extend_to_duration(
        sample_frames, current_duration, target_duration, fps=24.0
    )
    
    # Should return original
    assert len(extended) == len(sample_frames)
    assert new_fps == 24.0


def test_empty_frames(video_extender):
    """Test handling of empty frame arrays"""
    empty_frames = np.array([])
    
    # Slow motion
    extended, fps = video_extender.apply_slow_motion(empty_frames, 1.5, 24.0)
    assert len(extended) == 0
    
    # Freeze
    extended = video_extender.apply_smart_freeze(empty_frames, 1.0, 24.0)
    assert len(extended) == 0


def test_different_slow_factors(video_extender, sample_frames):
    """Test different slow motion factors"""
    factors = [1.2, 1.5, 2.0]
    
    for factor in factors:
        extended, new_fps = video_extender.apply_slow_motion(
            sample_frames, slow_factor=factor, fps=24.0
        )
        
        # Check frame count
        expected_frames = int(len(sample_frames) * factor)
        assert len(extended) == expected_frames
        
        # Check fps
        assert new_fps == pytest.approx(24.0 / factor, rel=0.01)


def test_stats_tracking(video_extender, sample_frames):
    """Test that statistics are tracked correctly"""
    # Reset stats
    video_extender.reset_stats()
    
    # Perform various operations
    video_extender.apply_slow_motion(sample_frames, 1.5, 24.0)
    video_extender.apply_smart_freeze(sample_frames, 1.0, 24.0)
    video_extender.apply_slow_motion_blend(sample_frames, 1.5, 24.0)
    
    stats = video_extender.get_stats()
    
    assert stats['slow_motion_count'] >= 2  # apply_slow_motion + apply_slow_motion_blend
    assert stats['freeze_count'] == 1
    assert stats['blend_count'] == 1
    assert stats['total_extended'] >= 3


def test_production_scenario_short_clip(video_extender):
    """Test realistic production scenario: 2s clip → 6s audio"""
    # 2-second clip (48 frames at 24fps)
    short_clip = np.random.randint(0, 255, (48, 720, 1280, 3), dtype=np.uint8)
    
    current_duration = 2.0
    target_duration = 6.0  # 3x extension
    
    extended, new_fps = video_extender.extend_to_duration(
        short_clip, current_duration, target_duration,
        fps=24.0, method=ExtensionMethod.COMBINED
    )
    
    # Should use combined method
    stats = video_extender.get_stats()
    assert stats['slow_motion_count'] >= 1
    assert stats['freeze_count'] >= 1
    
    # Result should be significantly longer
    assert len(extended) > len(short_clip) * 2


def test_production_scenario_medium_clip(video_extender):
    """Test realistic production scenario: 3s clip → 4.5s audio"""
    # 3-second clip
    medium_clip = np.random.randint(0, 255, (72, 720, 1280, 3), dtype=np.uint8)
    
    current_duration = 3.0
    target_duration = 4.5  # 50% extension
    
    extended, new_fps = video_extender.extend_to_duration(
        medium_clip, current_duration, target_duration,
        fps=24.0, method=ExtensionMethod.COMBINED
    )
    
    # Should primarily use slow motion
    assert len(extended) > len(medium_clip)


def test_frame_quality_preservation(video_extender, sample_frames):
    """Test that frame quality is preserved during extension"""
    extended, _ = video_extender.apply_slow_motion(sample_frames, 1.5, 24.0)
    
    # All extended frames should be valid
    assert extended.dtype == np.uint8
    assert extended.shape[1:] == sample_frames.shape[1:]  # Same height, width, channels
    
    # Values should be in valid range
    assert np.all(extended >= 0)
    assert np.all(extended <= 255)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
