"""
Visual Testing Script for Day 5: Smart Video Extension + Transitions

This script creates actual video files demonstrating:
1. Video looping (BEFORE - the problem)
2. Smart extension (AFTER - the solution)
3. Cinematic transitions

Run this to visually verify Day 5 features!
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add AnimateDiff to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_engine import (
    get_smart_video_extender,
    get_cinematic_transition_core,
    ExtensionMethod,
    TransitionType,
    TransitionParams
)


def create_test_clip(duration_sec=2.0, fps=24, text="Test Clip", color=(100, 150, 200)):
    """Create a simple test video clip with moving text"""
    width, height = 1280, 720
    num_frames = int(duration_sec * fps)
    frames = []
    
    for i in range(num_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add animated gradient
        gradient_offset = int((i / num_frames) * width)
        for x in range(width):
            color_val = int(((x + gradient_offset) % width) / width * 255)
            frame[:, x] = [color[0], color[1], color_val]
        
        # Add text
        text_position = (50 + (i * 5) % 500, height // 2)
        cv2.putText(frame, text, text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Add frame number
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return np.array(frames), fps


def save_video(frames, fps, output_path, description=""):
    """Save frames as video file using imageio (more reliable)"""
    import imageio
    
    # Convert BGR (OpenCV) to RGB (imageio)
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    # Save with imageio-ffmpeg (more compatible)
    imageio.mimsave(str(output_path), frames_rgb, fps=fps, codec='libx264', quality=8)
    
    print(f"✅ Saved: {output_path} ({len(frames)} frames @ {fps}fps, {len(frames)/fps:.1f}s)")
    if description:
        print(f"   📝 {description}")


def test_1_video_looping_problem():
    """Demonstrate the video looping problem (BEFORE)"""
    print("\n" + "="*70)
    print("TEST 1: VIDEO LOOPING PROBLEM (BEFORE)")
    print("="*70)
    
    # Create short 2-second clip
    short_clip, fps = create_test_clip(duration_sec=2.0, fps=24, 
                                       text="Short Clip", color=(100, 100, 200))
    
    # Need 6 seconds of video (e.g., to match audio)
    # OLD METHOD: Just loop the clip 3 times
    looped_clip = np.concatenate([short_clip, short_clip, short_clip], axis=0)
    
    output_dir = Path("outputs/day5_visual_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_video(short_clip, fps, output_dir / "1a_original_short_clip.mp4",
              "Original 2-second clip")
    save_video(looped_clip, fps, output_dir / "1b_looped_repetitive.mp4",
              "❌ PROBLEM: Same clip looped 3x (repetitive!)")
    
    return output_dir


def test_2_smart_extension_solution():
    """Demonstrate smart extension solution (AFTER)"""
    print("\n" + "="*70)
    print("TEST 2: SMART EXTENSION SOLUTION (AFTER)")
    print("="*70)
    
    extender = get_smart_video_extender()
    
    # Create same short clip
    short_clip, fps = create_test_clip(duration_sec=2.0, fps=24,
                                       text="Smart Extension", color=(100, 200, 100))
    
    output_dir = Path("outputs/day5_visual_tests")
    
    # Method 1: Slow motion only
    print("\n📹 Method 1: Slow Motion Extension")
    slow_clip, new_fps = extender.apply_slow_motion(short_clip, slow_factor=1.5, fps=fps)
    save_video(slow_clip, new_fps, output_dir / "2a_slow_motion.mp4",
              "✅ Slow motion: 2s → 3s (24fps → 16fps)")
    
    # Method 2: Smart freeze with zoom
    print("\n📹 Method 2: Smart Freeze with Zoom")
    freeze_clip = extender.apply_smart_freeze(short_clip, freeze_duration=2.0, fps=fps,
                                             zoom_amount=0.15, zoom_speed=0.03)
    save_video(freeze_clip, fps, output_dir / "2b_smart_freeze.mp4",
              "✅ Smart freeze: 2s original + 2s freeze with zoom = 4s")
    
    # Method 3: Combined (slow motion + freeze)
    print("\n📹 Method 3: Combined Extension (SlowMo + Freeze)")
    combined_clip, combined_fps = extender.extend_to_duration(
        short_clip, 
        current_duration=2.0,
        target_duration=6.0,
        fps=fps,
        method=ExtensionMethod.COMBINED
    )
    save_video(combined_clip, combined_fps, output_dir / "2c_combined_extension.mp4",
              "✅ SOLUTION: SlowMo(2s→3s) + Freeze(3s) = Natural 6s!")
    
    # Show statistics
    stats = extender.get_stats()
    print(f"\n📊 Extension Statistics:")
    print(f"   Total extensions: {stats['total_extended']}")
    print(f"   Slow motion count: {stats['slow_motion_count']}")
    print(f"   Freeze count: {stats['freeze_count']}")
    print(f"   Blend count: {stats['blend_count']}")


def test_3_cinematic_transitions():
    """Demonstrate cinematic transitions"""
    print("\n" + "="*70)
    print("TEST 3: CINEMATIC TRANSITIONS")
    print("="*70)
    
    transition_core = get_cinematic_transition_core()
    
    # Create two different clips
    clip_temple, fps = create_test_clip(duration_sec=1.0, fps=24,
                                        text="Temple Scene", color=(150, 100, 50))
    clip_forest, fps = create_test_clip(duration_sec=1.0, fps=24,
                                        text="Forest Scene", color=(50, 150, 50))
    
    output_dir = Path("outputs/day5_visual_tests")
    
    # 1. Fade to Black
    print("\n📹 Transition 1: Fade to Black")
    fade_black = transition_core.create_fade_transition(
        clip_temple, clip_forest, duration=0.5, fps=fps,
        fade_color=(0, 0, 0), easing="ease_in_out"
    )
    combined = np.concatenate([clip_temple, fade_black, clip_forest], axis=0)
    save_video(combined, fps, output_dir / "3a_fade_to_black.mp4",
              "Temple → [Fade Black] → Forest")
    
    # 2. Fade to White
    print("\n📹 Transition 2: Fade to White")
    fade_white = transition_core.create_fade_transition(
        clip_temple, clip_forest, duration=0.5, fps=fps,
        fade_color=(255, 255, 255), easing="ease_in_out"
    )
    combined = np.concatenate([clip_temple, fade_white, clip_forest], axis=0)
    save_video(combined, fps, output_dir / "3b_fade_to_white.mp4",
              "Temple → [Fade White] → Forest")
    
    # 3. Cross-Dissolve
    print("\n📹 Transition 3: Cross-Dissolve")
    dissolve = transition_core.create_dissolve_transition(
        clip_temple, clip_forest, duration=0.5, fps=fps, easing="linear"
    )
    combined = np.concatenate([clip_temple, dissolve, clip_forest], axis=0)
    save_video(combined, fps, output_dir / "3c_dissolve.mp4",
              "Temple → [Dissolve Blend] → Forest")
    
    # 4. Wipe Left
    print("\n📹 Transition 4: Wipe Left")
    wipe_left = transition_core.create_wipe_transition(
        clip_temple, clip_forest, direction="left", duration=0.5, fps=fps
    )
    combined = np.concatenate([clip_temple, wipe_left, clip_forest], axis=0)
    save_video(combined, fps, output_dir / "3d_wipe_left.mp4",
              "Temple → [Wipe Left] → Forest")
    
    # 5. Wipe Right
    print("\n📹 Transition 5: Wipe Right")
    wipe_right = transition_core.create_wipe_transition(
        clip_temple, clip_forest, direction="right", duration=0.5, fps=fps
    )
    combined = np.concatenate([clip_temple, wipe_right, clip_forest], axis=0)
    save_video(combined, fps, output_dir / "3e_wipe_right.mp4",
              "Temple → [Wipe Right] → Forest")
    
    # Show statistics
    stats = transition_core.get_stats()
    print(f"\n📊 Transition Statistics:")
    print(f"   Total transitions: {stats['total_transitions']}")
    print(f"   Fade count: {stats['fade_count']}")
    print(f"   Dissolve count: {stats['dissolve_count']}")
    print(f"   Wipe count: {stats['wipe_count']}")


def test_4_production_scenario():
    """Demonstrate realistic production scenario"""
    print("\n" + "="*70)
    print("TEST 4: PRODUCTION SCENARIO (Complete Example)")
    print("="*70)
    
    extender = get_smart_video_extender()
    transition_core = get_cinematic_transition_core()
    
    # Simulate real production: 3 scenes with different lengths
    scene1, fps = create_test_clip(duration_sec=2.5, fps=24,
                                   text="Scene 1: Temple", color=(150, 100, 50))
    scene2, fps = create_test_clip(duration_sec=1.8, fps=24,
                                   text="Scene 2: Forest", color=(50, 150, 50))
    scene3, fps = create_test_clip(duration_sec=2.2, fps=24,
                                   text="Scene 3: Mountain", color=(100, 100, 150))
    
    # Audio durations (simulated)
    audio_durations = [4.0, 3.5, 4.2]  # Longer than video clips!
    
    print("\n📊 Production Challenge:")
    print(f"   Scene 1: {len(scene1)/fps:.1f}s video, {audio_durations[0]}s audio → Need {audio_durations[0] - len(scene1)/fps:.1f}s extension")
    print(f"   Scene 2: {len(scene2)/fps:.1f}s video, {audio_durations[1]}s audio → Need {audio_durations[1] - len(scene2)/fps:.1f}s extension")
    print(f"   Scene 3: {len(scene3)/fps:.1f}s video, {audio_durations[2]}s audio → Need {audio_durations[2] - len(scene3)/fps:.1f}s extension")
    
    # Extend each scene to match audio
    print("\n📹 Extending scenes with smart extension...")
    scene1_extended, fps1 = extender.extend_to_duration(
        scene1, len(scene1)/fps, audio_durations[0], fps, ExtensionMethod.COMBINED
    )
    scene2_extended, fps2 = extender.extend_to_duration(
        scene2, len(scene2)/fps, audio_durations[1], fps, ExtensionMethod.COMBINED
    )
    scene3_extended, fps3 = extender.extend_to_duration(
        scene3, len(scene3)/fps, audio_durations[2], fps, ExtensionMethod.COMBINED
    )
    
    # Add transitions between scenes
    print("\n📹 Adding cinematic transitions...")
    trans1 = transition_core.create_fade_transition(
        scene1_extended, scene2_extended, duration=0.5, fps=fps,
        fade_color=(0, 0, 0), easing="ease_in_out"
    )
    trans2 = transition_core.create_dissolve_transition(
        scene2_extended, scene3_extended, duration=0.5, fps=fps, easing="ease_in_out"
    )
    
    # Combine everything
    final_video = np.concatenate([
        scene1_extended, trans1,
        scene2_extended, trans2,
        scene3_extended
    ], axis=0)
    
    output_dir = Path("outputs/day5_visual_tests")
    save_video(final_video, fps, output_dir / "4_production_complete.mp4",
              f"✅ Complete production: {len(final_video)/fps:.1f}s with extensions + transitions")
    
    print(f"\n✅ Final video: {len(final_video)/fps:.1f}s total")
    print(f"   - Scene 1: Extended {len(scene1)/fps:.1f}s → {len(scene1_extended)/fps:.1f}s")
    print(f"   - Transition 1: {len(trans1)/fps:.1f}s fade to black")
    print(f"   - Scene 2: Extended {len(scene2)/fps:.1f}s → {len(scene2_extended)/fps:.1f}s")
    print(f"   - Transition 2: {len(trans2)/fps:.1f}s dissolve")
    print(f"   - Scene 3: Extended {len(scene3)/fps:.1f}s → {len(scene3_extended)/fps:.1f}s")


def main():
    """Run all visual tests"""
    print("\n" + "="*70)
    print("  DAY 5 VISUAL TESTING - Smart Extension + Transitions")
    print("="*70)
    print("\nThis will create video files in: outputs/day5_visual_tests/")
    print("You can open and play these videos to visually verify the features!\n")
    
    try:
        # Run all tests
        output_dir = test_1_video_looping_problem()
        test_2_smart_extension_solution()
        test_3_cinematic_transitions()
        test_4_production_scenario()
        
        print("\n" + "="*70)
        print("  ✅ ALL VISUAL TESTS COMPLETE!")
        print("="*70)
        print(f"\n📁 Output Directory: {output_dir.absolute()}")
        print("\n📹 Generated Videos:")
        
        video_files = sorted(output_dir.glob("*.mp4"))
        for i, video_file in enumerate(video_files, 1):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"   {i}. {video_file.name} ({size_mb:.1f} MB)")
        
        print(f"\n🎬 Total: {len(video_files)} test videos created")
        print("\n💡 To view:")
        print(f"   1. Open File Explorer: {output_dir.absolute()}")
        print("   2. Play videos with VLC or Windows Media Player")
        print("   3. Compare BEFORE (1b_looped_repetitive.mp4)")
        print("      vs AFTER (2c_combined_extension.mp4)")
        print("\n✨ Key videos to check:")
        print("   • 1b_looped_repetitive.mp4 - The PROBLEM (repetitive looping)")
        print("   • 2c_combined_extension.mp4 - The SOLUTION (natural extension)")
        print("   • 3c_dissolve.mp4 - Smooth scene transition")
        print("   • 4_production_complete.mp4 - Full production example")
        
    except Exception as e:
        print(f"\n❌ Error during visual testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
