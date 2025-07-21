#!/usr/bin/env python3
"""
SIMPLIFIED LESSON VIDEO GENERATOR
Single input, single output, always with audio and subtitles

Usage: python generate_lesson_video.py <lesson_file.json> [style] [speech_rate]

Examples:
  python generate_lesson_video.py lesson_1_dharma.json
  python generate_lesson_video.py lesson_ocean_adventure.json realistic 1
  python generate_lesson_video.py lesson_forest_wisdom.json anime 2
"""

import os
import sys

def main():
    """Simple wrapper for the unified video generator"""
    
    print("🎬 GURUKUL LESSON VIDEO GENERATOR")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("❌ Please provide a lesson file!")
        print("\n📚 Usage:")
        print("   python generate_lesson_video.py <lesson_file.json> [style] [speech_rate]")
        print("\n📚 Available lessons:")
        
        if os.path.exists("lessons"):
            lesson_files = [f for f in os.listdir("lessons") if f.endswith('.json')]
            for lesson in sorted(lesson_files):
                print(f"   • {lesson}")
        else:
            print("   ❌ No lessons folder found!")
        
        print("\n🎨 Available styles: realistic, anime, artistic")
        print("🎵 Speech rates: 1 (normal), 2 (fast), 0.5 (slow)")
        return
    
    # Import and run the unified generator
    try:
        from unified_video_generator import main as unified_main
        
        # The unified_video_generator.main() will handle the arguments
        unified_main()
        
    except ImportError as e:
        print(f"❌ Error importing unified_video_generator: {e}")
        print("   Make sure you're in the AnimateDiff directory!")
    except Exception as e:
        print(f"❌ Error generating video: {e}")

if __name__ == "__main__":
    main()
