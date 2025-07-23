#!/usr/bin/env python3
"""
SAFE LESSON VIDEO GENERATOR - Unicode Safe Version
Single input, single output, always with audio and subtitles
Handles Unicode encoding issues for API calls

Usage: python generate_lesson_video_safe.py <lesson_file.json> [style] [speech_rate]
"""

import os
import sys

# Fix Unicode encoding issues for Windows console
if sys.platform == "win32":
    import codecs
    import locale

    # Set console encoding to UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'

    # Configure progress bars to use ASCII only (no Unicode)
    os.environ['TQDM_ASCII'] = '1'
    os.environ['TQDM_NCOLS'] = '80'

    # Try to set console code page to UTF-8
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass

    # Set locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

    # Reconfigure stdout/stderr with UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        # Fallback for older Python versions
        try:
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
        except:
            pass

def main():
    """Safe wrapper for the unified video generator with Unicode handling"""
    
    # Safe print function that handles Unicode issues
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)
    
    safe_print("GURUKUL LESSON VIDEO GENERATOR")
    safe_print("=" * 50)
    
    if len(sys.argv) < 2:
        safe_print("ERROR: Please provide a lesson file!")
        safe_print("\nUsage:")
        safe_print("   python generate_lesson_video_safe.py <lesson_file.json> [style] [speech_rate]")
        safe_print("\nAvailable lessons:")
        
        if os.path.exists("lessons"):
            lesson_files = [f for f in os.listdir("lessons") if f.endswith('.json')]
            for lesson in sorted(lesson_files):
                safe_print(f"   - {lesson}")
        else:
            safe_print("   ERROR: No lessons folder found!")
        
        safe_print("\nAvailable styles: realistic, anime, artistic")
        safe_print("Speech rates: 1 (normal), 2 (fast), 0.5 (slow)")
        return
    
    # Import and run the unified generator
    try:
        # Set environment variables for Unicode handling
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # Import with Unicode safety
        sys.path.insert(0, '.')

        from unified_video_generator import UnifiedVideoGenerator

        # Get parameters
        lesson_filename = sys.argv[1]
        style = sys.argv[2] if len(sys.argv) > 2 else "realistic"
        speech_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 1

        # Build lesson path
        lesson_path = os.path.join("lessons", lesson_filename)

        if not os.path.exists(lesson_path):
            safe_print(f"ERROR: Lesson file not found: {lesson_path}")
            return

        safe_print(f"GENERATING VIDEO FOR: {lesson_filename}")
        safe_print(f"Style: {style}")
        safe_print(f"Speech Rate: {speech_rate}")
        safe_print(f"Output: outputs/multi_clip/")

        # Generate video directly
        generator = UnifiedVideoGenerator()
        result = generator.generate_complete_video(lesson_path, style, speech_rate)

        if result:
            safe_print(f"\nSUCCESS! Complete video generated:")
            safe_print(f"File: {result}")
            safe_print(f"Includes: Video + Audio + Subtitles")
            safe_print(f"Ready for team sharing!")
        else:
            safe_print(f"\nERROR: Video generation failed")

        generator.cleanup()

    except ImportError as e:
        safe_print(f"ERROR: Error importing unified_video_generator: {e}")
        safe_print("   Make sure you're in the AnimateDiff directory!")
    except Exception as e:
        safe_print(f"ERROR: Error generating video: {e}")
        import traceback
        safe_print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
