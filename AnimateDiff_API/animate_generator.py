# animate_generator.py - UPDATED TO USE LATEST AnimateDiff SYSTEM
import os
import sys
import subprocess
import json
import tempfile
from datetime import datetime
from pathlib import Path

# Set UTF-8 encoding for subprocess calls to handle emoji characters
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add the main AnimateDiff folder to Python path
ANIMATEDIFF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AnimateDiff")
sys.path.insert(0, ANIMATEDIFF_PATH)

# Import text optimizer and fallback system
sys.path.append(ANIMATEDIFF_PATH)
try:
    from text_optimizer import TextOptimizer  # type: ignore
except ImportError:
    print("Warning: text_optimizer module not found, text optimization will be skipped")
    TextOptimizer = None

try:
    from fallback_generator import FallbackGenerator  # type: ignore
except ImportError:
    print("Warning: fallback_generator module not found, fallback generation will be skipped")
    FallbackGenerator = None

def generate_video(prompt, negative_prompt=None, num_frames=32, steps=25, guidance_scale=15, seed=333, fps=12, style="realistic"):
    """
    Generate video using the latest unified AnimateDiff system
    Now supports audio, subtitles, and multiple styles
    """
    try:
        # Create a temporary lesson file for the API request
        lesson_data = {
            "title": "API Generated Video",
            "level": "API",
            "text": prompt,
            "scenes": [
                {
                    "description": prompt,
                    "duration": 4.0
                }
            ],
            "tts": True
        }

        # Create temporary lesson file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_lesson_filename = f"api_lesson_{timestamp}.json"
        temp_lesson_path = os.path.join(ANIMATEDIFF_PATH, "lessons", temp_lesson_filename)

        with open(temp_lesson_path, 'w') as f:
            json.dump(lesson_data, f, indent=2)

        print(f"🎬 Created temporary lesson: {temp_lesson_filename}")

        # OPTIMIZE THE TEXT using Gemini API (if available)
        if TextOptimizer is not None:
            print(f"🧠 Optimizing text with Gemini API...")
            optimizer = TextOptimizer()
            optimized_lesson_filename = f"api_lesson_{timestamp}_optimized.json"
            optimized_lesson_path = os.path.join(ANIMATEDIFF_PATH, "lessons", optimized_lesson_filename)

            optimized_result = optimizer.create_optimized_lesson(temp_lesson_path, optimized_lesson_path)

            if optimized_result:
                print(f"✅ Text optimization successful!")
                # Use the optimized lesson instead
                temp_lesson_filename = optimized_lesson_filename
            else:
                print(f"⚠️ Text optimization failed, using original lesson")
        else:
            print(f"ℹ️ Text optimizer not available, using original lesson")

        # Run the unified video generator using Unicode-safe version
        cmd = [
            sys.executable,
            os.path.join(ANIMATEDIFF_PATH, "generate_lesson_video_safe.py"),
            temp_lesson_filename,
            style,
            "1"  # speech rate
        ]

        print(f"🚀 Running command: {' '.join(cmd)}")

        # Change to AnimateDiff directory and run with proper encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'

        result = subprocess.run(
            cmd,
            cwd=ANIMATEDIFF_PATH,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            timeout=1800  # 30 minutes timeout
        )

        if result.returncode == 0:
            # Find the generated video file
            output_dir = os.path.join(ANIMATEDIFF_PATH, "outputs", "multi_clip")
            expected_filename = f"API_Generated_Video_{style}_complete.mp4"
            output_path = os.path.join(output_dir, expected_filename)

            if os.path.exists(output_path):
                # Copy to API outputs folder
                api_output_filename = f"animation_{timestamp}.mp4"
                api_output_path = os.path.join("outputs", api_output_filename)

                import shutil
                shutil.copy2(output_path, api_output_path)

                print(f"✅ Video generated successfully: {api_output_path}")

                # Cleanup temporary lesson file
                try:
                    os.remove(temp_lesson_path)
                except:
                    pass

                return api_output_path
            else:
                raise Exception(f"Generated video not found at: {output_path}")
        else:
            error_msg = f"Video generation failed: {result.stderr}"
            print(f"❌ {error_msg}")

            # Try fallback video generation (if available)
            if FallbackGenerator is not None:
                print(f"🔄 Attempting fallback video generation...")
                try:
                    fallback_generator = FallbackGenerator()
                    fallback_filename = f"fallback_{timestamp}.mp4"

                    # Load the lesson data for fallback
                    with open(temp_lesson_path, 'r') as f:
                        lesson_data = json.load(f)

                    fallback_path = fallback_generator.create_fallback_video(
                        lesson_data,
                        fallback_filename,
                        duration=20
                    )

                    if fallback_path:
                        # Copy to API outputs folder
                        api_output_filename = f"animation_{timestamp}.mp4"
                        api_output_path = os.path.join("outputs", api_output_filename)

                        import shutil
                        shutil.copy2(fallback_path, api_output_path)

                        print(f"✅ Fallback video generated: {api_output_path}")

                        # Cleanup temporary lesson file
                        try:
                            os.remove(temp_lesson_path)
                        except:
                            pass

                        return api_output_path
                    else:
                        raise Exception("Fallback video generation also failed")

                except Exception as fallback_error:
                    print(f"❌ Fallback generation failed: {fallback_error}")
                    raise Exception(error_msg)
            else:
                print(f"ℹ️ Fallback generator not available")
                raise Exception(error_msg)

    except Exception as e:
        # Cleanup temporary lesson file on error
        try:
            if 'temp_lesson_path' in locals():
                os.remove(temp_lesson_path)
        except:
            pass

        print(f"❌ Error in generate_video: {str(e)}")
        raise e

def generate_lesson_video(lesson_filename, style="realistic", speech_rate=1):
    """
    Generate video from existing lesson file using the unified system
    """
    try:
        lesson_path = os.path.join(ANIMATEDIFF_PATH, "lessons", lesson_filename)

        if not os.path.exists(lesson_path):
            raise Exception(f"Lesson file not found: {lesson_path}")

        # Run the unified video generator using Unicode-safe version
        cmd = [
            sys.executable,
            os.path.join(ANIMATEDIFF_PATH, "generate_lesson_video_safe.py"),
            lesson_filename,
            style,
            str(speech_rate)
        ]

        print(f"🚀 Running lesson generation: {' '.join(cmd)}")

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'

        result = subprocess.run(
            cmd,
            cwd=ANIMATEDIFF_PATH,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            timeout=1800  # 30 minutes timeout
        )

        if result.returncode == 0:
            # Find the generated video in storage folder (for team sharing)
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            storage_dir = os.path.join(ANIMATEDIFF_PATH, "storage", today)

            if os.path.exists(storage_dir):
                # Find the most recent video file
                video_files = [f for f in os.listdir(storage_dir) if f.endswith('.mp4')]
                if video_files:
                    latest_video = max(video_files, key=lambda x: os.path.getctime(os.path.join(storage_dir, x)))
                    storage_path = os.path.join(storage_dir, latest_video)

                    # Copy to API outputs folder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    api_output_filename = f"lesson_{timestamp}.mp4"
                    api_output_path = os.path.join("outputs", api_output_filename)

                    import shutil
                    shutil.copy2(storage_path, api_output_path)

                    print(f"✅ Lesson video generated: {api_output_path}")
                    return api_output_path

            raise Exception("Generated video not found in storage folder")
        else:
            error_msg = f"Lesson video generation failed: {result.stderr}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

    except Exception as e:
        print(f"❌ Error in generate_lesson_video: {str(e)}")
        raise e
