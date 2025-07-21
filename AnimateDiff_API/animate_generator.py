# animate_generator.py - UPDATED TO USE LATEST AnimateDiff SYSTEM
import os
import sys
import subprocess
import json
import tempfile
from datetime import datetime
from pathlib import Path

# Add the main AnimateDiff folder to Python path
ANIMATEDIFF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AnimateDiff")
sys.path.insert(0, ANIMATEDIFF_PATH)

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

        # Run the unified video generator
        cmd = [
            sys.executable,
            os.path.join(ANIMATEDIFF_PATH, "generate_lesson_video.py"),
            temp_lesson_filename,
            style,
            "1"  # speech rate
        ]

        print(f"🚀 Running command: {' '.join(cmd)}")

        # Change to AnimateDiff directory and run
        result = subprocess.run(
            cmd,
            cwd=ANIMATEDIFF_PATH,
            capture_output=True,
            text=True,
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

        # Run the unified video generator
        cmd = [
            sys.executable,
            os.path.join(ANIMATEDIFF_PATH, "generate_lesson_video.py"),
            lesson_filename,
            style,
            str(speech_rate)
        ]

        print(f"🚀 Running lesson generation: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=ANIMATEDIFF_PATH,
            capture_output=True,
            text=True,
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
