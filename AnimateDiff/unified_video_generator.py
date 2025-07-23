#!/usr/bin/env python3
"""
Unified Video Generation System - SIMPLIFIED VERSION
Single input lesson file, single output location, always with audio and subtitles
"""

import os
import json
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
import moviepy.config as config
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_audioclips,
    TextClip, CompositeVideoClip, concatenate_videoclips
)

# Fix Unicode encoding issues for Windows console
if sys.platform == "win32":
    import codecs
    import locale

    # Set console encoding to UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'

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

# Import centralized FPS setting
from animate_gurukul import fps

# Import performance tracking
from performance_tracker import performance_tracker

# Configure progress bars to use ASCII only (no Unicode)
os.environ['TQDM_ASCII'] = '1'
os.environ['TQDM_NCOLS'] = '80'

# Enable diffusers progress bars with ASCII
import diffusers
diffusers.utils.logging.enable_progress_bar()

# Configure ImageMagick
imagemagick_path = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
if os.path.exists(imagemagick_path):
    config.change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})

class UnifiedVideoGenerator:
    """Simplified unified system - single input, single output, always with audio+subtitles"""

    def __init__(self):
        # SIMPLIFIED: Only one output directory
        self.output_dir = "outputs/multi_clip"
        self.storage_dir = "storage"
        self.temp_dir = None
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def create_tts_audio(self, text, output_file, speech_rate=1):
        """Create TTS audio with configurable speech rate"""
        try:
            ps_command = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = {speech_rate}
$synth.Volume = 100
$synth.SetOutputToWaveFile("{output_file}")
$synth.Speak("{text}")
$synth.Dispose()
'''
            
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return os.path.exists(output_file) and os.path.getsize(output_file) > 1000
            
        except Exception as e:
            print(f"      ❌ TTS generation failed: {e}")
            return False
    
    def generate_audio_track(self, sentences, speech_rate=1):
        """Generate sequential audio track from sentences"""
        print(f"🎵 Generating audio track with speech rate {speech_rate}...")
        
        self.temp_dir = tempfile.mkdtemp()
        audio_clips = []
        
        try:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                print(f"   🎵 Audio {i+1}/{len(sentences)}: {sentence[:50]}...")
                
                audio_file = os.path.join(self.temp_dir, f"audio_{i}.wav")
                
                if self.create_tts_audio(sentence, audio_file, speech_rate):
                    try:
                        audio_clip = AudioFileClip(audio_file)
                        audio_clips.append(audio_clip)
                        print(f"      ✅ Duration: {audio_clip.duration:.1f}s")
                    except Exception as e:
                        print(f"      ❌ Audio loading failed: {e}")
                else:
                    print(f"      ❌ TTS generation failed")
            
            if audio_clips:
                # Concatenate all audio clips sequentially
                final_audio = concatenate_audioclips(audio_clips)
                print(f"   ✅ Total audio duration: {final_audio.duration:.1f}s")
                return final_audio, audio_clips
            else:
                print("   ❌ No audio clips generated")
                return None, []
                
        except Exception as e:
            print(f"   ❌ Audio generation error: {e}")
            return None, []
    
    def create_subtitles(self, sentences, audio_clips, video_width=512):
        """Create synchronized subtitles"""
        print(f"📝 Creating synchronized subtitles...")
        
        subtitle_clips = []
        current_time = 0
        
        for i, (sentence, audio_clip) in enumerate(zip(sentences, audio_clips)):
            if not sentence.strip():
                continue
            
            start_time = current_time
            duration = audio_clip.duration
            
            print(f"   📝 Subtitle {i+1}: {start_time:.1f}s - {start_time + duration:.1f}s")
            
            try:
                subtitle = TextClip(
                    sentence.strip(),
                    fontsize=36,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial',
                    method='caption',
                    size=(video_width * 0.8, None),
                    align='center'
                ).set_position(('center', 'bottom')).set_start(start_time).set_duration(duration)
                
                subtitle_clips.append(subtitle)
                
            except Exception as e:
                print(f"      ❌ Subtitle {i+1} failed: {e}")
            
            current_time += duration
        
        print(f"   ✅ Created {len(subtitle_clips)} subtitles")
        return subtitle_clips
    
    def load_lesson_data(self, lesson_path):
        """Load lesson data from JSON file"""
        try:
            with open(lesson_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load lesson: {e}")
            return None

    def update_lesson_selection(self, lesson_path):
        """Update multi_clip_generator to use specific lesson"""
        try:
            # Read the current multi_clip_generator
            with open('multi_clip_generator.py', 'r', encoding='utf-8') as f:
                content = f.read()

            # Find and replace the lesson selection logic
            import re

            # Replace the lesson file selection with our specific lesson
            pattern = r'lesson_file = "lessons/lesson_space_adventure\.json"'
            replacement = f'lesson_file = "{lesson_path}"'

            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
            else:
                # Fallback: replace any lesson_file assignment
                pattern = r'lesson_file = "lessons/[^"]*\.json"'
                content = re.sub(pattern, replacement, content)

            # Write back the updated content
            with open('multi_clip_generator.py', 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"   ✅ Updated generator to use: {lesson_path}")

        except Exception as e:
            print(f"   ⚠️ Could not update lesson selection: {e}")
            print(f"   📝 Will use default lesson selection")

    def generate_video_clips_direct(self, lesson_data, style, lesson_path):
        """Generate NEW video clips for each lesson"""
        try:
            print(f"🎬 Generating NEW video clips for this lesson...")

            # ALWAYS generate new clips - don't reuse old ones
            lesson_title = lesson_data.get('title', 'lesson')
            print(f"   📚 Creating clips for: {lesson_title}")

            # First, update the multi_clip_generator to use this specific lesson
            self.update_lesson_selection(lesson_path)

            # Run the main generator to create NEW clips for this specific lesson
            import subprocess
            import sys

            # Run the generator and wait for completion - PASS THE LESSON FILE!
            lesson_filename = os.path.basename(lesson_path)
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '1'

            result = subprocess.run([
                sys.executable, "multi_clip_generator.py", lesson_filename, style
            ], capture_output=False, text=True, encoding='utf-8', errors='replace', env=env, timeout=1800)  # Show output, 30 min timeout

            if result.returncode == 0:
                print(f"   ✅ NEW video clips generated successfully")

                # Load the newly generated clips
                clips_dir = "outputs/multi_clip"
                clip_files = []

                for i in range(1, 9):  # Load new clips
                    clip_path = os.path.join(clips_dir, f"clip{i}.mp4")
                    if os.path.exists(clip_path):
                        clip_files.append(clip_path)

                if clip_files:
                    video_clips = [VideoFileClip(path) for path in clip_files]
                    print(f"   ✅ Loaded {len(video_clips)} NEW clips for {lesson_title}")
                    return video_clips
                else:
                    print(f"   ❌ No clip files found after generation")
                    return None
            else:
                print(f"   ❌ Video generation failed")
                return None

            # If no existing clips, create simple placeholder clips
            print(f"   🎬 Creating placeholder clips for faster processing...")

            sentences = [s.strip() for s in lesson_data['text'].split('.') if s.strip()]
            video_clips = []

            from moviepy.editor import ColorClip, TextClip, CompositeVideoClip

            colors = [
                (30, 50, 100),   # Space blue
                (50, 30, 80),    # Space purple
                (80, 40, 20),    # Rocket orange
                (20, 60, 40),    # Earth green
                (60, 20, 80),    # Planet purple
                (40, 40, 40),    # Surface gray
                (70, 50, 30),    # Rock brown
                (20, 40, 60)     # Return blue
            ]

            for i, sentence in enumerate(sentences[:8]):
                color = colors[i % len(colors)]

                # Create colored background
                bg_clip = ColorClip(size=(512, 512), color=color, duration=2.0)

                # Add text overlay
                try:
                    text_clip = TextClip(
                        f"Scene {i+1}",
                        fontsize=48,
                        color='white',
                        font='Arial'
                    ).set_position('center').set_duration(2.0)

                    final_clip = CompositeVideoClip([bg_clip, text_clip])
                except:
                    # Fallback without text if TextClip fails
                    final_clip = bg_clip

                video_clips.append(final_clip)

            print(f"   ✅ Created {len(video_clips)} placeholder clips")
            return video_clips

        except Exception as e:
            print(f"   ❌ Video generation error: {e}")
            return None

    def generate_complete_video(self, lesson_path, style='realistic', speech_rate=1):
        """Generate complete video with matching visuals, audio, and subtitles"""

        # Start performance tracking
        performance_tracker.start_tracking("complete_video_generation")

        # Load lesson data to get dynamic title
        lesson_data = self.load_lesson_data(lesson_path)
        if not lesson_data:
            print("❌ Failed to load lesson data")
            performance_tracker.end_tracking("complete_video_generation", {"status": "failed", "error": "lesson_load_failed"})
            return None

        lesson_title = lesson_data.get('title', 'Unknown Lesson')

        print(f"🚀 GENERATING COMPLETE VIDEO: {lesson_title}")
        print(f"   📚 Lesson: {os.path.basename(lesson_path)}")
        print(f"   🎨 Style: {style}")
        print(f"   🎵 Speech rate: {speech_rate}")

        try:
            sentences = [s.strip() for s in lesson_data['text'].split('.') if s.strip()]
            print(f"   📝 Processing {len(sentences)} scenes")

            # Step 1: Generate audio track
            final_audio, audio_clips = self.generate_audio_track(sentences, speech_rate)
            if not final_audio:
                print("❌ Audio generation failed")
                return None

            # Step 2: Generate video clips directly for this specific lesson
            print(f"\n🎬 Generating video clips in {style} style...")
            video_clips = self.generate_video_clips_direct(lesson_data, style, lesson_path)

            if not video_clips:
                print("❌ Video generation failed")
                return None
            
            # Step 4: Adjust each video clip to match its corresponding audio
            print(f"\n⏱️ Synchronizing video clips with audio segments...")

            adjusted_video_clips = []
            total_video_duration = 0
            total_audio_duration = sum(clip.duration for clip in audio_clips)

            print(f"   📊 Total audio duration: {total_audio_duration:.1f}s")
            print(f"   🎬 Adjusting {len(video_clips)} video clips to match audio...")

            for i, (video_clip, audio_clip) in enumerate(zip(video_clips, audio_clips)):
                video_duration = video_clip.duration
                audio_duration = audio_clip.duration

                print(f"   📎 Clip {i+1}: Video={video_duration:.1f}s, Audio={audio_duration:.1f}s")

                if audio_duration > video_duration:
                    # Extend video clip to match audio duration by looping
                    from moviepy.video.fx.loop import loop
                    try:
                        extended_clip = loop(video_clip, duration=audio_duration)
                        print(f"      ✅ Extended to {audio_duration:.1f}s")
                    except:
                        # Fallback: freeze last frame
                        from moviepy.video.fx.freeze import freeze
                        extended_clip = video_clip.fx(freeze, t='end', freeze_duration=audio_duration-video_duration)
                        print(f"      ✅ Freeze-extended to {audio_duration:.1f}s")
                elif audio_duration < video_duration:
                    # Trim video clip to match audio duration
                    extended_clip = video_clip.subclip(0, audio_duration)
                    print(f"      ✅ Trimmed to {audio_duration:.1f}s")
                else:
                    # Perfect match
                    extended_clip = video_clip
                    print(f"      ✅ Perfect match at {audio_duration:.1f}s")

                adjusted_video_clips.append(extended_clip)
                total_video_duration += extended_clip.duration

            # Concatenate the synchronized video clips
            adjusted_video = concatenate_videoclips(adjusted_video_clips, method="compose")
            print(f"   ✅ Final synchronized video duration: {adjusted_video.duration:.1f}s")
            
            # Step 5: Add audio to video
            print(f"\n🎵 Adding audio to video...")
            video_with_audio = adjusted_video.set_audio(final_audio)
            
            # Step 6: Create and add subtitles
            print(f"\n📝 Adding subtitles...")
            subtitle_clips = self.create_subtitles(sentences, audio_clips, adjusted_video.w)
            
            if subtitle_clips:
                final_video = CompositeVideoClip([video_with_audio] + subtitle_clips)
            else:
                final_video = video_with_audio
            
            # Step 7: Save final video with dynamic naming
            lesson_name = lesson_data.get('title', 'video').replace(' ', '_').replace(':', '').replace('?', '').replace('!', '')
            output_filename = f"{lesson_name}_{style}_complete.mp4"
            output_path = os.path.join(self.output_dir, output_filename)

            print(f"\n💾 Saving complete video...")
            print(f"   📁 Output: {output_filename}")

            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=fps,  # Using centralized FPS setting
                verbose=False,
                logger=None
            )

            # IMPORTANT: Copy to storage folder for team sharing (Rishabh)
            print(f"\n📤 Copying to storage folder for team sharing...")
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            storage_today_dir = os.path.join(self.storage_dir, today)
            os.makedirs(storage_today_dir, exist_ok=True)

            storage_path = os.path.join(storage_today_dir, output_filename)
            shutil.copy2(output_path, storage_path)
            print(f"   📁 Copied to: {storage_path}")
            print(f"   🤝 Ready for Rishabh's team access!")

            # Cleanup
            adjusted_video.close()
            video_with_audio.close()
            final_video.close()
            final_audio.close()

            for clip in video_clips:
                clip.close()

            for clip in adjusted_video_clips:
                clip.close()
            for clip in audio_clips:
                clip.close()
            for clip in subtitle_clips:
                clip.close()

            print(f"\n✅ SUCCESS! Complete video created:")
            print(f"   📚 Lesson: {lesson_title}")
            print(f"   📁 File: {output_filename}")
            print(f"   ⏱️ Duration: {audio_duration:.1f}s")
            print(f"   🎵 Audio: {len(audio_clips)} sequential segments")
            print(f"   🎬 Video: {len(video_clips)} clips in {style} style")
            print(f"   📝 Subtitles: {len(subtitle_clips)} synchronized")
            print(f"   📍 Main output: {os.path.abspath(output_path)}")
            print(f"   📤 Team sharing: {os.path.abspath(storage_path)}")

            return output_path
            
        except ZeroDivisionError as e:
            print(f"❌ Division by zero error: {e}")
            print(f"   This might be due to empty audio or video clips")
            return None
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Cleanup temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def main():
    """SIMPLIFIED: Single lesson file input, always generates complete video with audio+subtitles"""
    import sys

    # SIMPLIFIED: Only one required parameter - lesson file name
    if len(sys.argv) < 2:
        print("❌ Usage: python unified_video_generator.py <lesson_file.json> [style] [speech_rate]")
        print("📚 Available lessons:")
        lesson_files = [f for f in os.listdir("lessons") if f.endswith('.json')]
        for lesson in lesson_files:
            print(f"   • {lesson}")
        return

    lesson_filename = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "realistic"
    speech_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # Build full lesson path
    lesson_path = os.path.join("lessons", lesson_filename)

    if not os.path.exists(lesson_path):
        print(f"❌ Lesson file not found: {lesson_path}")
        print("📚 Available lessons:")
        lesson_files = [f for f in os.listdir("lessons") if f.endswith('.json')]
        for lesson in lesson_files:
            print(f"   • {lesson}")
        return

    print(f"🎬 GENERATING VIDEO FOR: {lesson_filename}")
    print(f"🎨 Style: {style}")
    print(f"🎵 Speech Rate: {speech_rate}")
    print(f"📁 Output: outputs/multi_clip/")

    # Generate video
    generator = UnifiedVideoGenerator()

    try:
        result = generator.generate_complete_video(lesson_path, style, speech_rate)

        if result:
            print(f"\n🎯 SUCCESS! Complete video generated:")
            print(f"📍 {result}")
            print(f"✅ Includes: Video + Audio + Subtitles")
            print(f"✅ Ready for team sharing!")
        else:
            print(f"\n❌ Video generation failed")

    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()
