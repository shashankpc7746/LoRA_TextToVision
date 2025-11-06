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
from datetime import datetime
from typing import List, Dict
import moviepy.config as config
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_audioclips,
    TextClip, CompositeVideoClip, concatenate_videoclips
)
from subtitle_sync_engine import SubtitleSyncEngine
from cinematic_flow_engine import CinematicFlowEngine

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
    
    def create_advanced_subtitles(self, sentences, audio_clips, video_width=512, language='english'):
        """Create advanced synchronized subtitles with Gurukul styling"""
        print(f"📝 Creating advanced synchronized subtitles...")

        try:
            # Initialize subtitle sync engine
            subtitle_engine = SubtitleSyncEngine()

            # Generate precise SRT file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            srt_path = os.path.join(self.temp_dir, f"subtitles_{timestamp}.srt")

            subtitle_engine.generate_precise_subtitles(
                audio_clips=audio_clips,
                text_segments=sentences,
                output_path=srt_path,
                language=language
            )

            # Generate subtitle report
            report = subtitle_engine.generate_subtitle_report(srt_path)
            print(f"   📊 Subtitle Report:")
            print(f"      • Total segments: {report.get('total_segments', 0)}")
            print(f"      • Total duration: {report.get('total_duration', 0):.1f}s")
            print(f"      • Average reading speed: {report.get('average_reading_speed', 0):.1f} chars/sec")

            # Parse SRT to create MoviePy clips for preview
            subtitle_clips = subtitle_engine._parse_srt_to_clips(srt_path, video_width, language)

            print(f"   ✅ Created {len(subtitle_clips)} advanced subtitles")
            print(f"   📁 SRT file: {srt_path}")

            return subtitle_clips, srt_path

        except Exception as e:
            print(f"   ❌ Advanced subtitle creation failed: {e}")
            # Fallback to simple subtitles
            return self.create_simple_subtitles(sentences, audio_clips, video_width), None

    def create_simple_subtitles(self, sentences, audio_clips, video_width=512):
        """Fallback simple subtitle creation"""
        print(f"   🔄 Using fallback simple subtitles...")

        subtitle_clips = []
        current_time = 0

        for i, (sentence, audio_clip) in enumerate(zip(sentences, audio_clips)):
            if not sentence.strip():
                continue

            start_time = current_time
            duration = audio_clip.duration

            try:
                # FIXED: Proper movie-style subtitle positioning (moved significantly up)
                subtitle = TextClip(
                    sentence.strip(),
                    fontsize=32,  # Reduced size as requested
                    color='white',  # Pure white
                    font='Arial-Bold',  # Bold for visibility
                    method='caption',
                    size=(video_width * 0.85, None),
                    align='center'
                ).set_position(('center', 350)).set_start(start_time).set_duration(duration)  # FIXED: Position at 350px from top (162px from bottom)

                subtitle_clips.append(subtitle)

            except Exception as e:
                print(f"      ❌ Subtitle {i+1} failed: {e}")

            current_time += duration

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
            ], capture_output=False, text=True, encoding='utf-8', errors='replace', env=env, timeout=7200)  # Show output, 2 hour timeout

            if result.returncode == 0:
                print(f"   ✅ NEW video clips generated successfully")

                # Load the newly generated clips
                clips_dir = "outputs/multi_clip"
                clip_files = []

                # Dynamically detect all available clips (not hardcoded to 8!)
                i = 1
                while True:
                    clip_path = os.path.join(clips_dir, f"clip{i}.mp4")
                    if os.path.exists(clip_path):
                        clip_files.append(clip_path)
                        i += 1
                    else:
                        break  # No more clips found

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
            
            # Step 4: Apply cinematic flow and transitions
            print(f"\n🎬 Applying cinematic flow and transitions...")

            # Extract scene contexts from lesson data
            scene_contexts = self._extract_scene_contexts(lesson_data)
            flow_instructions = self._generate_flow_instructions(lesson_data, len(video_clips))

            # Initialize cinematic flow engine
            cinematic_engine = CinematicFlowEngine()

            # Apply cinematic enhancements
            cinematic_video_clips = []
            for i, video_clip in enumerate(video_clips):
                scene = scene_contexts[i] if i < len(scene_contexts) else 'temple'
                flow_instruction = flow_instructions[i] if i < len(flow_instructions) else {}

                enhanced_clip = cinematic_engine._enhance_clip_with_flow(
                    video_clip, scene, flow_instruction, i
                )
                cinematic_video_clips.append(enhanced_clip)
                print(f"   🎬 Clip {i+1}: Applied {flow_instruction.get('movement', 'default')} in {scene} scene")

            # Step 5: Synchronize enhanced clips with audio
            print(f"\n⏱️ Synchronizing cinematic clips with audio segments...")

            adjusted_video_clips = []
            total_video_duration = 0
            total_audio_duration = sum(clip.duration for clip in audio_clips)

            print(f"   📊 Total audio duration: {total_audio_duration:.1f}s")
            print(f"   🎬 Adjusting {len(cinematic_video_clips)} cinematic clips to match audio...")

            for i, (video_clip, audio_clip) in enumerate(zip(cinematic_video_clips, audio_clips)):
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

            # Step 6: Create final video (TEMPORARILY DISABLED cinematic transitions to fix black screen)
            print(f"\n🎭 Creating final video sequence...")

            # SAFETY: Use simple concatenation to prevent black screen issues
            from moviepy.editor import concatenate_videoclips
            adjusted_video = concatenate_videoclips(adjusted_video_clips, method="compose")
            print(f"   ✅ Safe video sequence created: {adjusted_video.duration:.1f}s")
            print(f"   ⚠️ Cinematic transitions temporarily disabled for stability")
            
            # Step 5: Add audio to video
            print(f"\n🎵 Adding audio to video...")
            video_with_audio = adjusted_video.set_audio(final_audio)
            
            # Step 6: Create advanced subtitles with Gurukul styling
            print(f"\n📝 Adding advanced subtitles...")
            subtitle_clips, srt_path = self.create_advanced_subtitles(
                sentences, audio_clips, adjusted_video.w, language='english'
            )

            if subtitle_clips:
                final_video = CompositeVideoClip([video_with_audio] + subtitle_clips)
                print(f"   ✅ Subtitles embedded in video")
            else:
                final_video = video_with_audio
                print(f"   ⚠️ No subtitles added")

            # Store SRT path for later use
            if srt_path:
                self.last_srt_path = srt_path
            
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

            # Also copy SRT file to storage for team access
            if hasattr(self, 'last_srt_path') and self.last_srt_path and os.path.exists(self.last_srt_path):
                srt_filename = output_filename.replace('.mp4', '.srt')
                storage_srt_path = os.path.join(storage_today_dir, srt_filename)
                shutil.copy2(self.last_srt_path, storage_srt_path)
                print(f"   📝 Subtitles copied to: {storage_srt_path}")

            print(f"   📁 Video copied to: {storage_path}")
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

            # =================================================================
            # TASK 10: Security Integration - Watermarking & Fingerprinting
            # =================================================================
            try:
                print(f"\n🔒 Applying security measures...")
                
                # Import security modules
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from security import embed_watermark, compute_fingerprint
                from security.visible_watermark import add_visible_watermark
                from audit_logger import get_audit_logger
                
                # Get BUILD_ID from environment or generate
                build_id = os.getenv('BUILD_ID', f'build_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                print(f"   🏷️ BUILD_ID: {build_id}")
                
                # Step 1: Add invisible watermark (metadata-based)
                print(f"   💧 Adding invisible watermark...")
                watermarked_invisible = embed_watermark(
                    storage_path,  # Watermark the storage version
                    build_id=build_id,
                    output_path=storage_path.replace('.mp4', '_watermarked_temp.mp4')
                )
                
                # Step 2: Add visible logo watermark (subtle, production mode)
                print(f"   🎨 Adding visible logo watermark...")
                watermarked_final = add_visible_watermark(
                    watermarked_invisible,
                    style="subtle",  # 35% opacity, visible but professional
                    build_id=build_id
                )
                
                # Replace original with watermarked version
                if os.path.exists(watermarked_final):
                    # IMPORTANT: OpenCV watermarking strips audio, so we need to restore it!
                    print(f"   🎵 Restoring audio from original video...")
                    
                    # Re-encode to H.264 with audio from original video
                    print(f"   🔄 Re-encoding to H.264 for compatibility...")
                    h264_output = storage_path.replace('.mp4', '_h264_temp.mp4')
                    
                    import subprocess
                    try:
                        # Use FFmpeg to:
                        # 1. Take video from watermarked file (no audio)
                        # 2. Take audio from original storage file (with audio)
                        # 3. Combine them with H.264 encoding
                        ffmpeg_cmd = [
                            'ffmpeg', '-y',
                            '-i', watermarked_final,  # Video input (watermarked, no audio)
                            '-i', storage_path,        # Audio input (original with audio)
                            '-map', '0:v:0',          # Take video from first input
                            '-map', '1:a:0?',         # Take audio from second input (? = optional)
                            '-c:v', 'libx264',        # H.264 video codec
                            '-c:a', 'aac',            # AAC audio codec
                            '-b:a', '192k',           # Audio bitrate
                            '-preset', 'medium',      # Balance speed/quality
                            '-crf', '23',             # Quality (lower = better, 23 is good)
                            '-pix_fmt', 'yuv420p',    # Compatibility
                            '-movflags', '+faststart', # Web streaming optimization
                            '-shortest',              # Match shortest stream duration
                            h264_output
                        ]
                        
                        result = subprocess.run(
                            ffmpeg_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=300  # 5 minute timeout
                        )
                        
                        if result.returncode == 0 and os.path.exists(h264_output):
                            # Success! Use H.264 version
                            shutil.copy2(h264_output, storage_path)
                            shutil.copy2(h264_output, output_path)
                            os.remove(h264_output)
                            print(f"   ✅ Re-encoded to H.264 successfully")
                        else:
                            # FFmpeg failed, use original watermarked version
                            print(f"   ⚠️ H.264 encoding failed, using mp4v codec")
                            shutil.copy2(watermarked_final, storage_path)
                            shutil.copy2(watermarked_final, output_path)
                    
                    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                        print(f"   ⚠️ FFmpeg not available or timeout: {e}")
                        print(f"   ℹ️ Using mp4v codec (may not play in VS Code)")
                        shutil.copy2(watermarked_final, storage_path)
                        shutil.copy2(watermarked_final, output_path)
                    
                    # Clean up temp files
                    if os.path.exists(watermarked_invisible):
                        os.remove(watermarked_invisible)
                    if watermarked_final != storage_path and os.path.exists(watermarked_final):
                        os.remove(watermarked_final)
                    
                    print(f"   ✅ Watermarks applied successfully")
                else:
                    print(f"   ⚠️ Watermarking failed, using original video")
                
                # Step 3: Compute content fingerprint
                print(f"   🔍 Computing content fingerprint...")
                fingerprint = compute_fingerprint(storage_path, build_id=build_id)
                
                print(f"   ✅ Fingerprint: {fingerprint['sha256'][:16]}...")
                print(f"   ✅ Security measures applied")
                
                # Store fingerprint info with metadata
                fingerprint_file = storage_path.replace('.mp4', '_fingerprint.json')
                with open(fingerprint_file, 'w') as f:
                    json.dump(fingerprint, f, indent=2)
                
                # Step 4: Log to audit trail with security metadata
                print(f"   📝 Logging to audit trail...")
                audit_logger = get_audit_logger()
                
                # Create KSML token (if available from request context)
                ksml_token_data = {
                    "ksml_token": os.getenv('KSML_TOKEN', 'ksml_production'),
                    "intent": "video_generation",
                    "karma_state": "authorized",
                    "lineage": {
                        "lesson": lesson_title,
                        "style": style,
                        "build_id": build_id
                    }
                }
                
                # Log video generation with security metadata
                audit_logger.log_video_generation(
                    prompt=lesson_data.get('text', '')[:200],  # First 200 chars
                    output_path=storage_path,
                    ksml_token=ksml_token_data,
                    quality_metrics={
                        "duration": audio_duration,
                        "clips": len(video_clips),
                        "style": style
                    },
                    security_metadata={
                        "build_id": build_id,
                        "artifact_hash": fingerprint['sha256'],
                        "watermark_id": build_id,
                        "signed": False,  # Will be True after CI signs
                        "watermark_method": "dual_layer",  # invisible + visible
                        "fingerprint_method": "sha256+blake2b+perceptual"
                    }
                )
                
                print(f"   ✅ Audit log created")
                
            except Exception as security_error:
                print(f"   ⚠️ Security integration warning: {security_error}")
                print(f"   📝 Video saved without watermarks (module may not be available)")
                import traceback
                traceback.print_exc()
            # =================================================================
            # END: Security Integration
            # =================================================================

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

    def _extract_scene_contexts(self, lesson_data: dict) -> List[str]:
        """Extract scene contexts from lesson data for cinematic flow"""

        try:
            # Check if prompts contain scene information
            prompts = lesson_data.get('prompts', [])
            scenes = []

            # Scene detection keywords
            scene_keywords = {
                'temple': ['temple', 'shrine', 'sacred', 'prayer', 'worship', 'divine'],
                'forest': ['forest', 'tree', 'nature', 'woods', 'jungle', 'green'],
                'cosmic': ['cosmic', 'universe', 'space', 'stars', 'celestial', 'ethereal'],
                'mountain': ['mountain', 'peak', 'summit', 'hill', 'cliff', 'high'],
                'river': ['river', 'water', 'stream', 'flow', 'lake', 'ocean'],
                'palace': ['palace', 'royal', 'grand', 'majestic', 'golden', 'throne']
            }

            for prompt in prompts:
                prompt_lower = prompt.lower()
                detected_scene = 'temple'  # Default

                # Find best matching scene
                max_matches = 0
                for scene, keywords in scene_keywords.items():
                    matches = sum(1 for keyword in keywords if keyword in prompt_lower)
                    if matches > max_matches:
                        max_matches = matches
                        detected_scene = scene

                scenes.append(detected_scene)

            # If no prompts, use metadata or defaults
            if not scenes:
                metadata = lesson_data.get('metadata', {})
                segments = metadata.get('segments', [])

                for segment in segments:
                    scene = segment.get('scene', 'temple')
                    scenes.append(scene)

            # Ensure we have at least one scene
            if not scenes:
                scenes = ['temple']

            print(f"   🎭 Detected scenes: {scenes}")
            return scenes

        except Exception as e:
            print(f"   ⚠️ Scene extraction failed: {e}")
            return ['temple']  # Safe default

    def _generate_flow_instructions(self, lesson_data: dict, num_clips: int) -> List[Dict]:
        """Generate cinematic flow instructions for each clip"""

        try:
            # Base flow patterns for educational content
            flow_patterns = [
                {'movement': 'pan_right', 'intensity': 0.3, 'description': 'Gentle introduction'},
                {'movement': 'zoom_in', 'intensity': 0.4, 'description': 'Focus attention'},
                {'movement': 'orbit', 'intensity': 0.3, 'description': 'Dynamic perspective'},
                {'movement': 'dolly', 'intensity': 0.2, 'description': 'Depth movement'},
                {'movement': 'pan_left', 'intensity': 0.3, 'description': 'Smooth transition'},
                {'movement': 'tilt_up', 'intensity': 0.2, 'description': 'Uplifting motion'},
                {'movement': 'zoom_out', 'intensity': 0.3, 'description': 'Revealing view'},
                {'movement': 'pan_right', 'intensity': 0.2, 'description': 'Concluding sweep'}
            ]

            # Generate instructions for each clip
            instructions = []
            for i in range(num_clips):
                pattern = flow_patterns[i % len(flow_patterns)]

                # Adjust intensity based on clip position
                if i == 0:
                    # First clip - gentle introduction
                    pattern['intensity'] = min(pattern['intensity'], 0.2)
                elif i == num_clips - 1:
                    # Last clip - strong conclusion
                    pattern['intensity'] = max(pattern['intensity'], 0.4)

                instructions.append(pattern.copy())

            print(f"   🎬 Generated flow instructions: {[inst['movement'] for inst in instructions]}")
            return instructions

        except Exception as e:
            print(f"   ⚠️ Flow instruction generation failed: {e}")
            # Safe default
            return [{'movement': 'pan_right', 'intensity': 0.2}] * num_clips

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
            
            # Save performance metrics
            try:
                from performance_tracker import performance_tracker
                metrics_file = performance_tracker.save_metrics()
                print(f"📊 Performance metrics saved: {metrics_file}")
            except Exception as e:
                print(f"⚠️ Could not save metrics: {e}")
        else:
            print(f"\n❌ Video generation failed")

    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()
