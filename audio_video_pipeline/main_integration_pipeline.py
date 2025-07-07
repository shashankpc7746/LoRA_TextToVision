#!/usr/bin/env python3
"""
Main Audio-Video Integration Pipeline
Complete integration system that combines AnimateDiff videos with enhanced audio
"""

import os
import sys
import subprocess
import tempfile
import time
import shutil
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add AnimateDiff to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AnimateDiff'))

# Import our modules
from prompt_enhancer import PromptEnhancer, EnhancedPrompt
from multi_voice_tts import MultiVoiceTTS
from character_detector import CharacterDetector, DetectedCharacter
from sadtalker_integration import SadTalkerIntegration, LipSyncResult
from glue_pipeline import MultiLayerAudioProcessor, AudioLayer, ProcessingResult

@dataclass
class IntegrationConfig:
    """Configuration for the integration pipeline"""
    video_input_path: str = None  # Path to existing video or None to generate
    prompts: List[str] = None     # Text prompts for generation
    output_dir: str = "final_outputs"
    narrator_gender: str = "female"
    apply_lipsync: bool = True
    enhance_prompts: bool = True
    generate_video: bool = True

class MainIntegrationPipeline:
    """Main pipeline for complete audio-video integration"""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize the main integration pipeline"""
        self.config = config
        self.temp_dir = tempfile.mkdtemp(prefix="integration_")
        
        # Initialize processors
        self.audio_processor = MultiLayerAudioProcessor()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        print(f"🚀 Main Integration Pipeline initialized")
        print(f"📁 Output dir: {self.config.output_dir}")
        print(f"📁 Temp dir: {self.temp_dir}")
    
    def generate_video_from_prompts(self, prompts: List[str]) -> str:
        """Generate video using AnimateDiff multi-clip generator"""
        print(f"🎬 Generating video from {len(prompts)} prompts...")

        try:
            # Change to AnimateDiff directory and run multi_clip_generator
            import subprocess
            import sys

            # Create temporary prompts file in AnimateDiff directory
            animatediff_dir = os.path.join(os.path.dirname(__file__), '..', 'AnimateDiff')
            prompts_text = "\n".join(prompts)
            temp_prompts_file = os.path.join(animatediff_dir, "temp_prompts.txt")

            with open(temp_prompts_file, 'w', encoding='utf-8') as f:
                f.write(prompts_text)

            print(f"📝 Created prompts file: {temp_prompts_file}")

            # Run multi_clip_generator.py with the prompts
            cmd = [sys.executable, "multi_clip_generator.py"]

            print(f"🔧 Running: {' '.join(cmd)} in {animatediff_dir}")

            # Temporarily modify multi_clip_generator.py to use our prompts
            original_file = os.path.join(animatediff_dir, "multi_clip_generator.py")
            backup_file = os.path.join(animatediff_dir, "multi_clip_generator_backup.py")

            # Backup original file
            shutil.copy2(original_file, backup_file)

            # Read original content
            with open(original_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace the paragraph with our prompts
            new_content = content.replace(
                'paragraph = """',
                f'paragraph = """{prompts_text}'
            )

            # Write modified content
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            try:
                # Run the generator
                result = subprocess.run(
                    cmd,
                    cwd=animatediff_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )

                print(f"✅ Video generation completed")

                # Find the generated video
                outputs_dir = os.path.join(animatediff_dir, "outputs")
                final_video = os.path.join(outputs_dir, "final_output_stitched.mp4")

                if os.path.exists(final_video):
                    print(f"✅ Video generated: {os.path.basename(final_video)}")
                    return final_video
                else:
                    raise Exception("Generated video not found")

            finally:
                # Restore original file
                shutil.copy2(backup_file, original_file)
                os.remove(backup_file)
                if os.path.exists(temp_prompts_file):
                    os.remove(temp_prompts_file)

        except Exception as e:
            print(f"❌ Error generating video: {e}")
            raise

    def get_video_duration(self, video_path: str) -> float:
        """Get the actual duration of a video file"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            if fps > 0:
                duration = frame_count / fps
                print(f"📊 Video specs: {frame_count} frames @ {fps} fps = {duration:.2f}s")
                return duration
            else:
                # Fallback using moviepy
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(video_path)
                duration = clip.duration
                clip.close()
                print(f"📊 Video duration (moviepy): {duration:.2f}s")
                return duration

        except Exception as e:
            print(f"⚠️ Error getting video duration: {e}")
            # Estimate based on AnimateDiff specs: 32 frames @ 24 fps per clip
            estimated_duration = len(self.config.prompts) * (32 / 24)
            print(f"📊 Using estimated duration: {estimated_duration:.2f}s")
            return estimated_duration
    
    def combine_video_with_audio(self, video_path: str, audio_path: str) -> str:
        """Combine video with mixed audio track"""
        print(f"🎞️ Combining video with audio...")
        print(f"🎬 Video: {os.path.basename(video_path)}")
        print(f"🎵 Audio: {os.path.basename(audio_path)}")
        
        # Create output path
        timestamp = int(time.time())
        output_filename = f"final_video_{timestamp}.mp4"
        output_path = os.path.join(self.config.output_dir, output_filename)
        
        try:
            # Use ffmpeg to combine video and audio
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,      # Video input
                "-i", audio_path,      # Audio input
                "-c:v", "copy",        # Copy video codec (no re-encoding)
                "-c:a", "aac",         # Audio codec
                "-map", "0:v:0",       # Map video from first input
                "-map", "1:a:0",       # Map audio from second input
                "-shortest",           # End when shortest stream ends
                output_path
            ]
            
            print(f"🔧 Combining: {' '.join(cmd[:8])}...")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ Video-audio combination successful!")
            print(f"📹 Final video: {output_filename}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Video-audio combination failed: {e.stderr}")
            raise
        
        except Exception as e:
            print(f"❌ Unexpected error in video-audio combination: {e}")
            raise
    
    def apply_lipsync_to_video(self, video_path: str, dialogue_layers: List[AudioLayer]) -> str:
        """Apply lip-sync to characters in video when they have dialogue"""
        if not dialogue_layers:
            print("ℹ️ No dialogue layers - skipping lip-sync")
            return video_path
        
        print(f"👄 Applying lip-sync for {len(dialogue_layers)} dialogue clips...")
        
        try:
            # Detect characters in the video
            characters = self.audio_processor.character_detector.detect_characters_in_video(video_path)
            
            if not characters:
                print("⚠️ No characters detected in video - skipping lip-sync")
                return video_path
            
            # Get best character image for lip-sync
            best_character_image = self.audio_processor.character_detector.get_best_character_image(characters)
            
            if not best_character_image:
                print("⚠️ No suitable character image found - skipping lip-sync")
                return video_path
            
            # Apply lip-sync for each dialogue layer
            lipsync_results = []
            
            for i, dialogue_layer in enumerate(dialogue_layers):
                print(f"👄 Processing dialogue {i+1}/{len(dialogue_layers)}")
                
                result = self.audio_processor.sadtalker_integration.apply_lipsync_to_character(
                    best_character_image,
                    dialogue_layer.audio_path
                )
                
                if result.success:
                    lipsync_results.append(result)
                    print(f"✅ Lip-sync applied for dialogue {i+1}")
                else:
                    print(f"⚠️ Lip-sync failed for dialogue {i+1}: {result.error_message}")
            
            if lipsync_results:
                print(f"✅ Lip-sync processing completed for {len(lipsync_results)} clips")
                # For now, return original video - in advanced implementation,
                # you would composite the lip-sync results back into the original video
                return video_path
            else:
                print("⚠️ No successful lip-sync results")
                return video_path
                
        except Exception as e:
            print(f"❌ Error in lip-sync processing: {e}")
            return video_path  # Return original video on error
    
    def process_complete_pipeline(self) -> ProcessingResult:
        """Run the complete audio-video integration pipeline"""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print("🚀 STARTING COMPLETE AUDIO-VIDEO INTEGRATION PIPELINE")
        print(f"{'='*60}")
        
        try:
            # Step 1: Get or generate video
            if self.config.video_input_path and os.path.exists(self.config.video_input_path):
                print(f"📹 Using existing video: {self.config.video_input_path}")
                video_path = self.config.video_input_path
            elif self.config.prompts and self.config.generate_video:
                print(f"🎬 Generating new video from prompts...")
                video_path = self.generate_video_from_prompts(self.config.prompts)
            else:
                raise ValueError("No video input provided and video generation disabled")
            
            # Step 2: Enhance prompts for audio
            if self.config.enhance_prompts and self.config.prompts:
                print(f"\n📝 STEP 2: Enhancing prompts for audio...")
                enhanced_prompts = self.audio_processor.prompt_enhancer.process_prompt_list(self.config.prompts)
            else:
                # Create basic enhanced prompts from original prompts
                enhanced_prompts = []
                for prompt in self.config.prompts or []:
                    enhanced_prompts.append(EnhancedPrompt(
                        original=prompt,
                        video_prompt=prompt,
                        audio_prompt=prompt,
                        has_dialogue=False,
                        character_gender="neutral"
                    ))
            
            # Step 3: Get actual video duration for audio synchronization
            print(f"\n📊 STEP 3: Analyzing video for audio synchronization...")
            actual_video_duration = self.get_video_duration(video_path)

            # Step 4: Generate multi-layer audio synchronized to video duration
            print(f"\n🎵 STEP 4: Generating multi-layer audio (target: {actual_video_duration:.2f}s)...")
            audio_plan = self.audio_processor.analyze_prompts_for_audio_layers(enhanced_prompts)

            # Update audio plan with actual video duration
            audio_plan["total_estimated_duration"] = actual_video_duration

            # Adjust timing for each clip based on actual video duration
            clip_duration = actual_video_duration / len(enhanced_prompts)
            print(f"📊 Each clip duration: {clip_duration:.2f}s")

            # Update narration timing
            for i, narration_clip in enumerate(audio_plan["background_narration"]):
                narration_clip["start_time"] = i * clip_duration
                narration_clip["duration"] = clip_duration * 0.9  # Slightly shorter than clip

            # Update dialogue timing
            for dialogue_clip in audio_plan["character_dialogues"]:
                clip_index = dialogue_clip["clip_index"]
                dialogue_clip["start_time"] = clip_index * clip_duration + clip_duration * 0.3
                dialogue_clip["duration"] = min(dialogue_clip["duration"], clip_duration * 0.6)

            # Generate narration
            narration_layers = self.audio_processor.generate_background_narration(
                audio_plan["background_narration"]
            )

            # Generate dialogue
            dialogue_layers = self.audio_processor.generate_character_dialogue(
                audio_plan["character_dialogues"]
            )

            # Mix audio layers with exact video duration
            all_audio_layers = narration_layers + dialogue_layers
            mixed_audio_path = self.audio_processor.mix_audio_layers(
                all_audio_layers,
                actual_video_duration  # Use actual video duration
            )
            
            if not mixed_audio_path:
                raise Exception("Audio mixing failed")
            
            # Step 5: Apply lip-sync (optional)
            if self.config.apply_lipsync:
                print(f"\n👄 STEP 5: Applying lip-sync...")
                video_path = self.apply_lipsync_to_video(video_path, dialogue_layers)

            # Step 6: Combine final video with audio
            print(f"\n🎞️ STEP 6: Combining video with synchronized audio...")
            final_video_path = self.combine_video_with_audio(video_path, mixed_audio_path)
            
            processing_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"📹 Final video: {os.path.basename(final_video_path)}")
            print(f"⏱️ Total processing time: {processing_time:.2f}s")
            print(f"🎵 Audio layers: {len(all_audio_layers)} ({len(narration_layers)} narration, {len(dialogue_layers)} dialogue)")
            
            return ProcessingResult(
                success=True,
                final_video_path=final_video_path,
                audio_layers=all_audio_layers,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline failed: {str(e)}"
            
            print(f"\n{'='*60}")
            print("❌ PIPELINE FAILED!")
            print(f"{'='*60}")
            print(f"Error: {error_msg}")
            print(f"⏱️ Processing time: {processing_time:.2f}s")
            
            return ProcessingResult(
                success=False,
                final_video_path="",
                audio_layers=[],
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning up: {e}")
        
        # Cleanup audio processor
        self.audio_processor.cleanup()

def test_main_integration_pipeline():
    """Test the main integration pipeline"""
    print("🧪 Testing Main Integration Pipeline...")
    
    # Test configuration
    test_prompts = [
        "Anime boy wearing a hoodie walks on a quiet street under a grey sky.",
        "He stops and thinks, 'I need to find shelter from this rain.'",
        "Rain falls gently as he hurries toward a nearby building.",
    ]
    
    config = IntegrationConfig(
        prompts=test_prompts,
        output_dir="test_outputs",
        narrator_gender="female",
        apply_lipsync=False,  # Skip lip-sync for faster testing
        enhance_prompts=True,
        generate_video=False  # Skip video generation for testing
    )
    
    # Look for existing video to test with
    test_video_paths = [
        "../AnimateDiff/outputs",
        "../tts_module/results"
    ]
    
    for path in test_video_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.mp4'):
                    config.video_input_path = os.path.join(path, file)
                    print(f"📹 Using test video: {config.video_input_path}")
                    break
            if config.video_input_path:
                break
    
    if not config.video_input_path:
        print("⚠️ No test video found. Please generate a video first.")
        return
    
    # Run pipeline
    pipeline = MainIntegrationPipeline(config)
    
    try:
        result = pipeline.process_complete_pipeline()
        
        if result.success:
            print(f"\n🎉 Integration test successful!")
            print(f"📹 Final video: {result.final_video_path}")
        else:
            print(f"\n❌ Integration test failed: {result.error_message}")
            
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    test_main_integration_pipeline()
