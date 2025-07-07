#!/usr/bin/env python3
"""
Multi-Layer Audio Processing Pipeline
Combines background narration with character dialogue and manages audio mixing
"""

import os
import sys
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

# Import our custom modules
from prompt_enhancer import PromptEnhancer, EnhancedPrompt
from multi_voice_tts import MultiVoiceTTS
from character_detector import CharacterDetector, DetectedCharacter
from sadtalker_integration import SadTalkerIntegration, LipSyncResult

@dataclass
class AudioLayer:
    """Represents an audio layer in the mix"""
    audio_path: str
    start_time: float  # Start time in seconds
    duration: float    # Duration in seconds
    volume: float      # Volume level (0.0 to 1.0)
    layer_type: str    # 'narration' or 'dialogue'
    character_info: Optional[Dict] = None

@dataclass
class ProcessingResult:
    """Result of audio-video processing"""
    success: bool
    final_video_path: str
    audio_layers: List[AudioLayer]
    processing_time: float
    error_message: str = ""

class MultiLayerAudioProcessor:
    """Multi-layer audio processing and mixing system"""

    def __init__(self):
        """Initialize the multi-layer audio processor"""
        self.temp_dir = tempfile.mkdtemp(prefix="audioprocessor_")
        self.prompt_enhancer = PromptEnhancer()
        self.multi_voice_tts = MultiVoiceTTS()
        self.character_detector = CharacterDetector()
        self.sadtalker_integration = SadTalkerIntegration()

        print(f"✅ Multi-Layer Audio Processor initialized")
        print(f"📁 Temp dir: {self.temp_dir}")

    def analyze_prompts_for_audio_layers(self, enhanced_prompts: List[EnhancedPrompt]) -> Dict:
        """Analyze enhanced prompts to determine audio layer structure"""
        print(f"🔍 Analyzing {len(enhanced_prompts)} prompts for audio layers...")

        audio_plan = {
            "background_narration": [],
            "character_dialogues": [],
            "total_estimated_duration": 0.0
        }

        current_time = 0.0

        for i, prompt in enumerate(enhanced_prompts):
            # Estimate duration based on text length (rough approximation)
            estimated_duration = len(prompt.audio_prompt.split()) * 0.5  # ~0.5s per word

            # Background narration (always present)
            audio_plan["background_narration"].append({
                "clip_index": i,
                "text": prompt.audio_prompt,
                "start_time": current_time,
                "duration": estimated_duration,
                "volume": 0.6  # Background volume
            })

            # Character dialogue (if present)
            if prompt.has_dialogue and prompt.dialogue_text:
                dialogue_duration = len(prompt.dialogue_text.split()) * 0.6  # Slightly slower for dialogue

                audio_plan["character_dialogues"].append({
                    "clip_index": i,
                    "text": prompt.dialogue_text,
                    "character_gender": prompt.character_gender,
                    "start_time": current_time + estimated_duration * 0.7,  # Start during narration
                    "duration": dialogue_duration,
                    "volume": 1.0  # Full volume for dialogue
                })

            current_time += estimated_duration

        audio_plan["total_estimated_duration"] = current_time

        print(f"📊 Audio Plan Summary:")
        print(f"   Background narration clips: {len(audio_plan['background_narration'])}")
        print(f"   Character dialogue clips: {len(audio_plan['character_dialogues'])}")
        print(f"   Total estimated duration: {audio_plan['total_estimated_duration']:.2f}s")

        return audio_plan

    def generate_background_narration(self, narration_clips: List[Dict]) -> List[AudioLayer]:
        """Generate background narration audio"""
        print(f"🎙️ Generating background narration for {len(narration_clips)} clips...")

        narration_layers = []

        for clip in narration_clips:
            print(f"🎵 Generating narration for clip {clip['clip_index']}: {clip['text'][:50]}...")

            try:
                # Generate narration audio (female narrator by default)
                audio_path = self.multi_voice_tts.generate_narration_audio(
                    clip['text'],
                    narrator_gender="female"
                )

                # Get actual duration
                actual_duration = self.multi_voice_tts.get_audio_duration(audio_path)

                # Truncate audio to match clip duration if it's too long
                target_duration = clip.get('duration', 1.33)  # Default to 1.33s per clip
                if actual_duration > target_duration:
                    print(f"🔧 Truncating audio from {actual_duration:.2f}s to {target_duration:.2f}s")
                    truncated_path = self.truncate_audio(audio_path, target_duration)
                    if truncated_path:
                        audio_path = truncated_path
                        actual_duration = target_duration

                layer = AudioLayer(
                    audio_path=audio_path,
                    start_time=clip['start_time'],
                    duration=actual_duration,
                    volume=clip['volume'],
                    layer_type='narration'
                )

                narration_layers.append(layer)
                print(f"✅ Generated narration: {actual_duration:.2f}s")

            except Exception as e:
                print(f"❌ Error generating narration for clip {clip['clip_index']}: {e}")
                continue

        print(f"✅ Generated {len(narration_layers)} narration layers")
        return narration_layers

    def truncate_audio(self, audio_path: str, target_duration: float) -> str:
        """Truncate audio file to target duration"""
        try:
            truncated_path = os.path.join(self.temp_dir, f"truncated_{os.path.basename(audio_path)}")

            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-t", str(target_duration),  # Truncate to target duration
                "-c", "copy",  # Copy without re-encoding for speed
                truncated_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return truncated_path

        except Exception as e:
            print(f"⚠️ Error truncating audio: {e}")
            return audio_path  # Return original if truncation fails

    def generate_character_dialogue(self, dialogue_clips: List[Dict]) -> List[AudioLayer]:
        """Generate character dialogue audio"""
        print(f"🗣️ Generating character dialogue for {len(dialogue_clips)} clips...")

        dialogue_layers = []

        for clip in dialogue_clips:
            print(f"🎵 Generating dialogue for clip {clip['clip_index']}: {clip['text'][:50]}...")

            try:
                # Generate character dialogue audio
                audio_path = self.multi_voice_tts.generate_character_audio(
                    clip['text'],
                    clip['character_gender']
                )

                # Get actual duration
                actual_duration = self.multi_voice_tts.get_audio_duration(audio_path)

                layer = AudioLayer(
                    audio_path=audio_path,
                    start_time=clip['start_time'],
                    duration=actual_duration,
                    volume=clip['volume'],
                    layer_type='dialogue',
                    character_info={
                        'gender': clip['character_gender'],
                        'clip_index': clip['clip_index']
                    }
                )

                dialogue_layers.append(layer)
                print(f"✅ Generated dialogue ({clip['character_gender']}): {actual_duration:.2f}s")

            except Exception as e:
                print(f"❌ Error generating dialogue for clip {clip['clip_index']}: {e}")
                continue

        print(f"✅ Generated {len(dialogue_layers)} dialogue layers")
        return dialogue_layers

    def mix_audio_layers(self, audio_layers: List[AudioLayer], total_duration: float) -> str:
        """Mix multiple audio layers into a single track"""
        print(f"🎛️ Mixing {len(audio_layers)} audio layers...")

        if not audio_layers:
            print("⚠️ No audio layers to mix")
            return None

        # Create output path
        mixed_audio_path = os.path.join(self.temp_dir, "mixed_audio.wav")

        try:
            # Build ffmpeg command for audio mixing
            cmd = ["ffmpeg", "-y"]

            # Add input files
            for layer in audio_layers:
                cmd.extend(["-i", layer.audio_path])

            # Build filter complex for mixing
            filter_parts = []
            for i, layer in enumerate(audio_layers):
                # Apply volume and delay
                volume_filter = f"[{i}:a]volume={layer.volume}"
                if layer.start_time > 0:
                    volume_filter += f",adelay={int(layer.start_time * 1000)}|{int(layer.start_time * 1000)}"
                filter_parts.append(f"{volume_filter}[a{i}]")

            # Mix all processed streams
            mix_inputs = "".join([f"[a{i}]" for i in range(len(audio_layers))])
            filter_complex = ";".join(filter_parts) + f";{mix_inputs}amix=inputs={len(audio_layers)}:duration=longest[out]"

            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[out]"])
            cmd.extend(["-t", str(total_duration)])  # Set total duration
            cmd.append(mixed_audio_path)

            print(f"🔧 Mixing command: {' '.join(cmd[:10])}...")  # Show first part of command

            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            print(f"✅ Audio mixing completed: {os.path.basename(mixed_audio_path)}")
            return mixed_audio_path

        except subprocess.CalledProcessError as e:
            print(f"❌ Audio mixing failed: {e.stderr}")
            # Fallback: use first audio layer
            if audio_layers:
                print("🔄 Using fallback: first audio layer only")
                return audio_layers[0].audio_path
            return None

        except Exception as e:
            print(f"❌ Unexpected error in audio mixing: {e}")
            return None

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning up: {e}")

        # Cleanup individual processors
        self.multi_voice_tts.cleanup()
        self.character_detector.cleanup()
        self.sadtalker_integration.cleanup()

def test_multi_layer_audio_processor():
    """Test the multi-layer audio processor"""
    print("🧪 Testing Multi-Layer Audio Processor...")

    processor = MultiLayerAudioProcessor()

    # Test with sample prompts
    test_prompts = [
        "A young man walks down a quiet street in the rain.",
        "He stops and thinks, 'I need to find shelter soon.'",
        "The rain falls harder as he hurries toward a building.",
    ]

    try:
        # Enhance prompts
        enhanced_prompts = processor.prompt_enhancer.process_prompt_list(test_prompts)

        # Analyze for audio layers
        audio_plan = processor.analyze_prompts_for_audio_layers(enhanced_prompts)

        # Generate background narration
        narration_layers = processor.generate_background_narration(
            audio_plan["background_narration"]
        )

        # Generate character dialogue
        dialogue_layers = processor.generate_character_dialogue(
            audio_plan["character_dialogues"]
        )

        # Mix audio layers
        all_layers = narration_layers + dialogue_layers
        if all_layers:
            mixed_audio = processor.mix_audio_layers(
                all_layers,
                audio_plan["total_estimated_duration"]
            )

            if mixed_audio:
                print(f"🎉 Multi-layer audio processing successful!")
                print(f"🎵 Mixed audio: {mixed_audio}")
            else:
                print("❌ Audio mixing failed")
        else:
            print("❌ No audio layers generated")

    except Exception as e:
        print(f"❌ Error in multi-layer audio processing: {e}")

    finally:
        processor.cleanup()

if __name__ == "__main__":
    test_multi_layer_audio_processor()