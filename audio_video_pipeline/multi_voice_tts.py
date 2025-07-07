#!/usr/bin/env python3
"""
Multi-Voice TTS System
Generates different voices for narration vs character dialogue with gender-based selection
"""

import os
import sys
import subprocess
import tempfile
from typing import List, Dict, Tuple
from gtts import gTTS
from dataclasses import dataclass

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tts_module'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SadTalker', 'src', 'utils'))

# Try to import pydub, fallback to basic functionality if not available
try:
    from pydub import AudioSegment
    from pydub.effects import speedup, low_pass_filter, high_pass_filter
    PYDUB_AVAILABLE = True
    print("✅ PyDub available - full audio processing enabled")
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ PyDub not available - using basic audio processing")

@dataclass
class VoiceConfig:
    """Configuration for different voice types"""
    name: str
    language: str
    speed: float  # Speed multiplier (1.0 = normal)
    pitch_shift: int  # Semitones to shift pitch (-12 to +12)
    volume: float  # Volume multiplier (1.0 = normal)
    effects: List[str]  # Audio effects to apply

class MultiVoiceTTS:
    """Multi-voice TTS system with gender-based voice assignment"""
    
    def __init__(self):
        """Initialize the multi-voice TTS system"""
        self.voice_configs = self._setup_voice_configs()
        self.temp_dir = tempfile.mkdtemp(prefix="multitts_")
        print(f"✅ Multi-Voice TTS initialized. Temp dir: {self.temp_dir}")
    
    def _setup_voice_configs(self) -> Dict[str, VoiceConfig]:
        """Setup different voice configurations"""
        return {
            "narrator_female": VoiceConfig(
                name="narrator_female",
                language="en",
                speed=0.9,  # Slightly slower for narration
                pitch_shift=2,  # Slightly higher pitch
                volume=0.8,  # Softer for background
                effects=["warm", "clear"]
            ),
            "narrator_male": VoiceConfig(
                name="narrator_male", 
                language="en",
                speed=0.85,  # Slower, more authoritative
                pitch_shift=-3,  # Lower pitch
                volume=0.8,  # Softer for background
                effects=["deep", "clear"]
            ),
            "character_male": VoiceConfig(
                name="character_male",
                language="en", 
                speed=1.0,  # Normal speed
                pitch_shift=-2,  # Slightly lower
                volume=1.0,  # Full volume for dialogue
                effects=["clear", "present"]
            ),
            "character_female": VoiceConfig(
                name="character_female",
                language="en",
                speed=1.05,  # Slightly faster
                pitch_shift=4,  # Higher pitch
                volume=1.0,  # Full volume for dialogue
                effects=["bright", "clear"]
            )
        }
    
    def generate_base_audio(self, text: str, language: str = "en") -> str:
        """Generate base audio using gTTS"""
        temp_file = os.path.join(self.temp_dir, f"base_{hash(text)}.mp3")
        
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_file)
            print(f"✅ Generated base audio: {os.path.basename(temp_file)}")
            return temp_file
        except Exception as e:
            print(f"❌ Error generating base audio: {e}")
            raise
    
    def apply_voice_effects(self, audio_path: str, voice_config: VoiceConfig) -> str:
        """Apply voice effects based on configuration"""
        print(f"🎛️ Applying {voice_config.name} effects...")

        if not PYDUB_AVAILABLE:
            # Fallback: use ffmpeg for basic processing
            return self._apply_effects_with_ffmpeg(audio_path, voice_config)

        # Load audio with pydub
        audio = AudioSegment.from_mp3(audio_path)

        # Apply speed change
        if voice_config.speed != 1.0:
            audio = speedup(audio, playback_speed=voice_config.speed)

        # Apply volume change
        if voice_config.volume != 1.0:
            volume_change = 20 * (voice_config.volume - 1.0)  # Convert to dB
            audio = audio + volume_change

        # Apply pitch shift (simplified - using speed change as approximation)
        if voice_config.pitch_shift != 0:
            # Rough pitch shift using speed change
            pitch_factor = 2 ** (voice_config.pitch_shift / 12.0)
            if pitch_factor != 1.0:
                audio = speedup(audio, playback_speed=pitch_factor)

        # Apply audio effects
        for effect in voice_config.effects:
            audio = self._apply_audio_effect(audio, effect)

        # Save processed audio
        output_path = audio_path.replace(".mp3", f"_{voice_config.name}.mp3")
        audio.export(output_path, format="mp3")

        print(f"✅ Applied effects: {output_path}")
        return output_path

    def _apply_effects_with_ffmpeg(self, audio_path: str, voice_config: VoiceConfig) -> str:
        """Fallback method using ffmpeg for audio effects"""
        output_path = audio_path.replace(".mp3", f"_{voice_config.name}.mp3")

        # Build ffmpeg command for basic effects
        cmd = ["ffmpeg", "-y", "-i", audio_path]

        # Apply speed change
        if voice_config.speed != 1.0:
            cmd.extend(["-filter:a", f"atempo={voice_config.speed}"])

        # Apply volume change
        if voice_config.volume != 1.0:
            volume_db = 20 * (voice_config.volume - 1.0)
            cmd.extend(["-filter:a", f"volume={volume_db}dB"])

        cmd.append(output_path)

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Applied basic effects with ffmpeg: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"⚠️ ffmpeg effects failed, using original: {e}")
            return audio_path
    
    def _apply_audio_effect(self, audio, effect: str):
        """Apply specific audio effects"""
        if not PYDUB_AVAILABLE:
            return audio

        if effect == "warm":
            # Warm effect - slight low-pass filter
            return low_pass_filter(audio, cutoff=8000)
        elif effect == "bright":
            # Bright effect - slight high-pass filter
            return high_pass_filter(audio, cutoff=100)
        elif effect == "deep":
            # Deep effect - boost low frequencies
            return audio.low_pass_filter(6000)
        elif effect == "clear":
            # Clear effect - slight compression (simplified)
            return audio.normalize()
        elif effect == "present":
            # Present effect - boost mid frequencies
            return audio.normalize()
        else:
            return audio
    
    def convert_to_wav(self, mp3_path: str) -> str:
        """Convert MP3 to WAV for compatibility"""
        wav_path = mp3_path.replace(".mp3", ".wav")
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", mp3_path, 
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",      # Mono
                wav_path
            ], check=True, capture_output=True)
            
            print(f"✅ Converted to WAV: {os.path.basename(wav_path)}")
            return wav_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting to WAV: {e}")
            raise
    
    def generate_narration_audio(self, text: str, narrator_gender: str = "female") -> str:
        """Generate narration audio with appropriate voice"""
        voice_key = f"narrator_{narrator_gender}"
        voice_config = self.voice_configs[voice_key]
        
        print(f"🎙️ Generating narration ({narrator_gender}): {text[:50]}...")
        
        # Generate base audio
        base_audio = self.generate_base_audio(text, voice_config.language)
        
        # Apply voice effects
        processed_audio = self.apply_voice_effects(base_audio, voice_config)
        
        # Convert to WAV
        wav_audio = self.convert_to_wav(processed_audio)
        
        return wav_audio
    
    def generate_character_audio(self, text: str, character_gender: str) -> str:
        """Generate character dialogue audio"""
        # Handle neutral gender by defaulting to male
        if character_gender == "neutral":
            character_gender = "male"

        voice_key = f"character_{character_gender}"
        if voice_key not in self.voice_configs:
            print(f"⚠️ Unknown character gender '{character_gender}', using male")
            voice_key = "character_male"

        voice_config = self.voice_configs[voice_key]
        
        print(f"🗣️ Generating character dialogue ({character_gender}): {text[:50]}...")
        
        # Generate base audio
        base_audio = self.generate_base_audio(text, voice_config.language)
        
        # Apply voice effects
        processed_audio = self.apply_voice_effects(base_audio, voice_config)
        
        # Convert to WAV
        wav_audio = self.convert_to_wav(processed_audio)
        
        return wav_audio
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0  # Convert ms to seconds
            else:
                # Fallback using ffprobe
                cmd = ["ffprobe", "-v", "quiet", "-show_entries",
                       "format=duration", "-of", "csv=p=0", audio_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return float(result.stdout.strip())
        except Exception as e:
            print(f"⚠️ Error getting audio duration: {e}")
            return 0.0
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning up: {e}")

def test_multi_voice_tts():
    """Test the multi-voice TTS system"""
    print("🧪 Testing Multi-Voice TTS System...")
    
    tts = MultiVoiceTTS()
    
    # Test different voice types
    test_cases = [
        ("This is a story about a young man walking through the rain.", "narration", "female"),
        ("I need to find shelter from this storm.", "character", "male"),
        ("The girl looked up at the sky with wonder.", "narration", "female"),
        ("What a beautiful day this is!", "character", "female"),
    ]
    
    results = []
    
    for text, voice_type, gender in test_cases:
        print(f"\n🎵 Testing {voice_type} ({gender}):")
        print(f"Text: {text}")
        
        try:
            if voice_type == "narration":
                audio_path = tts.generate_narration_audio(text, gender)
            else:
                audio_path = tts.generate_character_audio(text, gender)
            
            duration = tts.get_audio_duration(audio_path)
            results.append((voice_type, gender, audio_path, duration))
            
            print(f"✅ Generated: {os.path.basename(audio_path)} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print("📊 MULTI-VOICE TTS RESULTS:")
    print(f"{'='*60}")
    
    for voice_type, gender, path, duration in results:
        print(f"🎵 {voice_type.title()} ({gender}): {os.path.basename(path)} - {duration:.2f}s")
    
    # Cleanup
    tts.cleanup()

if __name__ == "__main__":
    test_multi_voice_tts()
