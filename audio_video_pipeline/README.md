# Audio-Video Integration Pipeline

Complete system for integrating AnimateDiff videos with enhanced multi-layer audio including background narration, character dialogue, and lip-sync functionality.

## 🎯 Features

- **Enhanced Prompt Processing**: AI-powered conversion of video prompts to engaging story narration
- **Multi-Voice TTS System**: Gender-based voice assignment with different configurations for narration vs character dialogue
- **Character Detection**: OpenCV-based face detection and extraction from video frames
- **SadTalker Lip-Sync**: Integration with SadTalker for realistic lip-sync animation
- **Multi-Layer Audio**: Background narration combined with character dialogue
- **Complete Integration**: Full pipeline from text prompts to final video with synchronized audio

## 📁 Project Structure

```
audio_video_pipeline/
├── prompt_enhancer.py          # AI-powered prompt enhancement
├── multi_voice_tts.py          # Multi-voice TTS system
├── character_detector.py       # Character detection from videos
├── sadtalker_integration.py    # SadTalker lip-sync integration
├── glue_pipeline.py           # Multi-layer audio processing
├── main_integration_pipeline.py # Complete integration pipeline
├── results/                    # Generated results
├── test_outputs/              # Test outputs
└── README.md                  # This file
```

## 🚀 Quick Start

### Basic Usage

```python
from main_integration_pipeline import MainIntegrationPipeline, IntegrationConfig

# Define your prompts
prompts = [
    "Anime girl walks through a peaceful garden.",
    "She stops and says, 'What a beautiful day!'",
    "Birds chirp as she continues her walk."
]

# Configure the pipeline
config = IntegrationConfig(
    prompts=prompts,
    output_dir="my_outputs",
    narrator_gender="female",
    apply_lipsync=True,
    enhance_prompts=True,
    generate_video=True
)

# Run the pipeline
pipeline = MainIntegrationPipeline(config)
result = pipeline.process_complete_pipeline()

if result.success:
    print(f"✅ Success! Final video: {result.final_video_path}")
else:
    print(f"❌ Failed: {result.error_message}")

pipeline.cleanup()
```

### Using Existing Video

```python
config = IntegrationConfig(
    video_input_path="path/to/your/video.mp4",
    prompts=prompts,
    generate_video=False  # Use existing video
)
```

## 🔧 Individual Components

### 1. Prompt Enhancement

```python
from prompt_enhancer import PromptEnhancer

enhancer = PromptEnhancer()
enhanced = enhancer.process_prompt_list([
    "A girl walks in the park.",
    "She thinks, 'It's a nice day.'"
])

for prompt in enhanced:
    print(f"Original: {prompt.original}")
    print(f"Enhanced: {prompt.audio_prompt}")
    if prompt.has_dialogue:
        print(f"Dialogue: {prompt.dialogue_text}")
```

### 2. Multi-Voice TTS

```python
from multi_voice_tts import MultiVoiceTTS

tts = MultiVoiceTTS()

# Generate narration (female voice with effects)
narration_audio = tts.generate_narration_audio(
    "The sun was setting over the quiet town...",
    narrator_gender="female"
)

# Generate character dialogue (male voice)
dialogue_audio = tts.generate_character_audio(
    "I need to get home before dark.",
    character_gender="male"
)
```

### 3. Character Detection

```python
from character_detector import CharacterDetector

detector = CharacterDetector()
characters = detector.detect_characters_in_video("video.mp4")

for char in characters:
    print(f"Character: {char.gender} (confidence: {char.confidence:.2f})")
    print(f"Face image: {char.face_image_path}")
```

### 4. SadTalker Lip-Sync

```python
from sadtalker_integration import SadTalkerIntegration

sadtalker = SadTalkerIntegration()
result = sadtalker.apply_lipsync_to_character(
    character_image_path="face.jpg",
    dialogue_audio_path="dialogue.wav"
)

if result.success:
    print(f"Lip-sync video: {result.output_video_path}")
```

## ⚙️ Configuration Options

### IntegrationConfig Parameters

- `video_input_path`: Path to existing video (optional)
- `prompts`: List of text prompts for processing
- `output_dir`: Directory for final outputs (default: "final_outputs")
- `narrator_gender`: "female" or "male" for narration voice
- `apply_lipsync`: Enable/disable lip-sync processing
- `enhance_prompts`: Enable/disable AI prompt enhancement
- `generate_video`: Enable/disable video generation from prompts

### Voice Configuration

The system supports different voice configurations:

- **Narrator Female**: Slower, deeper voice for storytelling
- **Narrator Male**: Authoritative voice for narration
- **Character Female**: Natural female voice for dialogue
- **Character Male**: Natural male voice for dialogue

## 🎬 Pipeline Process

1. **Video Input**: Use existing video or generate from prompts
2. **Prompt Enhancement**: Convert video prompts to story format using AI
3. **Audio Generation**: Create background narration and character dialogue
4. **Character Detection**: Find characters in video for lip-sync
5. **Lip-Sync Application**: Apply SadTalker lip-sync to speaking characters
6. **Audio Mixing**: Combine narration and dialogue layers
7. **Final Combination**: Merge video with synchronized audio

## 📊 Performance

- **Prompt Enhancement**: ~2-3 seconds per prompt
- **Audio Generation**: ~5-10 seconds per audio clip
- **Character Detection**: ~3-5 seconds per video
- **Lip-Sync Processing**: ~10-15 seconds per character
- **Audio Mixing**: ~2-3 seconds
- **Video Combination**: ~1-2 seconds

**Total Processing Time**: ~30-60 seconds for 3-prompt sequence

## 🔍 Testing

Run individual component tests:

```bash
# Test prompt enhancement
python prompt_enhancer.py

# Test multi-voice TTS
python multi_voice_tts.py

# Test character detection
python character_detector.py

# Test SadTalker integration
python sadtalker_integration.py

# Test multi-layer audio processing
python glue_pipeline.py

# Test complete pipeline
python main_integration_pipeline.py
```

## 📋 Requirements

- Python 3.8+
- OpenCV (`cv2`)
- Google Generative AI (`google-generativeai`)
- gTTS (`gtts`)
- FFmpeg (system installation)
- SadTalker (in parent directory)
- AnimateDiff (in parent directory)

## 🎯 Use Cases

1. **Story Narration Videos**: Convert text stories into narrated videos
2. **Character Dialogue**: Add realistic dialogue to animated characters
3. **Educational Content**: Create engaging educational videos with narration
4. **Entertainment**: Generate story-driven animated content
5. **Presentations**: Add professional narration to visual presentations

## 🔧 Troubleshooting

### Common Issues

1. **"No audio generated"**: Check TTS service availability
2. **"Character detection failed"**: Ensure video has visible faces
3. **"SadTalker error"**: Verify SadTalker installation and checkpoints
4. **"FFmpeg not found"**: Install FFmpeg system-wide

### Debug Mode

Enable verbose logging by setting environment variable:
```bash
export AUDIO_VIDEO_DEBUG=1
```

## 🎉 Success Example

The system successfully processes prompts like:

**Input Prompts:**
```
Anime boy wearing a hoodie walks on a quiet street under a grey sky.
He stops and thinks, 'I need to find shelter from this rain.'
Rain falls gently as he hurries toward a nearby building.
```

**Output:**
- Enhanced story narration with atmospheric descriptions
- Character dialogue with male voice
- Background narration with female voice
- Mixed audio with proper timing
- Final video with synchronized audio

**Processing Time:** ~27 seconds
**Audio Layers:** 4 (3 narration, 1 dialogue)

## 📞 Support

For issues or questions, check the individual component files for detailed error messages and troubleshooting information.
