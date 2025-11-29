# Audio Manager Module

## Purpose
The Audio Manager module provides advanced audio synchronization and talking-head video generation using SadTalker. It generates realistic lip-synced videos from static images and audio, enabling creation of educational talking-head content for lessons and explanations.

## Key Components

### 1. **Enhanced SadTalker** (`enhanced_sadtalker.py`)
- Wrapper around SadTalker for simplified API
- Audio-driven facial animation generation
- Lip-sync accuracy optimization
- Expression and head pose control
- Batch processing for multiple audio clips

## Features

### Core Capabilities
- **Lip-Sync Generation**: Audio-driven mouth movement
- **Head Motion**: Natural head pose variation
- **Eye Blink**: Realistic blinking animations
- **Expression Control**: Emotional expression matching
- **High Quality**: 512x512+ output resolution
- **GPU Accelerated**: CUDA support for real-time generation

### Audio Processing
- **Format Support**: WAV, MP3, FLAC, OGG
- **Duration**: 1 second to 5 minutes per clip
- **Language Agnostic**: Works with any language audio
- **Background Music**: Optional BGM mixing
- **Noise Reduction**: Pre-processing for clean audio

### Video Generation
- **Input**: Static portrait image + audio file
- **Output**: Lip-synced talking-head video
- **FPS**: 25 FPS (configurable)
- **Resolution**: Up to 1024x1024
- **Quality Modes**: Fast (preview) / High (production)

## Usage

### Quick Start: Generate Talking Head
```python
from audio_manager.enhanced_sadtalker import EnhancedSadTalker

sadtalker = EnhancedSadTalker()
video_path = sadtalker.generate_talking_head(
    image_path="portraits/teacher.png",
    audio_path="audio/lesson_intro.wav",
    output_path="output/talking_head.mp4"
)
```

### Batch Processing
```python
# Generate multiple talking head clips
results = sadtalker.batch_generate(
    image_paths=["teacher1.png", "teacher2.png"],
    audio_paths=["lesson1.wav", "lesson2.wav"],
    output_dir="output/clips/"
)
```

### Advanced Options
```python
video = sadtalker.generate_talking_head(
    image_path="teacher.png",
    audio_path="lesson.wav",
    output_path="output.mp4",
    expression_scale=1.2,      # More expressive
    head_motion=True,          # Enable head movement
    blink=True,                # Enable eye blinking
    fps=25,
    quality="high"
)
```

## Configuration

### SadTalker Settings
```python
sadtalker_config = {
    "device": "cuda",           # GPU acceleration
    "expression_scale": 1.0,    # Expression intensity
    "pose_style": 0,            # Head pose style (0-45)
    "preprocess": "full",       # Face detection quality
    "still_mode": False,        # Minimal head motion
    "enhancer": "gfpgan",       # Face quality enhancer
    "background_enhancer": None # Optional background upscale
}
```

### Quality Presets
```python
# Fast preview (for testing)
quality_presets = {
    "preview": {
        "fps": 15,
        "resolution": 256,
        "enhancer": None
    },
    # Production quality
    "high": {
        "fps": 25,
        "resolution": 512,
        "enhancer": "gfpgan"
    },
    # Maximum quality
    "ultra": {
        "fps": 30,
        "resolution": 1024,
        "enhancer": "gfpgan",
        "background_enhancer": "realesrgan"
    }
}
```

## Performance

### Benchmarks (RTX 3060)
| Resolution | FPS | Processing Speed | VRAM Usage |
|-----------|-----|------------------|------------|
| 256x256   | 25  | ~10 FPS          | 3-4 GB     |
| 512x512   | 25  | ~4 FPS           | 5-6 GB     |
| 1024x1024 | 25  | ~1 FPS           | 7-8 GB     |

**Note**: Processing speed = frames generated per second (not real-time FPS)

### Optimization Tips
- Use "preview" mode for quick tests
- Pre-crop images to face region (saves processing)
- Use still_mode=True for less head motion (faster)
- Batch process multiple clips for efficiency
- Cache face detection results for same image

## Quality Metrics

The audio manager is evaluated on:
- **Lip-Sync Accuracy**: Audio-visual sync confidence > 0.85
- **Face Quality**: FID score < 15 for facial realism
- **Motion Smoothness**: Optical flow consistency > 0.9
- **Expression Match**: Emotion classifier agreement > 75%
- **Temporal Consistency**: <3% flicker between frames

## Dependencies

- **SadTalker**: Core talking-head generation
  - `torch` - PyTorch models
  - `face-alignment` - Face detection
  - `librosa` - Audio processing
  - `scipy` - Signal processing
- **GFPGAN**: Face enhancement (optional)
- **ffmpeg**: Video encoding

**Models**: Auto-downloaded on first use (~1.5 GB total)

## Troubleshooting

### Common Issues

**Q: Lip-sync is out of sync**
- Check audio quality (should be clear speech)
- Verify audio format (prefer WAV 16kHz mono)
- Try adjusting expression_scale
- Ensure image is frontal face portrait

**Q: Face looks distorted/low quality**
- Enable face enhancer: `enhancer="gfpgan"`
- Use higher resolution input image (512px+)
- Ensure face is clearly visible and well-lit
- Pre-crop to face region

**Q: No head motion/looks static**
- Set `still_mode=False`
- Increase `pose_style` (0-45 for more motion)
- Use longer audio clips (>5 seconds)
- Check that image has clear face landmarks

**Q: Out of memory errors**
- Reduce output resolution (1024 → 512)
- Disable enhancer temporarily
- Process shorter audio clips (<30s)
- Clear CUDA cache: `torch.cuda.empty_cache()`

**Q: Mouth movements look unnatural**
- Reduce expression_scale: `0.8` (less exaggerated)
- Use higher quality audio input
- Ensure audio has clear speech (not music)
- Try different pose_style values

## Advanced Features

### Custom Expression Control
```python
# Generate with specific emotion
video = sadtalker.generate_with_emotion(
    image="teacher.png",
    audio="lesson.wav",
    emotion="enthusiastic",  # happy, neutral, serious
    intensity=0.8
)
```

### Multi-Speaker Handling
```python
# Switch between different speaker images based on audio segments
video = sadtalker.multi_speaker_video(
    speaker_images=["speaker1.png", "speaker2.png"],
    audio_segments=["audio1.wav", "audio2.wav"],
    output="conversation.mp4"
)
```

### Background Music Integration
```python
# Add background music to talking head
video = sadtalker.generate_with_bgm(
    image="teacher.png",
    speech_audio="lesson.wav",
    bgm_audio="music.mp3",
    bgm_volume=0.2  # 20% volume
)
```

## Integration with TTV Pipeline

The audio manager integrates with:
1. **Lesson Generation**: Creates teacher talking-head segments
2. **AnimateDiff**: Can combine talking heads with animated scenes
3. **Upscaler**: Upscales generated videos for higher quality
4. **Final Assembly**: Composites talking heads into lesson videos

## Typical Workflow

```
Input: Teacher portrait (512x512) + Lesson audio (WAV)
    ↓
Face Detection & Alignment
    ↓
Audio Feature Extraction (mel-spectrogram)
    ↓
SadTalker: Generate 3D head motion + expression
    ↓
Render lip-synced frames (25 FPS)
    ↓
Face Enhancement (GFPGAN)
    ↓
Video Encoding (H.264)
    ↓
Output: Talking-head video
```

## Best Practices

### Input Image Requirements
- **Resolution**: 512x512 or higher
- **Composition**: Frontal face, clear features
- **Lighting**: Even, no harsh shadows
- **Background**: Simple, non-distracting
- **Expression**: Neutral or slight smile

### Audio Requirements
- **Format**: WAV 16kHz mono (best quality)
- **Duration**: 5-120 seconds per clip
- **Content**: Clear speech, minimal background noise
- **Volume**: Normalized to -3dB to -6dB peak

### Quality Optimization
1. Use high-quality input images (professional headshots)
2. Pre-process audio (noise reduction, normalization)
3. Enable face enhancement for production videos
4. Test with preview mode before high-quality render
5. Use consistent face images for same character

## Version History

- **v1.0** (Task 9): Initial SadTalker integration
- **v2.0** (Task 10): Enhanced quality controls and batch processing
- **v3.0** (Task 11): Multi-speaker support and BGM integration
- **v4.0** (Task 12): Expression control and optimization

## Related Modules

- **AnimateDiff**: Alternative for full-body animation
- **Upscaler**: Enhances talking-head video quality
- **Security**: Watermarking for generated content

## License

Part of the TTV (Text-to-Vision) production pipeline.
Uses SadTalker (Custom License) - research and educational use only.

## Credits

SadTalker implementation based on:
- Paper: "SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation"
- Repository: https://github.com/OpenTalker/SadTalker
