# Upscaler Module

## Purpose
The Upscaler module provides AI-powered video upscaling and quality enhancement. It uses ESRGAN (Enhanced Super-Resolution GAN) to increase video resolution while preserving and enhancing details, enabling production of high-quality output from lower-resolution intermediates.

## Key Components

### 1. **ESRGAN Upscaler** (`esrgan_upscaler.py`)
- Real-ESRGAN 4x upscaling for video frames
- GPU-accelerated inference
- Tile-based processing for large frames and 4K support
- Face enhancement mode for educational videos with people

### 2. **Tile Upscale** (`tile_upscale.py`)
- Memory-efficient tile-based processing
- Seamless tile blending to avoid artifacts
- Adaptive tile size based on VRAM availability
- Batch processing for multi-frame sequences

### 3. **Upscale Pipeline** (`upscale_pipeline.py`)
- Complete pipeline: video → denoise → upscaler → cinematic polish → final output
- Pre-upscale denoising and quality enhancement
- Post-upscale sharpening and color grading
- Quality validation and artifact detection

## Usage

### Quick Start: Upscale Single Frame
```python
from upscaler.esrgan_upscaler import get_esrgan_upscaler

upscaler = get_esrgan_upscaler()
upscaled_image = upscaler.upscale_image(
    image_path="input_frame.png",
    output_path="upscaled_frame.png",
    scale=4  # 4x resolution increase
)
```

### Upscale Video Sequence
```python
from upscaler.upscale_pipeline import UpscalePipeline

pipeline = UpscalePipeline()
result = pipeline.upscale_video(
    input_video="input_720p.mp4",
    output_path="output_1080p.mp4",
    target_resolution=(1920, 1080),
    denoise=True,
    enhance_faces=True
)
```

### Tile-Based Upscaling for 4K
```python
from upscaler.tile_upscale import get_tile_processor

tile_processor = get_tile_processor()
upscaled = tile_processor.upscale_with_tiles(
    image_path="large_frame.png",
    tile_size=512,  # Process in 512x512 tiles
    overlap=32,     # Blend overlap region
    scale=4
)
```

## Features

### ESRGAN Upscaling
- **Model Variants**: 
  - RealESRGAN_x4plus (general scenes)
  - RealESRGAN_x4plus_anime (animated content)
  - RealESRGANv2-animevideo (video optimization)
- **Face Enhancement**: GFPGAN integration for better face quality
- **FP16 Support**: Half-precision for 2x speed improvement
- **Adaptive Processing**: Automatically adjusts based on GPU capability

### Denoising Engine
- **Temporal Denoising**: 3-frame window for motion-aware noise reduction
- **Spatial Denoising**: Non-local means for detail preservation
- **Luminance/Color Separation**: Independent processing for better quality
- **Sharpen Enhancement**: Adaptive sharpening to restore fine details

### Cinematic Polish
- **Color Grading**: Automatic LUT application for film look
- **Contrast Enhancement**: Adaptive histogram equalization
- **Grain Addition**: Film grain for organic appearance
- **Vignette**: Subtle edge darkening for focus

## Configuration

### ESRGAN Settings
```python
esrgan_config = {
    "model_name": "RealESRGAN_x4plus",
    "denoise_strength": 0.5,      # Pre-upscale denoising
    "face_enhance": True,          # Enable GFPGAN for faces
    "tile_size": 512,              # Tile size for large images
    "tile_pad": 32,                # Overlap for seamless blending
    "pre_pad": 10,                 # Padding to reduce boundary effects
    "fp16": True                   # Half-precision mode
}
```

### Denoise Settings
```python
denoise_config = {
    "temporal_radius": 3,          # Temporal window
    "spatial_sigma": 15,           # Spatial denoising strength
    "luminance_sigma": 10,         # Luminance denoising
    "color_sigma": 20,             # Color denoising
    "sharpen_strength": 0.3        # Post-denoise sharpening
}
```

### Cinematic Polish Settings
```python
polish_config = {
    "color_grading": "cinematic",  # LUT preset
    "contrast_boost": 1.15,        # Contrast multiplier
    "saturation": 1.1,             # Color saturation
    "film_grain_strength": 0.05,   # Grain intensity
    "vignette_strength": 0.2       # Edge darkening
}
```

## Performance

### Benchmarks (RTX 3060)
| Input Resolution | Output Resolution | Processing Speed | VRAM Usage |
|-----------------|-------------------|------------------|------------|
| 512x512         | 2048x2048        | ~2 FPS           | 3-4 GB     |
| 720p            | 1440p            | ~1 FPS           | 4-5 GB     |
| 1080p           | 4K               | ~0.3 FPS         | 6-7 GB     |

### Optimization Tips
- Use FP16 mode: `upscaler.fp16 = True` (2x faster)
- Enable tile processing for >1080p: `tile_size=512`
- Batch process multiple frames in parallel
- Pre-downscale if input is already high-res
- Use `realesrgan-ncnn-vulkan` for CPU-only systems

## Quality Metrics

The upscaler is evaluated on:
- **Detail Preservation**: LPIPS < 0.15 vs ground truth 4K
- **Sharpness**: Laplacian variance > 100 (sharp details)
- **Artifact Level**: No visible ringing, aliasing, or blocking
- **Face Quality**: FID score < 10 for facial regions
- **Temporal Consistency**: <5% flicker between frames

## Dependencies

- `basicsr` - Super-resolution framework
- `realesrgan` - ESRGAN implementation
- `gfpgan` - Face enhancement (optional)
- `torch` - PyTorch for model inference
- `opencv-python` - Frame processing
- `pillow` - Image I/O

**Models**: Auto-downloaded on first use (~65 MB for RealESRGAN_x4plus)

## Troubleshooting

### Common Issues

**Q: Upscaled frames have artifacts/ringing**
- Reduce denoise strength: `denoise_strength=0.3`
- Enable tile processing with larger overlap: `tile_pad=64`
- Use anime model for cartoon content
- Check input quality (garbage in, garbage out)

**Q: Out of memory during upscaling**
- Reduce tile size: `tile_size=256`
- Enable FP16: `fp16=True`
- Process fewer frames in parallel
- Clear CUDA cache: `torch.cuda.empty_cache()`

**Q: Faces look distorted**
- Enable face enhancement: `face_enhance=True`
- Ensure GFPGAN model is downloaded
- Check face detection (requires clear frontal faces)
- Reduce upscale factor if faces are very small

**Q: Colors look washed out/over-saturated**
- Adjust saturation: `saturation=1.0` (neutral)
- Disable color grading: `color_grading=None`
- Use custom LUT for specific look

**Q: Processing is extremely slow**
- Enable FP16 mode (2x speedup)
- Use smaller tile size
- Disable face enhancement if not needed
- Consider using ncnn-vulkan version for faster CPU processing

## Advanced Features

### Custom Model Loading
```python
upscaler.load_custom_model(
    model_path="models/custom_esrgan.pth",
    model_name="custom_4x"
)
```

### Selective Face Enhancement
```python
# Only enhance detected faces
pipeline.enhance_faces_only(
    face_threshold=0.8,  # Confidence threshold
    upsample_factor=2    # Additional face upsampling
)
```

### Multi-Stage Upscaling
```python
# 2x → denoise → 2x for 4x total (better quality)
pipeline.multi_stage_upscale(
    stages=[2, 2],
    denoise_between_stages=True
)
```

### Adaptive Quality Mode
```python
# Automatically adjust settings based on content
pipeline.auto_quality_mode(
    content_type="educational",  # Optimizes for people + text
    target_quality="high"
)
```

## Integration with TTV Pipeline

The upscaler integrates with:
1. **Interpolator**: Upscales interpolated frames for final output
2. **AnimateDiff**: Enhances AnimateDiff-generated videos
3. **Adapters**: Can upscale LoRA-generated keyframes
4. **Final Encoding**: Provides high-res frames for video encoding

## Typical Workflow

```
Low-res frames (512x512)
    ↓
Denoise (temporal + spatial)
    ↓
ESRGAN Upscale (4x → 2048x2048)
    ↓
Face Enhancement (GFPGAN)
    ↓
Cinematic Polish (color grading, sharpening)
    ↓
High-quality output (ready for encoding)
```

## Version History

- **v1.0** (Task 7): Initial ESRGAN integration
- **v2.0** (Task 9): Tile processing for 4K support
- **v3.0** (Task 11): Denoise pipeline and cinematic polish
- **v4.0** (Task 12): Face enhancement and adaptive quality

## Related Modules

- **Interpolator**: Provides smooth frames for upscaling
- **AnimateDiff**: Source of video frames to upscale
- **Security**: Watermarking applied after upscaling

## License

Part of the TTV (Text-to-Vision) production pipeline.
Uses Real-ESRGAN (BSD-3-Clause) and GFPGAN (Apache 2.0).
