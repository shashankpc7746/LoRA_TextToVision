# Gurukul LoRA Adapter - Testing Guide

## Current Status: Testing Phase

After successfully training the Gurukul LoRA adapter for 10 epochs, we're now testing image generation capabilities.

## Training Results Summary

- **Model**: gurukul_lora.pt (89 MB, 23.2M parameters)
- **Training Time**: 3 hours 45 minutes (10 epochs)
- **Loss**: Started at 0.0101, ended at 0.0070 (30.7% reduction)
- **Status**: ✅ Training completed successfully, zero NaN issues

## Testing Approach

### Test Script: `test_generate_simple.py`

**Purpose**: Generate sample images to validate the trained LoRA adapter

**Test Images**:
1. Traditional Gurukul with students under banyan tree
2. Ancient Gurukul classroom with Sanskrit texts
3. Gurukul courtyard with meditation area

**Generation Settings**:
- Steps: 25 (faster inference for testing)
- Guidance Scale: 7.5
- Resolution: 512x512
- Negative Prompt: "blurry, low quality, distorted"

**Expected Runtime**: 
- Model loading: 30s-1min (if cached) or 5-10min (first time)
- Per image: 30-60 seconds
- Total: ~5-15 minutes for 3 images

## Output Location

Generated images will be saved to:
```
adapters/gurukul_lora/test_outputs/
  ├── gurukul_01.png
  ├── gurukul_02.png
  └── gurukul_03.png
```

## What to Look For

### Success Indicators:
- ✅ Images generate without errors
- ✅ Images show Gurukul-themed content (traditional Indian educational settings)
- ✅ Good quality and coherence
- ✅ Visible difference from base SDXL (more Gurukul-specific)

### Potential Issues:
- ❌ Generic images (not Gurukul-specific) = May need more training
- ❌ Blurry/low quality = May need higher resolution or more epochs
- ❌ OOM errors = Reduce batch size or resolution
- ❌ Still looks like base SDXL = LoRA may not be loading correctly

## Next Steps After Testing

### If Results are Good:
1. **Extended Training**: Train for 50-100 epochs for production quality
2. **Higher Resolution**: Test with 768x768 or 1024x1024 (if VRAM allows)
3. **Integration**: Connect with other Task 9 components:
   - Temporal consistency (`interpolator/`)
   - Upscaler (`upscaler/`)
   - Motion controller (`motion_controller/`)
   - Quality card (`test_quality_card.py`)

### If Results Need Improvement:
1. **More Epochs**: Continue training (current: 10, recommended: 50-100)
2. **Data Augmentation**: Add more diverse Gurukul images to dataset
3. **Hyperparameter Tuning**: Adjust learning rate, LoRA rank, alpha
4. **Prompt Engineering**: Test different prompt styles

## Training Script for Extended Training

If you want to train longer, use:
```bash
cd adapters/gurukul_lora
# Edit train_optimized.py and change:
#   num_epochs = 50  # or 100

python train_optimized.py
```

**Time Estimates**:
- 50 epochs: ~18.75 hours (overnight)
- 100 epochs: ~37.5 hours (1.5 days)

## Verification Commands

Check if images were generated:
```powershell
cd adapters/gurukul_lora
ls test_outputs/
```

View image properties:
```powershell
python -c "from PIL import Image; img = Image.open('test_outputs/gurukul_01.png'); print(f'Size: {img.size}, Mode: {img.mode}')"
```

## Comparison Test

To compare base SDXL vs LoRA adapter, the full `test_generate.py` includes a comparison feature that generates the same prompt with and without LoRA.

## Integration with Task 9 Pipeline

Once testing confirms the adapter works:

1. **Video Generation**: Use with AnimateDiff for Gurukul lesson videos
2. **Upscaling**: Apply tile-based upscaler for 4K quality
3. **Temporal Smoothing**: Ensure consistent Gurukul style across frames
4. **Motion Control**: Animate static Gurukul scenes with micro-expressions
5. **Quality Assessment**: Validate output against Task 9 requirements

---

**Current Status**: Image generation test running in external PowerShell window.
**Estimated Completion**: 5-15 minutes from launch time.
