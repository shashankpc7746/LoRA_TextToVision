# Adapters Module

## Purpose
The Adapters module provides LoRA (Low-Rank Adaptation) capabilities for fine-tuning the text-to-image diffusion models with custom visual styles, specifically optimized for educational content with indigenous Indian cultural elements.

## Key Components

### 1. **LoRA Adapter** (`lora_adapter.py`)
- Core LoRA implementation for Stable Diffusion models
- Supports loading, applying, and managing LoRA weights
- Provides `LoRAAdapter` and `GurukulLoRA` classes

### 2. **Adapter Manager** (`adapter_manager.py`)
- Manages multiple LoRA adapters
- Handles adapter lifecycle (load, save, switch)
- Maintains adapter metadata and versioning

### 3. **Adapter Trainer** (`adapter_trainer.py`)
- Training pipeline for custom LoRA adapters
- Dataset preparation and augmentation
- Training monitoring and checkpointing

### 4. **Keyframe Generator** (`keyframe_generator.py`)
- Generates high-quality keyframes using SDXL + LoRA
- Supports batch generation for video sequences
- Advanced prompt engineering for consistency

### 5. **AnimateDiff Bridge** (`animate_diff_bridge.py`)
- Connects keyframe generation to AnimateDiff pipeline
- Manages temporal consistency across frames
- Optimizes for smooth animation transitions

## Usage

### Quick Start: Generate Keyframes
```python
from adapters.keyframe_generator import get_keyframe_generator

generator = get_keyframe_generator()
keyframes = generator.generate_keyframes(
    prompt="Ancient Indian mathematician teaching geometry",
    num_keyframes=6,
    style="gurukul_educational"
)
```

### Train Custom LoRA Adapter
```python
from adapters.adapter_trainer import quick_train_gurukul_adapter

adapter_path = quick_train_gurukul_adapter(
    keyframes_dir="datasets/gurukul_keyframes"
)
```

### Load and Use LoRA
```python
from adapters.lora_adapter import get_gurukul_lora

lora = get_gurukul_lora()
lora.load_adapter("adapters/gurukul_lora_v1")
image = lora.generate(
    prompt="Traditional Indian classroom scene",
    lora_scale=0.8
)
```

## Gurukul LoRA Training Dataset

The `gurukul_lora/` subdirectory contains:
- **Dataset Generation**: Scripts to create training datasets with Indian cultural elements
- **Training Scripts**: Optimized training configurations for educational content
- **Dataset Curator**: Tools for dataset quality control and augmentation
- **Download Tools**: Automated download of cultural reference images from Pexels and Unsplash

### Dataset Features
- **Categories**: Architecture, Art, Clothing, Cultural Practices, Nature, People
- **Quality Control**: Automated filtering, deduplication, and validation
- **Augmentation**: Style-preserving data augmentation for robust training
- **Size**: 500+ curated images for production-quality LoRA

## Configuration

Key configuration files:
- `adapter_config.json` - LoRA architecture settings
- `training_config.json` - Training hyperparameters
- `style_presets.json` - Predefined visual styles

## Performance Optimization

- **Mixed Precision Training**: FP16 for faster training
- **Gradient Checkpointing**: Reduced memory footprint
- **Caching**: Keyframe caching to avoid redundant generation
- **Batch Processing**: Parallel keyframe generation

## Quality Metrics

The LoRA adapter is evaluated on:
- **Visual Consistency**: Temporal coherence across frames
- **Cultural Accuracy**: Authentic representation of Indian elements
- **Style Adherence**: Consistent visual style matching training data
- **Generation Speed**: <2s per keyframe on RTX 3060

## Dependencies

- `diffusers` - Stable Diffusion pipelines
- `transformers` - CLIP and text encoders
- `torch` - PyTorch for model training
- `safetensors` - Efficient model serialization
- `peft` - Parameter-efficient fine-tuning library

## Troubleshooting

### Common Issues

**Q: Keyframes lack cultural authenticity**
- Increase LoRA scale (0.7-0.9)
- Verify gurukul_lora adapter is loaded
- Check training dataset quality

**Q: Training runs out of memory**
- Enable gradient checkpointing
- Reduce batch size to 1
- Use mixed precision (FP16)

**Q: Generated keyframes are inconsistent**
- Use consistent seed across generation
- Enable style transfer mode
- Increase guidance scale (7-12)

## Version History

- **v1.0** (Task 7): Initial LoRA implementation with basic keyframe generation
- **v2.0** (Task 9): Gurukul LoRA training with cultural dataset
- **v3.0** (Task 11): AnimateDiff bridge for smooth video generation

## Related Modules

- **AnimateDiff**: Uses keyframes for video animation
- **Interpolator**: Enhances temporal smoothness
- **Upscaler**: Improves keyframe resolution

## License

Part of the TTV (Text-to-Vision) production pipeline.
