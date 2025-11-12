# 🎓 LoRA_TextToVision Developer Handbook

**Complete Architecture, Hand-off & Onboarding Guide**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Module Reference](#module-reference)
4. [Development Workflow](#development-workflow)
5. [Testing Strategy](#testing-strategy)
6. [Deployment Guide](#deployment-guide)
7. [Troubleshooting](#troubleshooting)
8. [Extension Points](#extension-points)

---

## 🎯 System Overview

### What is LoRA_TextToVision?

Enterprise-grade AI video generation platform that transforms text prompts into high-quality educational videos through:

- **Multi-stage Pipeline**: Text → Keyframes → Animation → Interpolation → Audio → Upscaling
- **Adaptive Quality**: RL-powered parameter optimization for cost/quality balance
- **Robust Fallback**: Intelligent GPU tier management + cloud fallback
- **Production Security**: Watermarking, fingerprinting, audit logging, CI/CD gates

### Project Structure (High-Level)

```
LoRA_TextToVision/
├── adapters/              # LoRA training & keyframe generation
├── AnimateDiff/           # Core animation engine
├── AnimateDiff_API/       # REST API & adaptive endpoints
├── interpolator/          # RIFE frame interpolation
├── audio_manager/         # SadTalker lip-sync integration
├── upscaler/             # Real-ESRGAN 1080p upscaling
├── motion_controller/    # RL parameter optimization
├── security/             # Task 10: Watermarking, signing, encryption
├── tests/                # Comprehensive test suite
├── Documentation/        # All task docs & reports
└── orchestrator.py       # Main pipeline coordinator
```

---

## 🏗️ Architecture Deep Dive

### 1. Pipeline Flow (Sequence Diagram)

```
┌─────────┐     ┌──────────┐     ┌────────────┐     ┌──────────────┐
│  User   │────▶│   API    │────▶│ Orchestr.  │────▶│  Adapter     │
│ Request │     │ Endpoint │     │            │     │  Manager     │
└─────────┘     └──────────┘     └────────────┘     └──────────────┘
                                         │                   │
                                         ▼                   ▼
                                  ┌────────────┐     ┌──────────────┐
                                  │   RL       │     │ LoRA Train   │
                                  │ Optimizer  │◀───▶│ & Generate   │
                                  └────────────┘     └──────────────┘
                                         │                   │
                                         ▼                   ▼
                                  ┌────────────┐     ┌──────────────┐
                                  │ AnimateDiff│────▶│ Interpolator │
                                  │  Engine    │     │ (RIFE)       │
                                  └────────────┘     └──────────────┘
                                         │                   │
                                         ▼                   ▼
                                  ┌────────────┐     ┌──────────────┐
                                  │   Audio    │────▶│  Upscaler    │
                                  │  Manager   │     │ (ESRGAN)     │
                                  └────────────┘     └──────────────┘
                                         │                   │
                                         ▼                   ▼
                                  ┌────────────┐     ┌──────────────┐
                                  │  Security  │────▶│   Output     │
                                  │ Watermark  │     │   Video      │
                                  └────────────┘     └──────────────┘
```

### 2. GPU Resource Allocation

| Stage | GPU | VRAM | Priority | Fallback |
|-------|-----|------|----------|----------|
| LoRA Training | RTX 3080 | 8GB | HIGH | Yotta Cloud |
| Keyframe Gen | RTX 3080 | 8GB | HIGH | RTX 3060 |
| Animation | RTX 3060 | 8GB | MEDIUM | RTX 3080 |
| Interpolation | RTX 3060 | 6GB | MEDIUM | CPU (slow) |
| Upscaling | RTX 3080 | 8GB | MEDIUM | Skip |
| Audio | CPU | - | LOW | Always |

**Intelligent Allocation Logic**:
- Monitor VRAM usage real-time (`torch.cuda.memory_allocated()`)
- Automatic GPU switching based on availability
- Cloud escalation if local GPUs saturated (>3 concurrent requests)

### 3. Data Flow

```python
# Complete pipeline data transformation

# INPUT: Text prompt
prompt = "A majestic eagle soaring through mountains"

# STAGE 1: Prompt Engineering (adapters/adapter_manager.py)
enhanced_prompt = {
    "positive": "majestic eagle, mountains, cinematic, 4k",
    "negative": "blurry, low quality, distorted"
}

# STAGE 2: LoRA Keyframe Generation (adapters/lora_adapter.py)
keyframes = [
    Image(512x512),  # Frame 0
    Image(512x512),  # Frame 4
    Image(512x512),  # Frame 8
]

# STAGE 3: AnimateDiff Animation (AnimateDiff/animatediff/pipelines/)
animated_video = VideoClip(
    resolution=512x512,
    fps=8,
    duration=2.0s,
    frames=16
)

# STAGE 4: RIFE Interpolation (interpolator/rife_interpolator.py)
interpolated_video = VideoClip(
    resolution=512x512,
    fps=24,  # 8fps → 24fps (3x interpolation)
    duration=2.0s,
    frames=48
)

# STAGE 5: Audio Synthesis (audio_manager/enhanced_sadtalker.py)
video_with_audio = VideoClip(
    resolution=512x512,
    fps=24,
    audio=AudioTrack(duration=2.0s),
    lip_sync_quality=0.87
)

# STAGE 6: Upscaling (upscaler/esrgan_upscaler.py)
upscaled_video = VideoClip(
    resolution=1920x1080,  # 512 → 1080p (4x upscale)
    fps=24,
    audio=AudioTrack(duration=2.0s)
)

# STAGE 7: Security (security/watermark.py)
final_video = VideoClip(
    resolution=1920x1080,
    watermark={"BUILD_ID": "build_20251112_...", "fingerprint": "sha256:..."},
    metadata={"signed": True, "provenance": "verified"}
)

# OUTPUT: Final video file
output_path = "storage/2025-11-12/eagle_mountains_complete.mp4"
```

---

## 📦 Module Reference

### Core Modules

#### 1. **orchestrator.py** (Main Pipeline Coordinator)

**Purpose**: Coordinates entire video generation pipeline with adaptive quality

**Key Classes**:
```python
class ProductionOrchestrator:
    def __init__(self):
        self.adapter_manager = AdapterManager()
        self.interpolator = RIFEInterpolator()
        self.audio_manager = EnhancedSadTalker()
        self.upscaler = ESRGANUpscaler()
        self.rl_optimizer = RLOptimizer()
    
    async def generate_video(
        self,
        prompt: str,
        target_quality: float = 0.8,
        max_cost_usd: float = 1.0,
        max_latency_sec: int = 300
    ) -> Dict:
        """
        Main generation entry point
        
        Returns:
        {
            "success": True,
            "output_path": "path/to/video.mp4",
            "quality_score": 0.87,
            "cost_usd": 0.45,
            "latency_seconds": 145.2
        }
        """
```

**Dependencies**:
- `adapters.adapter_manager.AdapterManager`
- `interpolator.rife_interpolator.RIFEInterpolator`
- `audio_manager.enhanced_sadtalker.EnhancedSadTalker`
- `upscaler.esrgan_upscaler.ESRGANUpscaler`
- `motion_controller.rl_optimizer.RLOptimizer`

**Integration Points**:
- FastAPI app exposed at `/ttv/generate`
- InsightFlow telemetry via `insightflow_client.py`
- Audit logging via `audit_logger.py`

---

#### 2. **adapters/** (LoRA Training & Keyframe Generation)

**Purpose**: Fine-tune Stable Diffusion with LoRA for domain-specific content

**Key Files**:
- `adapter_manager.py`: Orchestrates LoRA training and generation
- `lora_adapter.py`: LoRA weight management
- `adapter_trainer.py`: Training loop implementation
- `keyframe_generator.py`: Keyframe extraction and generation

**Example Usage**:
```python
from adapters import AdapterManager

manager = AdapterManager()

# Train LoRA adapter on custom dataset
await manager.train_adapter(
    dataset_path="datasets/gurukul_keyframes",
    adapter_name="gurukul_v2",
    steps=1000
)

# Generate keyframes
keyframes = await manager.generate_keyframes(
    prompt="Ancient temple in misty mountains",
    num_keyframes=8,
    adapter="gurukul_v2"
)
```

**Configuration** (`adapters/config.json`):
```json
{
    "base_model": "stabilityai/stable-diffusion-2-1",
    "lora_rank": 64,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4
}
```

---

#### 3. **AnimateDiff/** (Animation Engine)

**Purpose**: Convert static keyframes into animated video sequences

**Key Files**:
- `animatediff/pipelines/pipeline_animation.py`: Core animation pipeline
- `cinematic_flow_engine.py`: Camera movements (pan, zoom, dolly)
- `unified_video_generator.py`: High-level generation interface
- `subtitle_sync_engine.py`: Subtitle synchronization

**Example Usage**:
```python
from AnimateDiff.unified_video_generator import UnifiedVideoGenerator

generator = UnifiedVideoGenerator()

video = generator.generate_complete_video(
    lesson_path="lessons/lesson_space_adventure.json",
    style="realistic",
    speech_rate=1
)
# Returns: "storage/2025-11-12/Space_Adventure_realistic_complete.mp4"
```

**Cinematic Flow**:
```python
from AnimateDiff.cinematic_flow_engine import CinematicFlowEngine

engine = CinematicFlowEngine()

enhanced_clip = engine._enhance_clip_with_flow(
    video_clip=clip,
    scene='temple',
    flow_instruction={'movement': 'pan_right', 'intensity': 0.3},
    clip_index=0
)
```

---

#### 4. **interpolator/** (Frame Interpolation)

**Purpose**: Increase frame rate from 8fps → 24fps using RIFE neural interpolation

**Key Files**:
- `rife_interpolator.py`: RIFE model inference
- `interpolation_pipeline.py`: Multi-stage interpolation
- `temporal_consistency.py`: Temporal stability checks

**Example Usage**:
```python
from interpolator import RIFEInterpolator

interpolator = RIFEInterpolator()

# Interpolate 8fps → 24fps
high_fps_video = interpolator.interpolate_video(
    input_path="video_8fps.mp4",
    target_fps=24,
    output_path="video_24fps.mp4"
)
```

**Performance**:
- RTX 3060: ~0.15s per frame pair
- RTX 3080: ~0.10s per frame pair
- Quality: PSNR ~35dB, SSIM ~0.95

---

#### 5. **audio_manager/** (Lip-sync & Audio)

**Purpose**: Generate synchronized audio and lip movements using SadTalker

**Key Files**:
- `enhanced_sadtalker.py`: SadTalker integration with enhancements
- `SadTalker/src/`: Original SadTalker implementation

**Example Usage**:
```python
from audio_manager import EnhancedSadTalker

sadtalker = EnhancedSadTalker()

synced_video = sadtalker.generate_lipsync(
    video_path="video.mp4",
    audio_path="audio.wav",
    output_path="synced_video.mp4"
)
```

**Quality Metrics**:
- Lip-sync confidence: >0.85 for good sync
- Audio-video delay: <50ms acceptable

---

#### 6. **upscaler/** (Video Upscaling)

**Purpose**: Upscale 512x512 → 1920x1080 using Real-ESRGAN

**Key Files**:
- `esrgan_upscaler.py`: Main upscaling interface
- `tile_processor.py`: Memory-efficient tile-based upscaling
- `cinematic_polish.py`: Post-processing effects

**Example Usage**:
```python
from upscaler import ESRGANUpscaler

upscaler = ESRGANUpscaler()

hd_video = upscaler.upscale_video(
    input_path="video_512p.mp4",
    output_path="video_1080p.mp4",
    scale_factor=4  # 512 → 2048, then downscale to 1080
)
```

**Optimization**:
- Tile size: 256x256 (balance quality/memory)
- Batch processing: 4-8 frames at once
- GPU memory: ~6GB for 4x upscale

---

#### 7. **motion_controller/** (RL Optimization)

**Purpose**: Optimize generation parameters using Reinforcement Learning

**Key Files**:
- `rl_optimizer.py`: PPO-based parameter tuning
- `quality_predictor.py`: VMAF score prediction
- `parameter_search.py`: Hyperparameter exploration

**Example Usage**:
```python
from motion_controller import RLOptimizer

optimizer = RLOptimizer()

# Get optimized parameters for target quality
params = optimizer.optimize_for_quality(
    target_quality=0.85,
    max_cost=0.50
)

print(params)
# {
#     "interpolation_enabled": True,
#     "upscale_enabled": True,
#     "target_fps": 24,
#     "inference_steps": 30
# }
```

**Reward Function**:
```python
reward = (
    quality_score * 10.0 +  # VMAF/predicted quality
    (-cost_usd * 5.0) +      # Cost penalty
    (-latency_sec / 60.0)    # Latency penalty
)
```

---

#### 8. **security/** (Task 10: Anti-Clone & Security)

**Purpose**: Watermarking, fingerprinting, encryption, signing for production security

**Key Files**:
- `watermark.py`: Invisible FFmpeg metadata watermarking
- `visible_watermark.py`: BHI logo watermark (35% opacity)
- `artifact_signer.py`: Ed25519 artifact signing
- `runtime_validator.py`: Core-signed runtime key validation
- `ksml_encryption.py`: AES-256-GCM encryption with KSML tokens

**Example Usage**:
```python
from security import embed_watermark, compute_fingerprint, add_visible_watermark

# Step 1: Invisible watermark
watermarked = embed_watermark(
    "video.mp4",
    build_id="build_20251112_123456",
    output_path="video_watermarked.mp4"
)

# Step 2: Visible logo
final = add_visible_watermark(
    watermarked,
    style="subtle"  # 35% opacity
)

# Step 3: Fingerprint
fingerprint = compute_fingerprint(final, build_id="build_20251112_123456")
print(fingerprint['sha256'])  # Content hash
```

**Watermark Detection**:
```bash
# Use provenance detection tool
python tools/detect_provenance.py "video.mp4"

# Output:
# ✅ Watermark detected!
#    Build ID: build_20251112_123456
#    Method: ffmpeg_metadata
# ✅ VERIFIED - File has valid provenance
```

**CI/CD Integration**:
- `.github/workflows/security-artifact-signing.yml`: Sign models/adapters
- `.github/workflows/security-docker-signing.yml`: Sign container images
- `.github/workflows/security-gates.yml`: Mandatory security checks

---

#### 9. **yotta_fallback.py** (Cloud Fallback)

**Purpose**: Intelligent cloud escalation when local GPUs saturated

**Key Functions**:
```python
from yotta_fallback import get_fallback_manager

manager = get_fallback_manager()

# Automatic fallback logic
result = await manager.process_with_fallback(
    prompt="Complex high-quality scene",
    target_quality=0.9,
    local_gpu_threshold=0.8  # Trigger cloud if GPU >80% utilized
)

print(result['processing_path'])
# ["local", "cloud"] - tried local, escalated to cloud
```

**Fallback Triggers**:
1. GPU memory < 4GB available
2. Estimated generation time > 15 minutes
3. Quality target > 0.9
4. Concurrent requests > 3
5. Local GPU failure/timeout

---

## 🔬 Testing Strategy

### Test Organization

```
tests/
├── task9/                      # Task 9 comprehensive tests
│   ├── components/             # Component-level tests
│   │   ├── motion/             # Motion controller tests
│   │   ├── temporal/           # Temporal consistency tests
│   │   └── upscaler/           # Upscaler tests
│   ├── integration/            # Integration tests
│   │   ├── test_task9_integration.py
│   │   └── test_task9_simple.py
│   └── quality/                # Quality validation tests
│       ├── test_comprehensive.py
│       └── test_quality_card.py
├── task10/                     # Task 10 security tests
│   ├── test_task10_integration.py
│   └── test_watermark_quick.py
├── integration/                # Full pipeline tests
│   └── test_end_to_end.py      # New: Full workflow test
└── ttv_service/tests/          # API service tests
    ├── test_unit.py
    └── test_integration.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific task tests
pytest tests/task9/ -v
pytest tests/task10/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/task10/test_task10_integration.py -v
```

### Test Coverage Targets

| Module | Current Coverage | Target |
|--------|------------------|--------|
| adapters/ | 85% | 90% |
| interpolator/ | 78% | 85% |
| audio_manager/ | 65% | 75% |
| upscaler/ | 82% | 90% |
| security/ | 91% | 95% |
| orchestrator.py | 88% | 95% |
| **Overall** | **81%** | **90%** |

---

## 🚀 Development Workflow

### 1. Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/shashankpc7746/LoRA_TextToVision.git
cd LoRA_TextToVision

# Create virtual environment
python -m venv gurukul-lora-env
source gurukul-lora-env/bin/activate  # Linux/Mac
# OR
.\gurukul-lora-env\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements-dev.txt  # Includes test/dev tools
pip install -r requirements-runtime.txt  # Core dependencies

# Verify GPU setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run health check
python -c "
from orchestrator import get_orchestrator
orch = get_orchestrator()
print(orch.get_statistics())
"
```

### 2. Making Changes

```bash
# Create feature branch
git checkout -b feature/new-interpolation-method

# Make changes
# ... edit files ...

# Run tests
pytest tests/ -v

# Check code quality
flake8 .
black .  # Auto-format
mypy .   # Type checking

# Commit changes
git add .
git commit -m "feat(interpolator): add new interpolation method

- Implemented XYZ interpolation algorithm
- Improved temporal consistency by 15%
- Added unit tests with 95% coverage"

# Push and create PR
git push origin feature/new-interpolation-method
```

### 3. Code Review Checklist

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code coverage ≥ 80% for new code
- [ ] Type hints added for all functions
- [ ] Docstrings follow NumPy style
- [ ] Performance benchmarks included (if applicable)
- [ ] Security implications reviewed (if touching sensitive areas)
- [ ] Documentation updated (`Documentation/`)
- [ ] Backwards compatibility maintained

### 4. Branching Strategy

```
main
├── develop                   # Integration branch
│   ├── feature/new-feature   # Feature branches
│   ├── bugfix/issue-123      # Bug fix branches
│   └── task/task-11          # Task implementation branches
└── hotfix/critical-fix       # Emergency production fixes
```

**Branch Naming**:
- `feature/description`: New features
- `bugfix/issue-number`: Bug fixes
- `task/task-number`: Task implementations
- `hotfix/description`: Critical production fixes

---

## 📊 Performance Monitoring

### Metrics Collection

The system automatically collects metrics via:
- `AnimateDiff/performance_tracker.py`: Per-component timing
- `insightflow_client.py`: Telemetry to InsightFlow
- `audit_logger.py`: Audit trail (JSONL format)

**Example: Performance Tracking**:
```python
from AnimateDiff.performance_tracker import performance_tracker

# Start tracking
performance_tracker.start_tracking("video_generation")

# ... perform video generation ...

# End tracking
metrics = performance_tracker.end_tracking("video_generation", {
    "prompt": "...",
    "quality": 0.87,
    "cost": 0.45
})

# Save metrics
performance_tracker.save_metrics("metrics/2025-11-12.json")
```

### Viewing Metrics

```python
# Load and analyze metrics
import json

with open("metrics/2025-11-12.json") as f:
    metrics = json.load(f)

print(f"Total generations: {len(metrics['operations'])}")
print(f"Avg duration: {sum(op['duration'] for op in metrics['operations']) / len(metrics['operations']):.1f}s")
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. GPU Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
os.environ['BATCH_SIZE'] = '2'

# Enable gradient checkpointing
os.environ['GRADIENT_CHECKPOINTING'] = '1'

# Clear cache
import torch
torch.cuda.empty_cache()

# Use smaller resolution
result = await generate_video(prompt, resolution=384)  # Instead of 512
```

#### 2. Watermark Not Detected

**Symptoms**: `detect_provenance.py` shows "❌ No watermark detected"

**Root Cause**: FFmpeg metadata stripped during video re-encoding (5 bugs discovered Nov 8, 2025)

**Common Causes**:
1. Missing `-movflags +use_metadata_tags` in FFmpeg commands
2. Using `-c copy` without metadata flags
3. `-map_metadata` only copies standard tags, not custom MP4 tags
4. Multi-stage pipelines stripping metadata at each re-encoding step
5. H.264/H.265 encoding without explicit metadata flags

**Solution**:
```bash
# ❌ WRONG - Strips custom metadata tags:
ffmpeg -i input.mp4 -c copy output.mp4
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4

# ✅ CORRECT - Preserves custom metadata:
ffmpeg -i input.mp4 -c copy -movflags +use_metadata_tags output.mp4
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -movflags +faststart+use_metadata_tags output.mp4

# ✅ For multi-stage pipelines (audio restoration):
ffmpeg -i temp.mp4 -i audio.wav -c:v copy -c:a aac \
  -map 0:v -map 1:a -map_metadata 0 \
  -metadata lora_adapter="indigenous_v1.0" \
  -metadata watermark_version="1.0" \
  -movflags +use_metadata_tags \
  output.mp4
```

**Verification**:
```bash
# Check if watermark is present
ffprobe -v quiet -print_format json -show_format output.mp4 | grep -A5 tags

# Should see:
# "tags": {
#   "lora_adapter": "indigenous_v1.0",
#   "watermark_version": "1.0",
#   ...
# }
```

**Critical Lessons**:
- **ALWAYS** use `-movflags +use_metadata_tags` when writing MP4 files with custom tags
- Apply metadata flags at **EVERY** re-encoding step in multi-stage pipelines
- `-map_metadata` alone is insufficient for custom tags
- Test with `ffprobe` after each pipeline stage

**Full Bug Details**: See `Documentation/ERRORS_AND_BUGS_LOG.md` - Task 10 section  
**Timeline**: 4-hour debugging session, 5 cascading bugs, 5 commits (Nov 8, 2025)

#### 3. Poor Lip-sync Quality

**Symptoms**: Lip-sync confidence < 0.7, visible audio delay

**Solutions**:
```python
# Adjust audio preprocessing
from audio_manager import EnhancedSadTalker

sadtalker = EnhancedSadTalker()
result = sadtalker.generate_lipsync(
    video_path="...",
    audio_path="...",
    preprocess_audio=True,  # Enable audio cleanup
    sync_threshold=0.85     # Higher quality threshold
)
```

#### 4. Slow Generation Times

**Symptoms**: Generation taking > 5 minutes

**Diagnosis**:
```python
from AnimateDiff.performance_tracker import performance_tracker

# Check per-component timing
metrics = performance_tracker.get_metrics()
for component, duration in metrics.items():
    print(f"{component}: {duration:.1f}s")
```

**Solutions**:
- Reduce inference steps: `inference_steps=20` instead of 50
- Disable upscaling for preview: `upscale_enabled=False`
- Use cloud fallback: `prefer_local=False`

---

## 🔌 Extension Points

### Adding New Components

#### 1. New Interpolation Method

```python
# interpolator/custom_interpolator.py

from interpolator.base_interpolator import BaseInterpolator

class CustomInterpolator(BaseInterpolator):
    def __init__(self):
        super().__init__()
        # Initialize your model
    
    def interpolate_frames(self, frame1, frame2, num_intermediate):
        """Generate intermediate frames between frame1 and frame2"""
        # Your interpolation logic
        return intermediate_frames
    
    def interpolate_video(self, input_path, target_fps, output_path):
        """Interpolate entire video"""
        # Your video processing logic
        return output_path
```

**Register in orchestrator**:
```python
# orchestrator.py

from interpolator.custom_interpolator import CustomInterpolator

class ProductionOrchestrator:
    def __init__(self):
        self.interpolator = CustomInterpolator()  # Use custom interpolator
```

#### 2. New Quality Metric

```python
# motion_controller/custom_metric.py

def compute_custom_quality(video_path):
    """
    Compute custom quality metric
    
    Returns:
        float: Quality score (0-1)
    """
    # Your quality assessment logic
    return quality_score
```

**Integrate in RL optimizer**:
```python
# motion_controller/rl_optimizer.py

from motion_controller.custom_metric import compute_custom_quality

class RLOptimizer:
    def compute_reward(self, result):
        quality = compute_custom_quality(result['output_path'])
        # Use in reward calculation
        return quality * 10.0 - cost * 5.0
```

#### 3. New Security Feature

```python
# security/custom_feature.py

def apply_custom_security(video_path, metadata):
    """
    Apply custom security measure
    
    Args:
        video_path: Path to video file
        metadata: Security metadata dict
    
    Returns:
        str: Path to secured video
    """
    # Your security implementation
    return secured_video_path
```

**Register in security pipeline**:
```python
# AnimateDiff/unified_video_generator.py

from security.custom_feature import apply_custom_security

# After watermarking
secured = apply_custom_security(watermarked_video, {
    "build_id": build_id,
    "custom_field": "value"
})
```

---

## 📚 Additional Resources

### Documentation Structure

```
Documentation/
├── Tasks/                          # Task implementation docs
│   ├── Task-1-README.md            # LoRA fine-tuning
│   ├── Task-2-README.md            # AnimateDiff integration
│   ├── Task-3-README.md            # RIFE interpolation
│   ├── Task-4-README.md            # SadTalker lip-sync
│   ├── Task-5-README.md            # ESRGAN upscaling
│   ├── Task-6-README.md            # RL optimization
│   ├── Task-7-README.md            # Yotta fallback
│   ├── Task-8-README.md            # API development
│   ├── Task-9-README.md            # Production readiness
│   ├── Task-10-README.md           # Security hardening
│   └── TASK-10-COMPLETION-VERIFICATION.md
├── Reports/                        # Task completion reports (PDFs)
│   ├── Task-1-Report.pdf
│   ├── Task-2-Report.pdf
│   └── ...
└── DEVELOPER_HANDBOOK.md           # This file
```

### External Links

- **Stable Diffusion**: https://github.com/Stability-AI/stablediffusion
- **AnimateDiff**: https://github.com/guoyww/AnimateDiff
- **RIFE**: https://github.com/hzwer/ECCV2022-RIFE
- **SadTalker**: https://github.com/OpenTalker/SadTalker
- **Real-ESRGAN**: https://github.com/xinntao/Real-ESRGAN
- **PyTorch**: https://pytorch.org/docs/

### Team Contacts

| Role | Name | Focus Area |
|------|------|------------|
| Lead Developer | Shashank Gupta | Overall architecture, RL optimization |
| Security Engineer | - | Task 10 implementation, CI/CD |
| ML Engineer | - | Model fine-tuning, quality optimization |
| DevOps | Rishabh | Cloud infrastructure, deployment |

---

## 🎯 Quick Command Reference

```bash
# Development
pytest tests/ -v                    # Run all tests
pytest tests/task10/ -v            # Run Task 10 tests
python -m black .                   # Format code
python -m flake8 .                  # Lint code

# Generation
python AnimateDiff/unified_video_generator.py lesson_space_adventure.json realistic 1

# Detection
python tools/detect_provenance.py "video.mp4"

# Monitoring
python -c "from orchestrator import get_orchestrator; print(get_orchestrator().get_statistics())"

# Deployment
docker build -t loratv-production .
docker run --gpus all -p 8001:8001 loratv-production
```

---

**Last Updated**: November 12, 2025  
**Version**: 1.0.0  
**Maintainer**: Development Team

*For questions or issues, check existing documentation in `Documentation/Tasks/` or create a GitHub issue.*
