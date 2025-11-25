# 🎬 TTV Studio - Complete Handover Master Document

**Project:** LoRA_TextToVision (TTV Studio)  
**Version:** 2.0.0  
**Last Updated:** November 25, 2025  
**Prepared By:** Shashank Gupta  
**For:** Next Engineer / Team Continuity  
**Status:** Production Ready ✅

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Important Concepts](#important-concepts)
4. [Setup Guide](#setup-guide)
5. [How to Extend / Modify](#how-to-extend--modify)
6. [Best Practices](#best-practices)
7. [Quick Reference](#quick-reference)

---

## 🎯 System Overview

### What is TTV Studio?

**TTV Studio** (Text-to-Vision) is an enterprise-grade AI video generation platform that transforms text prompts and educational scripts into high-quality, cinematic videos. It combines multiple AI technologies into a seamless production pipeline.

**Project Name:** "Gurukul" (brand name only - supports ANY educational content, not limited to specific themes)

### Core Capabilities

**Input:** Text prompt or lesson JSON  
**Output:** Professional 1080p video with audio, subtitles, watermarking, and security features

**What Makes TTV Studio Unique:**
- ✅ **Multi-stage Pipeline:** 7 processing stages (LoRA → Animation → Interpolation → Audio → Upscaling → Security)
- ✅ **Intelligent Adaptation:** RL-based quality/cost/latency optimization
- ✅ **Production Security:** Watermarking, fingerprinting, signing, audit logging
- ✅ **Story Intelligence:** Scene graph, narrative sequencing, character consistency, emotion tracking
- ✅ **Robust Fallback:** GPU tier management + cloud escalation
- ✅ **95%+ Test Coverage:** 152 tests passing, production-ready

### High-Level Pipeline Flow

```
┌─────────────────┐
│  Text Prompt    │  "A wise teacher explains quantum physics"
│  or Lesson JSON │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Story Analysis (Task 11 - TTV Intelligence)       │
│ - Analyze full story for character gender, roles           │
│ - Build scene graph (NetworkX) for entity tracking         │
│ - Parse narrative structure (story beats, character arcs)  │
│ - Initialize emotion states for characters                 │
│ - Condense narration by 20-30% (reduce looping)           │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Keyframe Generation (Task 1 - LoRA Adapter)       │
│ - Fine-tuned Stable Diffusion (domain-specific LoRA)       │
│ - Generate 512x512 keyframes (8-16 frames)                 │
│ - Enhanced prompts with character consistency              │
│ Output: Static images with consistent characters           │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Animation (Task 2 - AnimateDiff)                  │
│ - Convert keyframes → animated video (8fps, 2s clips)      │
│ - Apply cinematic flow (pan, zoom, dolly movements)        │
│ - Emotion-based motion intensity adjustments                │
│ Output: 512x512 @ 8fps animated clips                      │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: Frame Interpolation (Task 3 - RIFE)               │
│ - Smooth motion: 8fps → 24fps (3x interpolation)           │
│ - AI-powered frame generation between keyframes            │
│ Output: 512x512 @ 24fps smooth video                       │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: Audio & Lip-Sync (Task 4 - SadTalker)            │
│ - Text-to-speech generation                                │
│ - Lip-sync animation synchronized with audio               │
│ - Smart video extension (SlowMo + Freeze, NO looping)      │
│ Output: 512x512 @ 24fps with synchronized audio            │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 6: Upscaling (Task 5 - Real-ESRGAN)                 │
│ - AI upscaling: 512x512 → 1920x1080 (4x)                  │
│ - Maintain quality, reduce artifacts                       │
│ Output: 1080p @ 24fps HD video                            │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 7: Security & Export (Task 10 - Watermarking)       │
│ - Invisible watermark (FFmpeg metadata, 11 tags)           │
│ - Visible watermark (BHI logo, 35% opacity)                │
│ - Content fingerprinting (SHA256 + BLAKE2b + perceptual)   │
│ - KSML-compliant audit logging                             │
│ Output: Final 1080p video (H.264, 8000k bitrate)          │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Video    │  professional_video_complete.mp4
│  + Audit Logs   │  logs/audit/audit_YYYYMMDD.jsonl
└─────────────────┘
```

### Performance Metrics (Production)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Avg Latency** | 145s | <180s | ✅ 19% faster |
| **Quality (VMAF)** | 0.87 | >0.80 | ✅ 9% better |
| **Cost per Video** | $0.08 | <$0.10 | ✅ 20% cheaper |
| **Success Rate** | 97% | >95% | ✅ 2% higher |
| **Concurrent Users** | 50 | 50 | ✅ Target met |
| **Test Coverage** | 95%+ | >90% | ✅ 5% higher |

---

## 🏗️ Architecture

### Component Map

```
TTV Studio Architecture (11 Tasks Complete)
│
├─── Foundation Layer (Tasks 1-3)
│    ├── adapters/               LoRA fine-tuning & keyframe generation
│    ├── AnimateDiff/            Animation engine with cinematic flow
│    └── interpolator/           RIFE frame interpolation
│
├─── Enhancement Layer (Tasks 4-5)
│    ├── audio_manager/          SadTalker lip-sync integration
│    └── upscaler/              Real-ESRGAN 1080p upscaling
│
├─── Optimization Layer (Tasks 6-7)
│    ├── motion_controller/     RL parameter optimization
│    └── yotta_fallback.py      Cloud fallback orchestration
│
├─── Production Layer (Tasks 8-9)
│    ├── AnimateDiff_API/       REST API endpoints (FastAPI)
│    ├── orchestrator.py        Main pipeline coordinator
│    ├── docker-compose.yml     Multi-GPU deployment
│    └── tests/                 Comprehensive test suite (152 tests)
│
├─── Security Layer (Task 10)
│    ├── security/              Watermarking, signing, encryption
│    ├── audit_logger.py        KSML-compliant audit logging
│    └── tools/detect_provenance.py  Watermark detection
│
└─── Intelligence Layer (Task 11)
     └── AnimateDiff/adaptive_engine/  TTV Studio Intelligence (8 modules)
          ├── story_context_parser.py      Story NLP + text condensation
          ├── identity_memory.py           Character identity tracking
          ├── scene_memory_core.py         Scene graph (NetworkX)
          ├── narrative_sequencer_v1.py    Story beats + character arcs
          ├── emotion_controller.py        Emotion-motion coupling
          ├── smart_video_extender.py      Smart extension (NO RIFE)
          ├── cinematic_transition_core.py Cinematic transitions
          └── Extended audit_logger.py     26 TTV metrics logging
```

### GPU Resource Allocation

**Hardware Setup:**
- Primary GPU: NVIDIA RTX 3080 (10GB VRAM)
- Secondary GPU: NVIDIA RTX 3060 (8GB VRAM)
- Fallback: Yotta Cloud (unlimited, $0.15/min)

**Allocation Strategy:**

| Component | Preferred GPU | VRAM Required | Execution Time | Fallback |
|-----------|---------------|---------------|----------------|----------|
| **LoRA Training** | RTX 3080 | 8GB | ~15 min | Yotta Cloud |
| **Keyframe Gen** | RTX 3080 | 8GB | ~30s | RTX 3060 |
| **Animation** | RTX 3060 | 6GB | ~60s | RTX 3080 |
| **Interpolation** | RTX 3060 | 4GB | ~40s | CPU (slow) |
| **Upscaling** | RTX 3080 | 8GB | ~50s | Skip (optional) |
| **Audio/Lip-sync** | CPU | N/A | ~45s | Always CPU |

**Smart Routing Logic** (`yotta_fallback.py`):
```python
if local_gpu_available and vram_sufficient:
    use_local_gpu()
elif office_gpu_available:
    use_office_gpu()
elif concurrent_requests > 3:
    escalate_to_yotta_cloud()
else:
    wait_and_retry()
```

### Data Flow & File Structure

**Input Files:**
```
AnimateDiff/lessons/
├── lesson_comprehensive_1.json    Sample lesson (temple story)
├── lesson_comprehensive_2.json    Sample lesson (forest journey)
└── lesson_space_adventure.json    Custom lesson example
```

**Output Structure:**
```
storage/
└── YYYY-MM-DD/
    ├── LessonName_style_complete.mp4      Final video
    ├── LessonName_style_preview.mp4       Preview (low-res)
    ├── LessonName_audio.wav               Extracted audio
    └── LessonName_subtitles.srt           Subtitle file
```

**Logs & Metrics:**
```
logs/
├── audit/
│   └── audit_YYYYMMDD.jsonl               KSML-compliant audit trail
├── performance/
│   └── metrics_YYYYMMDD.json              Performance metrics
└── errors/
    └── errors_YYYYMMDD.log                Error logs
```

### Integration Points

**1. InsightFlow Telemetry Integration**
```python
# File: insightflow_client.py
from insightflow_client import InsightFlowClient

client = InsightFlowClient()
client.log_event("video_generation_complete", {
    "quality_score": 0.87,
    "latency": 145.2,
    "cost": 0.08
})
```

**2. KSML Audit Logging**
```python
# File: audit_logger.py
logger.log_operation(
    operation="video_generation",
    status="success",
    ksml_token={
        "token": "ksml_video_complete",
        "lineage": {...},
        "provenance": {...}
    }
)
```

**3. Security Watermarking**
```python
# File: security/watermark.py
from security import embed_watermark

watermarked_video = embed_watermark(
    video_path="output.mp4",
    build_id="build_20251125_abc123",
    metadata={"project": "gurukul", "version": "2.0.0"}
)
```

---

## 💡 Important Concepts

### 1. KSML Tokens (Knowledge Security & Machine Learning)

**What:** KSML is a framework for tracking lineage and provenance of AI-generated content.

**Why:** Ensures every generated video has:
- Verifiable origin (which models, which parameters)
- Tamper-evident audit trail
- Attribution and accountability

**How It Works:**
```json
{
  "ksml_token": "ksml_video_complete",
  "intent": "video_generation",
  "karma_state": "completed",
  "lineage": {
    "lora_model": "gurukul_v2",
    "base_model": "stable-diffusion-2-1",
    "prompt": "Ancient temple...",
    "generation_params": {
      "steps": 50,
      "guidance_scale": 7.5
    }
  },
  "provenance": {
    "build_id": "build_20251125_abc123",
    "timestamp": "2025-11-25T10:30:00Z",
    "watermark_hash": "sha256:abc..."
  }
}
```

**Where Used:**
- `audit_logger.py`: Every operation logged with KSML token
- `security/watermark.py`: Watermark contains KSML lineage
- `insightflow_client.py`: Telemetry includes KSML context

### 2. Lineage Tracking

**Purpose:** Track every component and decision that influenced the final video.

**Tracked Elements:**
- Model versions (LoRA checkpoint, base model)
- Hyperparameters (steps, CFG scale, sampler)
- Input prompts (original + enhanced)
- Processing decisions (upscaling enabled, interpolation FPS)
- GPU used (local RTX 3080 vs Yotta Cloud)
- Execution time per stage

**Benefits:**
- Reproducibility: Regenerate exact same video
- Debugging: Trace quality issues to specific stages
- Compliance: Prove legitimate generation process
- Analytics: Understand what parameters produce best results

### 3. TTV Intelligence (Task 11)

**Scene Graph Memory:**
- Built with NetworkX (directed graph)
- Tracks entities (characters, objects, locations) across scenes
- Maintains temporal relationships (Scene A → Scene B)
- Enables cross-scene consistency

**Narrative Sequencing:**
- Story beat parser (Setup, Rising Action, Climax, Falling Action, Resolution)
- Character arc tracking (Introduction → Transformation → New Equilibrium)
- Dialogue flow analysis
- Pacing and tension curve tracking

**Emotion-Motion Coupling:**
- Characters have emotional states per scene
- Emotions influence motion intensity (joy = +30%, sadness = -40%)
- Smooth emotion transitions between scenes
- Micro-expressions with keyframe timing

**Smart Video Extension:**
- Problem: Short clips looped 3x = repetitive
- Solution: SlowMo (1.5x) + Smart Freeze with zoom
- NO RIFE for extension (avoids black screens)
- Result: Natural 30-40% reduction in perceived repetition

### 4. Identity Embeddings

**Purpose:** Ensure character consistency across scenes.

**How It Works:**
1. Extract face embeddings from generated keyframes
2. Store in `identity_memory.py` cache (pickle format)
3. For new scenes, verify character similarity (>0.7 threshold)
4. Alert if identity drift detected

**Technology:**
- OpenCV Haar Cascade for face detection
- Histogram + grayscale image (4352-dim embedding)
- Cosine similarity for matching
- Persistent cache across video generation sessions

### 5. Motion Controllers

**RL-Based Optimization:**
- Reinforcement learning agent learns optimal parameters
- Balances quality vs cost vs latency
- Rewards: High VMAF + Low cost + Low latency
- State: GPU load, queue depth, prompt complexity
- Action: Adjust steps, guidance scale, enable/disable upscaling

**Current Strategy:**
```python
if prompt_complexity > 0.7 and gpu_available:
    increase_steps()  # Better quality
elif queue_depth > 5:
    reduce_steps()    # Faster processing
elif cost_budget_remaining < 0.5:
    disable_upscaling()  # Save cost
```

### 6. Two-Pass Upscaling

**Why Two Passes:**
- Single 4x upscale (512 → 2048) = poor quality
- Two 2x upscales (512 → 1024 → 2048) = better results

**Implementation:**
```python
# Pass 1: 512 → 1024
intermediate = upscaler.upscale(frame_512, scale=2)

# Pass 2: 1024 → 2048 (optional, for 2K/4K)
final = upscaler.upscale(intermediate, scale=2)
```

**Performance:**
- 1080p (single pass): ~50s, VMAF 0.85
- 2K (two-pass): ~120s, VMAF 0.90

### 7. Telemetry Integration (InsightFlow)

**Purpose:** Real-time monitoring and analytics.

**Metrics Tracked (26 total):**

**Story Analysis (4 metrics):**
- character_count, gender_resolved
- text_condensation_percent, enhanced_prompts_count

**Scene Graph (4 metrics):**
- total_scenes, total_entities
- avg_entities_per_scene, total_edges

**Narrative (6 metrics):**
- story_beats, character_arcs
- avg_tension, peak_tension, pacing_score

**Emotion (3 metrics):**
- emotion_changes, avg_motion_intensity
- emotion_distribution

**Extension (5 metrics):**
- clips_extended, clips_trimmed, total_clips
- avg_extension_duration, method

**Quality (4 metrics):**
- audio_video_sync_diff, total_duration
- fps, bitrate

**Where Logged:**
- `logs/audit/audit_YYYYMMDD.jsonl` (immutable append-only)
- Sent to InsightFlow dashboard (if configured)

---

## 🚀 Setup Guide

### Prerequisites

**System Requirements:**
- OS: Windows 10/11 or Linux (Ubuntu 20.04+)
- Python: 3.10 or 3.11 (NOT 3.12 - compatibility issues)
- CUDA: 11.8+ (for GPU acceleration)
- RAM: 32GB+ recommended
- Storage: 100GB+ free space

**GPU Requirements:**
- Minimum: NVIDIA RTX 3060 (8GB VRAM)
- Recommended: NVIDIA RTX 3080/3090 (10GB+ VRAM)
- Optional: Dual GPU setup for parallel processing

**Software Dependencies:**
- Git (for repository cloning)
- FFmpeg (for video processing)
- Conda or venv (for Python environment)

### Step 1: Clone Repository

```bash
git clone https://github.com/shashankpc7746/LoRA_TextToVision.git
cd LoRA_TextToVision

# Checkout production branch
git checkout task_quality_harden_secure
```

### Step 2: Create Python Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n ttv-studio python=3.10
conda activate ttv-studio
```

**Option B: Using venv**
```bash
python -m venv gurukul-lora-env

# Windows
.\gurukul-lora-env\Scripts\Activate.ps1

# Linux/Mac
source gurukul-lora-env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Runtime dependencies (production)
pip install -r requirements-runtime.txt

# Development dependencies (testing, linting)
pip install -r requirements-dev.txt

# Verify installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Should print: CUDA Available: True
```

**Key Dependencies:**
- torch==2.0.1+cu118
- diffusers==0.21.4
- transformers==4.34.0
- opencv-python==4.8.1.78
- moviepy==1.0.3
- networkx==3.2 (Task 11 - scene graph)
- fastapi==0.104.1 (API)
- pytest==7.4.3 (testing)

### Step 4: Download Models

**Required Models (Total: ~25GB):**

```bash
# AnimateDiff motion modules
python AnimateDiff/download_animatediff_models.py

# Stable Diffusion base model (automatic via diffusers)
# Downloaded on first run to: ~/.cache/huggingface/

# LoRA adapters
# Located in: adapters/gurukul_lora/
```

**Model Locations:**
```
AnimateDiff/models/
├── Motion_Module/
│   └── mm_sd_v15_v2.ckpt          (1.7GB)
├── DreamBooth_LoRA/
│   └── realistic_vision_v2.safetensors  (2.1GB)
└── ControlNet/
    └── control_v11p_sd15_openpose.pth  (1.4GB)
```

### Step 5: Configure Environment Variables

Create `.env` file in project root:

```bash
# .env (DO NOT COMMIT TO GIT)

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=4

# Storage Paths
STORAGE_PATH=./storage
CACHE_PATH=./cache
LOGS_PATH=./logs

# Yotta Cloud (Optional - for fallback)
YOTTA_API_KEY=your_yotta_api_key_here
YOTTA_ENDPOINT=https://api.yotta.bhiv.com

# Security (Task 10)
WATERMARK_ENABLED=true
BUILD_ID_PREFIX=build_

# InsightFlow Telemetry (Optional)
INSIGHTFLOW_ENABLED=false
INSIGHTFLOW_ENDPOINT=https://insight.bhiv.com

# Development Mode
DEBUG=false
LOG_LEVEL=INFO
```

**⚠️ CRITICAL:** Never commit `.env` file to Git! Add to `.gitignore`.

### Step 6: Verify Installation

```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Expected output: 152 passed in ~45s
```

### Step 7: Generate First Video

```bash
cd AnimateDiff

# Generate video from lesson file
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1

# Expected output after ~2.5 minutes:
# ✅ SUCCESS! Complete video created
# 📁 Output: storage/2025-11-25/The_Temple_Mystery_realistic_complete.mp4
```

### Step 8: Docker Setup (Production)

```bash
# Build Docker image
docker build -t ttv-studio:2.0.0 .

# Run with GPU support
docker run --gpus all \
  -p 8001:8001 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  ttv-studio:2.0.0

# Or use docker-compose
docker-compose up -d
```

### Step 9: Verify Security Features

```bash
# Generate watermarked video
cd AnimateDiff
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1

# Detect watermark
cd ..
python tools/detect_provenance.py "storage/2025-11-25/The_Temple_Mystery_realistic_complete.mp4"

# Expected output:
# ✅ Watermark detected!
#    Build ID: build_20251125_abc123
#    Timestamp: 2025-11-25T10:30:00Z
# ✅ VERIFIED - File has valid provenance
```

### Common Setup Issues

**Issue 1: CUDA not available**
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

**Issue 2: Out of memory (OOM)**
```python
# Reduce batch size in config
# File: adapters/config.json
{
    "batch_size": 2,  # Default: 4, reduce to 2 or 1
    "gradient_accumulation_steps": 8  # Increase to compensate
}
```

**Issue 3: FFmpeg not found**
```bash
# Windows (using Chocolatey)
choco install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Verify
ffmpeg -version
```

---

## 🔧 How to Extend / Modify

### Adding New Modules

**Example: Add a new video filter module**

**Step 1: Create module structure**
```bash
mkdir filters
cd filters
touch __init__.py
touch color_grading.py
```

**Step 2: Implement module**
```python
# filters/color_grading.py
from typing import Optional
import cv2
import numpy as np

class ColorGrading:
    """Apply cinematic color grading to video frames"""
    
    def __init__(self, preset: str = "warm"):
        """
        Args:
            preset: Color preset (warm, cool, vintage, dramatic)
        """
        self.preset = preset
        self.lut = self._load_lut(preset)
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply color grading to single frame
        
        Args:
            frame: RGB frame (H, W, 3)
            
        Returns:
            Graded frame (H, W, 3)
        """
        # Apply LUT transformation
        graded = cv2.LUT(frame, self.lut)
        return graded
    
    def _load_lut(self, preset: str):
        # Load or generate color LUT
        # Implementation details...
        pass
```

**Step 3: Add tests**
```python
# tests/filters/test_color_grading.py
import pytest
from filters.color_grading import ColorGrading
import numpy as np

def test_color_grading_warm():
    grader = ColorGrading(preset="warm")
    frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = grader.apply(frame)
    
    assert result.shape == frame.shape
    assert result.dtype == np.uint8

def test_color_grading_presets():
    presets = ["warm", "cool", "vintage", "dramatic"]
    for preset in presets:
        grader = ColorGrading(preset=preset)
        assert grader.preset == preset
```

**Step 4: Integrate into pipeline**
```python
# AnimateDiff/unified_video_generator.py
from filters.color_grading import ColorGrading

class UnifiedVideoGenerator:
    def __init__(self):
        # ... existing initialization ...
        self.color_grading = ColorGrading(preset="warm")
    
    def _enhance_clip(self, clip):
        # Apply color grading to each frame
        graded_frames = [
            self.color_grading.apply(frame)
            for frame in clip.iter_frames()
        ]
        return VideoClip(make_frame=lambda t: graded_frames[int(t * clip.fps)])
```

**Step 5: Update documentation**
```markdown
# Add to Documentation/Tasks/Task-12-Extensions.md
## Color Grading Module

**Location:** `filters/color_grading.py`
**Purpose:** Apply cinematic color grading presets
**Usage:**
    from filters.color_grading import ColorGrading
    grader = ColorGrading(preset="warm")
    graded_frame = grader.apply(frame)
```

### Updating LoRA Models

**Step 1: Prepare new training dataset**
```bash
# Organize images in dataset folder
mkdir -p datasets/new_lora_v3
cd datasets/new_lora_v3

# Add images (recommended: 20-100 images)
# Format: image_001.png, image_002.png, etc.
```

**Step 2: Train new LoRA adapter**
```python
from adapters import AdapterManager

manager = AdapterManager()

# Train new adapter
await manager.train_adapter(
    dataset_path="datasets/new_lora_v3",
    adapter_name="gurukul_v3",
    steps=1000,
    learning_rate=1e-4,
    batch_size=4
)

# Adapter saved to: adapters/gurukul_lora/adapters/gurukul_v3/
```

**Step 3: Test new adapter**
```python
# Generate test keyframes
keyframes = await manager.generate_keyframes(
    prompt="Ancient temple in misty mountains",
    num_keyframes=8,
    adapter="gurukul_v3"  # Use new adapter
)
```

**Step 4: Update default adapter**
```python
# File: adapters/config.json
{
    "default_adapter": "gurukul_v3",  # Changed from gurukul_v2
    "lora_rank": 64,
    "learning_rate": 1e-4
}
```

**Step 5: Validate in production**
```bash
# Generate full video with new adapter
cd AnimateDiff
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1

# Compare quality with previous version
python tools/compare_videos.py \
    storage/old_video.mp4 \
    storage/new_video.mp4 \
    --metrics vmaf ssim psnr
```

### Integrating New Telemetry Metrics

**Step 1: Define new metric**
```python
# File: audit_logger.py

def log_custom_metric(
    self,
    metric_name: str,
    metric_value: float,
    metric_category: str = "custom",
    ksml_token: Optional[Dict] = None
) -> str:
    """
    Log custom telemetry metric
    
    Args:
        metric_name: Name of metric (e.g., "frame_sharpness")
        metric_value: Numeric value
        metric_category: Category for grouping
        ksml_token: Optional KSML lineage
    
    Returns:
        entry_id: Unique audit log entry ID
    """
    entry = {
        "entry_id": self._generate_entry_id(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "operation": "custom_metric",
        "status": "success",
        "metadata": {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "category": metric_category
        }
    }
    
    if ksml_token:
        entry["ksml_compliance"] = ksml_token
    
    self._append_to_log(entry)
    return entry["entry_id"]
```

**Step 2: Collect metric during processing**
```python
# File: upscaler/esrgan_upscaler.py

def upscale(self, frame: np.ndarray) -> np.ndarray:
    upscaled = self._run_esrgan(frame)
    
    # Calculate sharpness metric
    sharpness = self._calculate_sharpness(upscaled)
    
    # Log to telemetry
    from audit_logger import get_audit_logger
    logger = get_audit_logger()
    logger.log_custom_metric(
        metric_name="upscaled_frame_sharpness",
        metric_value=sharpness,
        metric_category="quality"
    )
    
    return upscaled

def _calculate_sharpness(self, frame: np.ndarray) -> float:
    """Calculate Laplacian variance as sharpness measure"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()
```

**Step 3: Query metrics**
```python
# File: tools/query_metrics.py

import json
from pathlib import Path

def get_custom_metrics(metric_name: str, date: str = None):
    """
    Query custom metrics from audit logs
    
    Args:
        metric_name: Name of metric to query
        date: Date string (YYYYMMDD), defaults to today
    
    Returns:
        List of metric values with timestamps
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    
    log_file = Path(f"logs/audit/audit_{date}.jsonl")
    metrics = []
    
    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)
            if (entry.get("operation") == "custom_metric" and
                entry["metadata"]["metric_name"] == metric_name):
                metrics.append({
                    "timestamp": entry["timestamp"],
                    "value": entry["metadata"]["metric_value"]
                })
    
    return metrics

# Usage
sharpness_metrics = get_custom_metrics("upscaled_frame_sharpness")
avg_sharpness = sum(m["value"] for m in sharpness_metrics) / len(sharpness_metrics)
print(f"Average sharpness: {avg_sharpness:.2f}")
```

### Updating Security Lineage

**Scenario:** Add new watermark metadata field

**Step 1: Extend watermark metadata**
```python
# File: security/watermark.py

def embed_watermark(
    video_path: str,
    build_id: str,
    project_name: str = "gurukul",  # NEW FIELD
    model_version: str = None,       # NEW FIELD
    **kwargs
) -> str:
    """
    Embed invisible watermark with extended metadata
    
    Args:
        video_path: Path to video file
        build_id: Unique build identifier
        project_name: Project name (default: "gurukul")
        model_version: Model version used (e.g., "v2.0.0")
    """
    metadata_tags = {
        "BHIV_WATERMARK": "true",
        "BUILD_ID": build_id,
        "TIMESTAMP": datetime.utcnow().isoformat() + "Z",
        "PROJECT_NAME": project_name,          # NEW
        "MODEL_VERSION": model_version or "",  # NEW
        # ... existing tags ...
    }
    
    # Apply metadata with FFmpeg
    # ... implementation ...
```

**Step 2: Update detection tool**
```python
# File: tools/detect_provenance.py

def detect_watermark(video_path: str) -> Dict:
    """Detect watermark with extended fields"""
    metadata = extract_metadata(video_path)
    
    return {
        "detected": metadata.get("BHIV_WATERMARK") == "true",
        "build_id": metadata.get("BUILD_ID"),
        "timestamp": metadata.get("TIMESTAMP"),
        "project_name": metadata.get("PROJECT_NAME"),      # NEW
        "model_version": metadata.get("MODEL_VERSION"),    # NEW
        "verified": verify_signature(metadata)
    }
```

**Step 3: Update KSML lineage**
```python
# File: audit_logger.py

def log_ttv_intelligence(self, ..., model_info: Dict = None):
    """Extended with model information"""
    entry = {
        # ... existing fields ...
        "ksml_compliance": {
            "token": "ksml_ttv_complete",
            "lineage": {
                "lesson": lesson_name,
                "style": style,
                "model": {                           # NEW
                    "base": "stable-diffusion-2-1",
                    "lora": model_info.get("lora_version", "gurukul_v2"),
                    "version": model_info.get("version", "2.0.0")
                }
            }
        }
    }
```

---

## 📚 Best Practices

### Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `story_context_parser.py`)
- Test files: `test_<module_name>.py` (e.g., `test_story_context_parser.py`)
- Config files: `lowercase.json` or `lowercase.yaml`
- Documentation: `UPPERCASE.md` or `Title-Case.md`

**Classes:**
- PascalCase (e.g., `StoryContextParser`, `SceneMemoryCore`)
- Descriptive names indicating purpose

**Functions:**
- snake_case (e.g., `analyze_story`, `build_scene_graph`)
- Verbs indicating action (get, set, calculate, generate, process)

**Variables:**
- snake_case (e.g., `character_count`, `video_duration`)
- Descriptive, avoid abbreviations except common ones (fps, gpu, api)

**Constants:**
- UPPER_SNAKE_CASE (e.g., `MAX_CONCURRENT_JOBS`, `DEFAULT_FPS`)

**Example:**
```python
# Good
class VideoProcessor:
    MAX_RESOLUTION = 1920
    
    def process_video(self, input_path: str) -> str:
        video_duration = self._get_duration(input_path)
        return output_path

# Avoid
class vidProc:
    maxRes = 1920
    
    def procVid(self, inp: str) -> str:
        dur = self.getDur(inp)
        return out
```

### Logging Standards

**Use Python's logging module (NOT print statements):**

```python
import logging

logger = logging.getLogger(__name__)

# Log levels (in order of severity)
logger.debug("Detailed debugging information")     # Development only
logger.info("General informational message")        # Normal operations
logger.warning("Warning: potential issue")          # Unexpected but handled
logger.error("Error occurred but recoverable")      # Error with recovery
logger.critical("Critical failure, cannot continue") # Fatal error

# Example usage
def generate_video(prompt: str):
    logger.info(f"Starting video generation: {prompt}")
    
    try:
        keyframes = generate_keyframes(prompt)
        logger.info(f"Generated {len(keyframes)} keyframes")
    except Exception as e:
        logger.error(f"Keyframe generation failed: {e}", exc_info=True)
        raise
    
    logger.info("Video generation complete")
```

**Structured logging for telemetry:**
```python
# Use audit_logger for production events
from audit_logger import get_audit_logger

logger = get_audit_logger()
logger.log_operation(
    operation="video_generation",
    status="success",
    metadata={
        "prompt": prompt,
        "duration": 145.2,
        "quality": 0.87
    }
)
```

### Commit Structuring

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code restructuring (no behavior change)
- `perf`: Performance improvements
- `chore`: Build/tooling changes

**Examples:**

```bash
# Feature
git commit -m "feat(task11): add scene memory core with NetworkX graph

- Implement SceneMemoryCore class with 13 query methods
- Add entity tracking across temporal scenes
- Include cache persistence with pickle
- Add comprehensive tests (18/18 passing)

Closes #42"

# Bug fix
git commit -m "fix(watermark): resolve FFmpeg metadata stripping bug

Previously H.264 encoding stripped custom metadata tags.
Added -movflags +use_metadata_tags to preserve watermark.

Fixes #67
Testing: 100% watermark detection achieved"

# Documentation
git commit -m "docs(handbook): add TTV handover master document

Complete architecture guide for next engineer including:
- System overview and pipeline flow
- All 11 tasks documented
- Setup guide and troubleshooting
- Extension guide for new modules"
```

### Testing Workflow

**1. Write tests BEFORE implementation (TDD)**
```python
# tests/test_new_feature.py (WRITE THIS FIRST)
def test_video_filter_applies_correctly():
    filter = VideoFilter(effect="blur")
    frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = filter.apply(frame)
    
    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)  # Filter changed frame

# THEN implement feature to make test pass
```

**2. Run tests locally before committing**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_video_filter.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Open coverage report
# open htmlcov/index.html
```

**3. Test categories**
```python
# Unit test (single function/method)
def test_calculate_sharpness():
    frame = create_test_frame()
    sharpness = calculate_sharpness(frame)
    assert 0 <= sharpness <= 100

# Integration test (multiple components)
def test_video_pipeline():
    prompt = "Test scene"
    keyframes = generate_keyframes(prompt)
    animated = animate(keyframes)
    assert animated.duration > 0

# End-to-end test (complete system)
def test_full_generation():
    result = generate_video("Ancient temple")
    assert result["success"]
    assert os.path.exists(result["output_path"])
```

**4. Maintain >90% coverage**
```bash
# Check current coverage
pytest --cov=. --cov-report=term

# Target: 95%+ for new code
```

### Code Quality Checklist

**Before Every Commit:**

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code coverage >90% for modified files
- [ ] Type hints added for function signatures
- [ ] Docstrings added for public functions/classes
- [ ] No commented-out code (remove or explain)
- [ ] No hard-coded secrets (use environment variables)
- [ ] Logging used instead of print statements
- [ ] Error handling for expected failures
- [ ] Git commit message follows convention
- [ ] Documentation updated if public API changed

**Code Review Focus Areas:**

1. **Security:** No secrets exposed, input validation
2. **Performance:** No obvious bottlenecks, efficient algorithms
3. **Maintainability:** Clear naming, appropriate comments
4. **Testing:** Edge cases covered, failure scenarios tested
5. **Documentation:** Public APIs documented, complex logic explained

---

## ⚡ Quick Reference

### Common Commands

**Video Generation:**
```bash
# Generate from lesson JSON
cd AnimateDiff
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1

# Generate preview (faster, lower quality)
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1 --preview

# Generate with custom parameters
python generate_lesson_video_safe.py \
    lesson_custom.json \
    cinematic \
    1 \
    --fps 30 \
    --resolution 2048
```

**Testing:**
```bash
# All tests
pytest tests/ -v

# Task-specific tests
pytest tests/task11/ -v

# Single test file
pytest tests/task11/test_scene_memory.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

**Watermark Detection:**
```bash
# Detect watermark
python tools/detect_provenance.py "path/to/video.mp4"

# Batch detection
python tools/detect_provenance.py storage/2025-11-25/*.mp4
```

**Metrics & Monitoring:**
```bash
# View audit logs
cat logs/audit/audit_$(date +%Y%m%d).jsonl | jq

# Generate benchmarks dashboard
python tools/benchmarks_dashboard.py

# Query specific metrics
python tools/query_metrics.py --metric vmaf --date 20251125
```

**Docker:**
```bash
# Build
docker build -t ttv-studio:2.0.0 .

# Run
docker run --gpus all -p 8001:8001 ttv-studio:2.0.0

# Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### File Locations Quick Reference

| What | Where |
|------|-------|
| **Main pipeline** | `orchestrator.py` |
| **Video generator** | `AnimateDiff/unified_video_generator.py` |
| **LoRA training** | `adapters/adapter_trainer.py` |
| **Scene graph** | `AnimateDiff/adaptive_engine/scene_memory_core.py` |
| **Watermarking** | `security/watermark.py` |
| **Audit logging** | `audit_logger.py` |
| **Tests** | `tests/` (152 tests) |
| **Documentation** | `Documentation/` |
| **Lesson files** | `AnimateDiff/lessons/` |
| **Output videos** | `storage/YYYY-MM-DD/` |
| **Audit logs** | `logs/audit/audit_YYYYMMDD.jsonl` |
| **Config files** | `adapters/config.json`, `.env` |

### Troubleshooting Quick Fixes

**GPU out of memory:**
```python
# Reduce batch size
# File: adapters/config.json
{"batch_size": 1}  # Down from 4
```

**Watermark not detected:**
```bash
# Verify FFmpeg has metadata
ffprobe -v quiet -show_format video.mp4

# Check for BHIV_WATERMARK tag
```

**Tests failing:**
```bash
# Clear cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Reinstall dependencies
pip install -r requirements-runtime.txt --force-reinstall
```

**Video generation slow:**
```python
# Disable upscaling for testing
# File: orchestrator.py
enable_upscaling = False

# Or reduce interpolation
target_fps = 12  # Down from 24
```

### Important URLs

- **Repository:** https://github.com/shashankpc7746/LoRA_TextToVision
- **Documentation:** `Documentation/DEVELOPER_HANDBOOK.md`
- **Task Details:** `Documentation/Tasks/Task-1-README.md` through `Task-11-README.md`
- **User Guide:** `TTV_STUDIO_USER_GUIDE.md`

---

## 📞 Support & Contact

**For Next Engineer:**

If you encounter issues not covered in this document:

1. Check `Documentation/DEVELOPER_HANDBOOK.md` (900+ lines, comprehensive)
2. Review task-specific README in `Documentation/Tasks/`
3. Check `Documentation/ERRORS_AND_BUGS_LOG.md` for known issues
4. Read FAQ in `Documentation/Handover/FAQ_NEW_ENGINEER.md`
5. Review test files for usage examples

**Common Resources:**
- Architecture diagrams: `Documentation/Handover/ARCHITECTURE_DIAGRAMS.md`
- Current status: `Documentation/Handover/STATUS_REPORT.md`
- Demo walkthrough: `Documentation/Handover/DEMO_WALKTHROUGH.md`

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Next Review:** When major changes occur  
**Maintained By:** Project Lead / Next Engineer

---

*This document represents complete knowledge transfer for TTV Studio. All code, architecture, and operational knowledge captured for seamless continuity.* 🚀
