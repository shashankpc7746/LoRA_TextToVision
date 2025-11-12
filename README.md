# 🎬 LoRA_TextToVision

**Enterprise-Grade AI Video Generation Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> Transform text prompts into high-quality educational videos through AI-powered multi-stage pipeline

---

## 📚 Quick Links

- **[Developer Handbook](Documentation/DEVELOPER_HANDBOOK.md)** - Complete architecture & onboarding guide
- **[Production Deployment](README_PRODUCTION.md)** - API reference & deployment guide  
- **[Task Documentation](Documentation/Tasks/)** - Implementation details for all 10 tasks
- **[Test Suite](tests/)** - Comprehensive testing infrastructure
- **[Benchmarks Dashboard](tools/benchmarks_dashboard.py)** - Performance visualization

---

## 🎯 What is LoRA_TextToVision?

Enterprise-grade video generation platform that combines:

- **LoRA Fine-tuning**: Domain-specific Stable Diffusion adaptation
- **AnimateDiff**: Static keyframes → animated sequences
- **RIFE Interpolation**: 8fps → 24fps smooth motion
- **SadTalker Lip-sync**: Synchronized audio-visual generation
- **Real-ESRGAN Upscaling**: 512p → 1080p HD enhancement
- **RL Optimization**: Adaptive quality/cost/latency balancing
- **Cloud Fallback**: Intelligent Yotta cloud escalation
- **Production Security**: Watermarking, signing, audit logging

### Key Features

✅ **50+ concurrent users** with 97% success rate  
✅ **VMAF ≥0.87** average quality score  
✅ **145s average latency** (2.4 minutes per video)  
✅ **$0.08 cost per video** with cloud fallback  
✅ **100% watermark detection** with dual-layer security  
✅ **81% test coverage** with comprehensive validation  

---

## 🚀 Quick Start

### Installation

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
pip install -r requirements-runtime.txt

# Verify GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Generate Your First Video

```python
from orchestrator import generate_video

# Generate video
result = await generate_video(
    "A majestic eagle soaring through mountains",
    target_quality=0.8,
    max_cost_usd=1.0
)

print(f"Video generated: {result['final_result']['output_path']}")
```

### Using the Unified Generator (Lesson-based)

```bash
# Generate video from lesson file
cd AnimateDiff
python unified_video_generator.py lesson_space_adventure.json realistic 1

# Output: storage/2025-11-12/Space_Adventure_realistic_complete.mp4
```

---

## 🏗️ Architecture Overview

### Pipeline Flow

```
Text Prompt → LoRA Adapter → Keyframes (512x512)
                                    ↓
              AnimateDiff Animation (8fps, 2s)
                                    ↓
              RIFE Interpolation (24fps)
                                    ↓
              Audio + Lip-sync (SadTalker)
                                    ↓
              Upscaling (1080p ESRGAN)
                                    ↓
    Security Watermarking → Final Video (H.264)
```

### Module Structure

```
LoRA_TextToVision/
├── adapters/              # LoRA training & keyframe generation
├── AnimateDiff/           # Animation engine & cinematic flow
├── AnimateDiff_API/       # REST API endpoints
├── interpolator/          # RIFE frame interpolation
├── audio_manager/         # SadTalker lip-sync
├── upscaler/             # Real-ESRGAN upscaling
├── motion_controller/    # RL parameter optimization
├── security/             # Watermarking, signing, encryption (Task 10)
├── tests/                # Comprehensive test suite
│   ├── task9/            # Production readiness tests
│   ├── task10/           # Security tests
│   └── integration/      # End-to-end tests
├── Documentation/        # Complete documentation
│   ├── Tasks/            # Task-1 to Task-10 READMEs
│   ├── Reports/          # PDF reports for each task
│   └── DEVELOPER_HANDBOOK.md  # Architecture & hand-off guide
└── tools/                # Utilities (provenance detection, benchmarks)
```

---

## 📊 Performance Metrics

### Current Benchmarks (Production)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Concurrent Users** | 50 | 50 | ✅ |
| **Success Rate** | 95% | 97% | ✅ |
| **Avg Quality (VMAF)** | 0.80 | 0.87 | ✅ |
| **Avg Latency** | <180s | 145s | ✅ |
| **P95 Latency** | <300s | 280s | ✅ |
| **Cost per Video** | <$0.10 | $0.08 | ✅ |
| **Test Coverage** | 85% | 81% | ⚠️ |

### GPU Resource Allocation

| Component | GPU | VRAM | Duration |
|-----------|-----|------|----------|
| LoRA Training | RTX 3080 | 8GB | ~15min |
| Keyframe Gen | RTX 3080 | 8GB | ~30s |
| Animation | RTX 3060 | 8GB | ~60s |
| Interpolation | RTX 3060 | 6GB | ~40s |
| Upscaling | RTX 3080 | 8GB | ~50s |
| **Total Pipeline** | Dual GPU | 16GB | **~2.5min** |

---

## 🔒 Security Features (Task 10)

### Watermarking & Provenance

```python
from security import embed_watermark, compute_fingerprint
from tools.detect_provenance import detect_watermark

# Apply invisible watermark
watermarked = embed_watermark("video.mp4", build_id="build_20251112_123456")

# Compute fingerprint
fingerprint = compute_fingerprint(watermarked)

# Detect watermark
result = detect_watermark(watermarked)
print(result['build_id'])  # build_20251112_123456
```

### Detection on Other PCs

**Option 1: Using Detection Tool** (Recommended)
```bash
# Copy tools/detect_provenance.py to target PC
python tools/detect_provenance.py "video.mp4"

# Output:
# ✅ Watermark detected!
#    Build ID: build_20251112_123456
# ✅ VERIFIED - File has valid provenance
```

**Option 2: Using FFprobe** (Requires FFmpeg installation)
```powershell
# PowerShell (recommended for Windows)
ffprobe -v quiet -print_format json -show_format "video.mp4" | ConvertFrom-Json | Select-Object -ExpandProperty format | Select-Object -ExpandProperty tags | Format-List *

# Command Prompt (cmd.exe)
ffprobe -v quiet -show_entries format_tags=BHIV_WATERMARK,BUILD_ID -of default=noprint_wrappers=1 "video.mp4"
```

### Security Features

- ✅ **Invisible Watermark**: FFmpeg metadata (11 custom tags)
- ✅ **Visible Watermark**: BHI logo (35% opacity, bottom-right)
- ✅ **Content Fingerprinting**: SHA256 + BLAKE2b + perceptual hash
- ✅ **Artifact Signing**: Ed25519 signatures for models/adapters
- ✅ **Runtime Key Validation**: Core-signed 12-24h runtime keys
- ✅ **KSML Encryption**: AES-256-GCM with PBKDF2 key derivation
- ✅ **CI/CD Security Gates**: 3 mandatory workflows (signing, scanning, gates)
- ✅ **Audit Logging**: Immutable JSONL logs with InsightFlow integration

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Task-specific tests
pytest tests/task9/ -v      # Production readiness
pytest tests/task10/ -v     # Security features

# End-to-end integration
pytest tests/integration/test_end_to_end.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Coverage by Module

| Module | Coverage | Status |
|--------|----------|--------|
| security/ | 91% | ✅ |
| orchestrator.py | 88% | ✅ |
| adapters/ | 85% | ✅ |
| upscaler/ | 82% | ✅ |
| interpolator/ | 78% | ⚠️ |
| audio_manager/ | 65% | ⚠️ |

---

## 📊 Benchmarks Dashboard

Generate visual performance dashboard:

```bash
python tools/benchmarks_dashboard.py

# Opens: benchmarks_dashboard.html
# Shows:
# - Quality over time
# - Latency distribution
# - Cost trends
# - Success rate trends
```

---

## 📚 Documentation

### For Developers

- **[Developer Handbook](Documentation/DEVELOPER_HANDBOOK.md)** - Complete architecture, module reference, development workflow, troubleshooting
- **[Production Guide](README_PRODUCTION.md)** - API reference, deployment, monitoring

### Task Implementation Docs

Located in `Documentation/Tasks/`:

1. [Task 1](Documentation/Tasks/Task-1-README.md) - LoRA Fine-tuning
2. [Task 2](Documentation/Tasks/Task-2-README.md) - AnimateDiff Integration
3. [Task 3](Documentation/Tasks/Task-3-README.md) - RIFE Interpolation
4. [Task 4](Documentation/Tasks/Task-4-README.md) - SadTalker Lip-sync
5. [Task 5](Documentation/Tasks/Task-5-README.md) - ESRGAN Upscaling
6. [Task 6](Documentation/Tasks/Task-6-README.md) - RL Optimization
7. [Task 7](Documentation/Tasks/Task-7-README.md) - Yotta Fallback
8. [Task 8](Documentation/Tasks/Task-8-README.md) - API Development
9. [Task 9](Documentation/Tasks/Task-9-README.md) - Production Readiness
10. [Task 10](Documentation/Tasks/Task-10-README.md) - Security Hardening

### Reports

PDF reports for each task available in `Documentation/Reports/`

---

## 🔧 Development Workflow

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 .

# Auto-format code
black .

# Type checking
mypy .
```

### Contribution Guidelines

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes with tests
3. Run test suite: `pytest tests/ -v`
4. Ensure coverage ≥80%: `pytest --cov=.`
5. Submit pull request

### Code Review Checklist

- [ ] Tests passing
- [ ] Coverage ≥80% for new code
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Security reviewed (if applicable)
- [ ] Performance benchmarked

---

## 🐳 Docker Deployment

```bash
# Build production image
docker build -t loratv-production .

# Run with GPU support
docker run --gpus all -p 8001:8001 loratv-production

# Using docker-compose
docker-compose up -d
```

---

## 🎯 Roadmap

### Current Status (Task 10 Complete)

✅ **Foundation** (Tasks 1-3): LoRA, AnimateDiff, Interpolation  
✅ **Enhancement** (Tasks 4-5): Lip-sync, Upscaling  
✅ **Optimization** (Tasks 6-7): RL, Cloud Fallback  
✅ **Production** (Tasks 8-9): API, Testing, Monitoring  
✅ **Security** (Task 10): Watermarking, Signing, Audit Logging  

### Feedback & Improvements (In Progress)

Based on recent review (Score: 8.5/10):

**✅ Completed**:
- [x] Comprehensive Developer Handbook (Architecture, hand-off, onboarding)
- [x] End-to-end integration tests
- [x] Benchmarks dashboard with visual metrics
- [x] Organized documentation (Documentation/ folder)
- [x] Consolidated test files (tests/ folder)

**🔄 In Progress**:
- [ ] Expand test coverage to 90% (currently 81%)
- [ ] Add dashboard UI for telemetry visualization
- [ ] Document scene graph memory & character continuity features

**📅 Planned (Future Phases)**:
- [ ] Scene graph memory for multi-scene continuity
- [ ] Cross-shot character consistency
- [ ] Multi-scene narrative sequencing engine
- [ ] Advanced UI/operator tooling

### Next Phase Goals

1. **Scene Memory & Continuity**
   - Scene graph for tracking entities across clips
   - Character embedding consistency
   - Temporal narrative flow

2. **UI/Operator Tooling**
   - Real-time telemetry dashboard
   - Non-technical user interface
   - Visual quality comparison tools

3. **Advanced Analytics**
   - VMAF benchmarking over time
   - Automated regression detection
   - Cost optimization recommendations

---

## 📞 Support

### Common Issues

See [Developer Handbook - Troubleshooting](Documentation/DEVELOPER_HANDBOOK.md#troubleshooting) for:
- GPU out of memory errors
- Watermark detection issues
- Poor lip-sync quality
- Slow generation times

### Resources

- **Documentation**: `Documentation/DEVELOPER_HANDBOOK.md`
- **API Reference**: `README_PRODUCTION.md`
- **Task Details**: `Documentation/Tasks/Task-*-README.md`
- **GitHub Issues**: [Create issue](https://github.com/shashankpc7746/LoRA_TextToVision/issues)

### Team

| Role | Focus Area |
|------|------------|
| Lead Developer | Architecture, RL optimization |
| Security Engineer | Task 10 implementation, CI/CD |
| ML Engineer | Model fine-tuning, quality |
| DevOps | Cloud infrastructure, deployment |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🎓 About "Gurukul" Project

**Important**: "Gurukul" is the **project name**, not a thematic constraint.

- **General-purpose educational video platform** for ANY subject
- Supports diverse content: physics, programming, history, art, sports, cooking, etc.
- No visual style limitations - adapts to content requirements
- "Gurukul" is just a brand name (like "YouTube" or "Khan Academy")

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | 180+ |
| **Lines of Code** | ~25,000 |
| **Test Coverage** | 81% |
| **Documentation** | 12,000+ lines |
| **Tasks Completed** | 10/10 (100%) |
| **Security Features** | 9/9 mandatory |
| **Production Ready** | ✅ Yes |

---

**Last Updated**: November 12, 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅

---

*LoRA_TextToVision - Transforming text into cinematic video experiences* 🎬✨
