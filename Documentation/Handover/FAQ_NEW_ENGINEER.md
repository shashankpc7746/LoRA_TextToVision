# ❓ TTV Studio - FAQ for New Engineers

**Version:** 1.0.0  
**Last Updated:** November 26, 2025  
**Audience:** New engineers joining the TTV Studio project  
**Purpose:** Quick answers to common questions about architecture, setup, and development  

---

## 🎯 Getting Started

### Q1: What is TTV Studio in one sentence?

**A:** TTV Studio is an AI-powered educational video generation platform that transforms text lessons into high-quality, cinematic videos with intelligent story understanding, adaptive optimization, and enterprise security.

---

### Q2: What does "Gurukul" mean? Is this only for Indian education content?

**A:** **"Gurukul" is just the project brand name** - like "YouTube" or "Netflix." 

**The system handles ANY educational content:**
- ✅ Physics, Chemistry, Mathematics
- ✅ Programming, Data Science, AI/ML
- ✅ History, Geography, Literature
- ✅ Cooking, Art, Music, Sports
- ✅ Business, Finance, Marketing
- ✅ ANY concept a user wants to learn

**NOT limited to:** Traditional Indian themes, specific cultural aesthetics, or religious content.

---

### Q3: How do I generate my first video?

**A:** Three simple steps:

```powershell
# 1. Activate Python environment
.\gurukul-lora-env\Scripts\Activate.ps1

# 2. Navigate to AnimateDiff
cd AnimateDiff

# 3. Generate video
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1
```

**Output:** Find your video in `AnimateDiff/storage/YYYY-MM-DD/` folder.

---

### Q4: Which Python version should I use?

**A:** Python 3.10.11 (already set up in `gurukul-lora-env/`).

**Do NOT use:** Python 3.11+ or 3.9- (dependency compatibility issues).

---

### Q5: What GPU do I need?

**A:** Minimum RTX 3060 Ti (12GB VRAM) for local development.

**Supported GPUs:**
- RTX 3060 Ti (12GB) - Good for development
- RTX 3090 (24GB) - Better for production
- RTX 4090 (24GB) - Best performance

**If no GPU:** System automatically falls back to Yotta Cloud.

---

## 🏗️ Architecture Questions

### Q6: I'm confused - are Tasks 1-5 sequential pipeline stages?

**A:** **NO!** This is the most common misunderstanding.

**Reality:**
- **Task 1:** Learning exercise (NOT production)
- **Task 2:** Component development (integrated into Task 3)
- **Task 3:** **THE COMPLETE TTV ENGINE** (the main production system)
- **Task 4:** Adaptive intelligence layer (scaling, caching, RL)
- **Task 5:** Production API layer (/ttv/generate endpoint)

**Think of it like this:**
- Task 3 built the complete car 🚗
- Tasks 4-11 added GPS, turbo, leather seats, security system

---

### Q7: Where is the main entry point to generate videos?

**A:** `AnimateDiff/generate_lesson_video_safe.py` (Task 3 core engine)

**What it does:**
1. Gemini API text optimization
2. AnimateDiff motion generation
3. Multi-voice TTS + audio integration
4. Subtitle synchronization
5. Security (watermarking + fingerprinting)
6. Intelligence (story analysis, smart extension)

**Full pipeline in ONE script.**

---

### Q8: What is `adaptive_engine/` and when was it created?

**A:** **CRITICAL:** `adaptive_engine/` was **created in Task 4**, **extended in Task 11**.

**Task 4 created (4-day sprint):**
- Device probe, tier routing
- Intelligent caching (40-60% speedup)
- NAS storage, GPU queue
- RL policy, Yotta fallback

**Task 11 extended (7-day sprint):**
- Story context parser (gender resolution)
- Scene memory core (NetworkX graph)
- Narrative sequencer (story beats, arcs)
- Emotion controller (motion-emotion coupling)
- Smart video extension (SlowMo + Freeze)
- TTV intelligence metrics (26 metrics)

**Location:** `AnimateDiff/adaptive_engine/`

---

### Q9: What's the difference between Task 3 and Task 7?

**A:** 

**Task 3 (5-day Gurukul Sprint):**
- Built the **complete TTV engine**
- Gemini API, AnimateDiff, TTS, subtitles
- 6 complete videos generated (9.0/10 score)
- **This is the main production system**

**Task 7 (4-day Quality Leap Sprint):**
- Enhanced Task 3 with **better quality**
- Added RIFE interpolation (12fps → 24fps)
- Added Real-ESRGAN upscaling (1080p)
- Added RL policy optimization
- 50+ concurrent users, 97% success rate

**Analogy:** Task 3 built a good car, Task 7 upgraded it to luxury.

---

### Q10: How does the security system work?

**A:** Every video gets **5 layers of security** (Task 10):

1. **Invisible Watermark:** FFmpeg metadata (32-bit watermark ID)
2. **Visible Watermark:** BHI logo (51x50px, 35% opacity, bottom-right)
3. **Content Fingerprint:** SHA256 + BLAKE2b hashes
4. **Artifact Signature:** Ed25519 cryptographic signing
5. **Build Fingerprint:** BUILD_ID seeding

**All automatic** - happens during video generation.

**Why 5 bugs were fixed:** Watermarking is hard! Took 4 hours of debugging to get FFmpeg metadata working correctly.

---

## 🔧 Development Questions

### Q11: How do I add background music to videos?

**A:** Already integrated in Task 6!

```python
# In your lesson JSON
{
    "bgm_enabled": true,
    "bgm_volume": 0.3  # 30% volume (subtle background)
}
```

**BGM files location:** `assets/background_music/` or `assets/bgm/`

**Audio mixing:** Done automatically via FFmpeg.

---

### Q12: Why do some videos have repetitive looping?

**A:** **This was fixed in Task 11 Day 5!**

**Before (Task 3):**
- 2s video clip looped 3x to match 6s audio = repetitive

**After (Task 11):**
- Smart Video Extension: SlowMo (0.8x) + Freeze (last frame)
- NO repetitive looping
- Result: Cinematic, professional

**Module:** `AnimateDiff/adaptive_engine/smart_video_extender.py`

---

### Q13: How do I fix gender confusion in characters?

**A:** **Already fixed in Task 11 Day 1!**

**Before (Task 3):**
- "seeker" → assumed male
- Later "She" → switched to female
- Result: Inconsistent character

**After (Task 11):**
- Full story NLP analysis
- Resolves gender from ALL sentences
- Result: Consistent character throughout

**Module:** `AnimateDiff/adaptive_engine/story_context_parser.py`

---

### Q14: What if I see RIFE black screens?

**A:** **RIFE is NOT used for video extension** (Task 11 design decision).

**RIFE is used for:**
- ✅ Frame interpolation (12fps → 24fps) in Task 7
- ✅ Smoothing existing motion

**RIFE is NOT used for:**
- ❌ Extending video duration (causes black screens)
- ❌ Matching audio length

**For extension, we use:** Frame duplication + freeze (reliable, artifact-free).

---

### Q15: How do I run tests?

**A:** 

```powershell
# Run all tests
pytest tests/

# Run specific task tests
pytest tests/test_task9_integration.py
pytest tests/test_task10_integration.py
pytest tests/test_day6_ttv_metrics.py

# Run with coverage
pytest tests/ --cov=AnimateDiff --cov-report=html
```

**Current status:** 152/152 tests passing (100%), 95%+ coverage.

---

## 📁 File Structure Questions

### Q16: Where are the generated videos stored?

**A:** Multiple locations depending on configuration:

**Local storage (default):**
```
AnimateDiff/storage/YYYY-MM-DD/
├── Lesson_Title_realistic_complete.mp4
├── Lesson_Title_realistic_complete.srt
├── Lesson_Title_realistic_complete_fingerprint.json
└── [intermediate files]
```

**NAS storage (Task 4):**
```
\\192.168.0.94\ttv_videos\YYYY-MM-DD\
```

**Cloud storage (Task 8):**
- S3 bucket
- Supabase storage
- BHIV bucket

---

### Q17: Where are the LoRA models stored?

**A:** 

**Task 1 learning models (NOT used in production):**
```
LoRA_Text/outputs/
LoRA_StableDiffusion/outputs/
```

**Task 7 production LoRA models:**
```
adapters/lora_adapter.py  # Base LoRA wrapper
```

**Task 9 indigenous Gurukul LoRA:**
```
adapters/gurukul_lora/
├── checkpoint.pt  # 89MB trained model
├── dataset_curator.py
└── train_adapter.py
```

---

### Q18: Where are the audit logs?

**A:** 

```
logs/audit/audit_YYYYMMDD.jsonl
```

**Format:** JSONL (one JSON object per line)

**Contains:**
- 26 intelligence metrics (Task 11)
- Security metadata (Task 10)
- Generation parameters
- Performance metrics
- Error tracking

**KSML-compliant:** Append-only, tamper-evident.

---

### Q19: How do I check what's in a lesson JSON file?

**A:** 

```powershell
# View lesson file
cat AnimateDiff/lessons/lesson_comprehensive_1.json

# Pretty print
cat AnimateDiff/lessons/lesson_comprehensive_1.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**Lesson JSON structure:**
```json
{
  "title": "Lesson Title",
  "scenes": [
    {
      "text": "Scene description",
      "duration": 6,
      "style": "realistic"
    }
  ],
  "bgm_enabled": true,
  "watermark_enabled": true
}
```

---

## 🚀 Deployment Questions

### Q20: How do I deploy TTV Studio to production?

**A:** Use Docker (Task 6):

```powershell
# Build Docker image
docker-compose build

# Run production server
docker-compose up -d

# Check logs
docker-compose logs -f ttv-studio
```

**What happens:**
- Multi-stage Docker build
- Gunicorn + Uvicorn workers
- Production dependencies only
- Health checks enabled
- Auto-restart on failure

---

### Q21: How many concurrent users can the system handle?

**A:** **50+ concurrent users** (stress tested in Tasks 4, 6, 7)

**Performance metrics:**
- 97.1% success rate
- 86.2% cost efficiency
- <3 min average generation time
- Graceful degradation on overload
- Automatic Yotta cloud fallback

---

### Q22: What happens if local GPU fails?

**A:** **Automatic Yotta Cloud fallback** (Task 4 + Task 7)

**Fallback logic:**
1. Try local GPU first (RTX 3060 Ti)
2. If fails → Try office GPU pool (4 GPUs)
3. If still fails → Escalate to Yotta Cloud
4. Return result to user seamlessly

**User never knows** - transparent failover.

---

## 🐛 Debugging Questions

### Q23: How do I debug video generation failures?

**A:** Check these in order:

**1. Check audit log:**
```powershell
Get-Content logs/audit/audit_20251126.jsonl | Select-Object -Last 5
```

**2. Check error logs:**
```powershell
Get-Content logs/error.log
```

**3. Check Sentry (if enabled):**
- Automatic error tracking
- Stack traces captured
- Context included

**4. Check video fingerprint:**
```powershell
cat AnimateDiff/storage/2025-11-26/Lesson_Title_realistic_complete_fingerprint.json
```

---

### Q24: Why is my video generation slow?

**A:** Check these optimization opportunities:

**1. Enable caching (Task 4):**
```python
# In config
cache_enabled = True  # 40-60% speedup
```

**2. Use NAS storage (Task 4):**
```python
# Pre-cache backgrounds, poses, seeds on NAS
nas_enabled = True
```

**3. Enable mixed precision (Task 7):**
```python
# Use FP16 instead of FP32
mixed_precision = True
```

**4. Check GPU utilization:**
```powershell
nvidia-smi
```

**5. Use quality presets (Task 5):**
```python
# mobile_480p is faster than desktop_720p
quality_preset = "mobile_480p"
```

---

### Q25: How do I verify watermarks are working?

**A:** 

**Check invisible watermark:**
```powershell
# Extract FFmpeg metadata
ffprobe -v quiet -print_format json -show_format Lesson_Title_realistic_complete.mp4
```

Look for: `watermark_id`, `build_id`, `generation_timestamp`

**Check visible watermark:**
- Open video in any player
- Look for BHI logo (51x50px, 35% opacity, bottom-right corner)

**Verify fingerprint:**
```powershell
# Check fingerprint file
cat Lesson_Title_realistic_complete_fingerprint.json
```

**Run verification script:**
```powershell
python security/verify_watermark.py Lesson_Title_realistic_complete.mp4
```

---

## 📊 Intelligence & Analytics Questions

### Q26: What are the 26 intelligence metrics tracked?

**A:** Organized by category (Task 11 Day 6):

**Story Analysis (6 metrics):**
- Character count
- Gender resolution success
- Text condensation percentage
- Prompt enhancement count
- Story complexity score
- Narrative coherence

**Scene Graph (5 metrics):**
- Total scenes
- Unique entities
- Scene transitions
- Entity co-occurrence
- Temporal relationships

**Narrative (5 metrics):**
- Story beats identified
- Character arcs tracked
- Dialogue count
- Tension curve points
- Pacing variations

**Emotion (4 metrics):**
- Emotion changes
- Motion intensity levels
- Micro-expression count
- Emotional continuity score

**Extension (3 metrics):**
- Clips extended
- Extension method used (SlowMo/Freeze)
- Audio-video sync accuracy

**Quality (3 metrics):**
- Final video duration
- FPS achieved
- Bitrate quality

---

### Q27: How do I view the scene graph for a video?

**A:** 

```python
# In Python (after generation)
from AnimateDiff.adaptive_engine.scene_memory_core import SceneMemoryCore

scene_memory = SceneMemoryCore()
scene_memory.load_from_audit_log("logs/audit/audit_20251126.jsonl")

# Query scene graph
print(scene_memory.get_entity_history("protagonist"))
print(scene_memory.get_scene_transitions())
print(scene_memory.visualize_graph())  # Generates NetworkX graph
```

**Output:** NetworkX graph showing scenes, entities, temporal relationships.

---

### Q28: Can I disable intelligence features for faster generation?

**A:** Yes, but **NOT recommended** for production.

```python
# In config (for testing only)
story_analysis_enabled = False  # Skip gender resolution
scene_graph_enabled = False     # Skip scene tracking
narrative_enabled = False       # Skip story beats
emotion_enabled = False         # Skip emotion coupling
smart_extension_enabled = False # Use old looping method

# WARNING: Disabling these brings back production problems:
# - Gender confusion
# - Video looping
# - Inconsistent characters
```

**For production:** Keep all intelligence features enabled.

---

### Q29: Why are lineage tokens (KSML tokens) required?

**A:** Lineage tokens provide **complete traceability** of every video from input to output.

**Without lineage tokens:**
- ❌ Can't prove who generated a video
- ❌ Can't track which models were used
- ❌ Can't detect tampered/unauthorized videos
- ❌ Can't meet regulatory compliance requirements

**With lineage tokens (KSML):**
- ✅ Every video cryptographically bound to KSML token
- ✅ Full audit trail in logs (append-only, tamper-evident)
- ✅ Watermarks linked to build ID and worker ID
- ✅ Can verify provenance years later

**Example lineage chain:**
```
Input → KSML Token → Models Used → Processing Steps → Watermark → Fingerprint → Audit Log
```

**How it works:**
```python
# At video generation start
ksml_token = {
    "ksml_token": "ksml_production",
    "intent": "video_generation",
    "karma_state": "authorized",
    "lineage": {
        "lesson": "Photosynthesis Basics",
        "style": "realistic",
        "build_id": "build_20251127_001"
    }
}

# Token embedded in:
# - Watermark metadata
# - Content fingerprint
# - Audit log entry
# - Security metadata
```

**Production requirement:** All videos MUST have valid KSML tokens in production mode.

---

### Q30: How do I add new scenes or transitions to a lesson?

**A:** Edit the lesson JSON file or use the scene editor.

**Method 1: Edit Lesson JSON**

```json
{
  "title": "My Lesson",
  "scenes": [
    {
      "scene_id": "scene_001",
      "text": "First scene content",
      "duration_sec": 5,
      "style": "realistic",
      "transition": "fade"
    },
    {
      "scene_id": "scene_002",
      "text": "Second scene content",
      "duration_sec": 5,
      "style": "realistic",
      "transition": "dissolve"
    }
  ]
}
```

**Available transitions (Task 11 Day 5):**
- `fade` - Smooth fade in/out
- `dissolve` - Cross-dissolve between scenes
- `wipe` - Wipe left/right
- `slide` - Slide left/right/up/down
- `zoom` - Zoom in/out
- `blur` - Blur transition
- `pixelate` - Pixelate effect
- `circle` - Circle wipe

**Method 2: Use Scene Editor (if available)**

```python
from AnimateDiff.scene_editor import SceneEditor

editor = SceneEditor("lesson_comprehensive_1.json")

# Add new scene
editor.add_scene(
    text="New scene content",
    duration=5,
    style="realistic",
    transition="fade"
)

# Modify existing scene
editor.update_scene(
    scene_id="scene_002",
    transition="dissolve"
)

# Save changes
editor.save()
```

**Testing new scenes:**
```powershell
# Regenerate video with new scenes
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1
```

---

### Q31: What common mistakes should I avoid?

**A:** Here are the top 10 mistakes that waste time:

**1. ❌ Treating Tasks 1-5 as pipeline stages**
- ✅ Task 3 is the complete engine, Tasks 4-11 are enhancements

**2. ❌ Using Python 3.11+ or 3.9-**
- ✅ Use Python 3.10.11 only (dependency compatibility)

**3. ❌ Editing binary files (checkpoints, cache, models)**
- ✅ Never touch `.pt`, `.pkl`, `.safetensors` files directly

**4. ❌ Committing test images/videos to Git**
- ✅ Test outputs belong in `.gitignore`

**5. ❌ Disabling security features "to make it faster"**
- ✅ Security is production-required, not optional

**6. ❌ Assuming "Gurukul" = Indian traditional themes only**
- ✅ Gurukul is just a project name, system handles ANY educational content

**7. ❌ Using RIFE for video extension**
- ✅ Use SlowMo + Freeze (Task 11 Day 5) - avoids black screens

**8. ❌ Hardcoding file paths**
- ✅ Use Path objects and environment variables

**9. ❌ Storing secrets in code or Git**
- ✅ Use `.env` files and secret managers

**10. ❌ Skipping tests before committing**
- ✅ Always run `pytest tests/` before pushing

**Pro tip:** Read the "Important Concepts" section in TTV_HANDOVER_MASTER.md to avoid architectural misunderstandings.

---

### Q32: What files should never be touched or modified?

**A:** **Critical files** - touching these can break production:

**1. Binary Model Files (NEVER EDIT):**
- ❌ `adapters/gurukul_lora/checkpoint.pt` (89MB LoRA model)
- ❌ `AnimateDiff/models/*.safetensors` (AnimateDiff weights)
- ❌ `AnimateDiff/models/*.ckpt` (Stable Diffusion checkpoints)
- ❌ Any `.pt`, `.pth`, `.safetensors` files

**Why:** Binary corruption = hours of retraining/redownloading

**2. Cache Files (NEVER COMMIT):**
- ❌ `AnimateDiff/cache/*.pkl` (scene memory, narrative cache)
- ❌ `__pycache__/` directories
- ❌ `.pyc` files

**Why:** Machine-specific, breaks cross-platform compatibility

**3. Security Keys (NEVER COMMIT):**
- ❌ `security/keys/signing_key.priv` (Ed25519 private key)
- ❌ `.env` file (contains secrets)
- ❌ Any file with passwords/API keys

**Why:** Security breach, unauthorized access

**4. Generated Artifacts (NEVER COMMIT):**
- ❌ `AnimateDiff/storage/**/*.mp4` (generated videos)
- ❌ `test_results/**/*.png` (test images)
- ❌ `adapters/gurukul_lora/test_outputs/` (test images)
- ❌ `logs/audit/*.jsonl` (except production samples)

**Why:** Bloats repository, wastes storage

**5. System Configuration (MODIFY WITH CAUTION):**
- ⚠️ `requirements-runtime.txt` (production dependencies)
- ⚠️ `Dockerfile` (production deployment)
- ⚠️ `docker-compose.yml` (orchestration)

**Why:** Breaking changes affect all deployments

**6. Core Engine Logic (UNDERSTAND BEFORE CHANGING):**
- ⚠️ `AnimateDiff/unified_video_generator.py` (main orchestrator)
- ⚠️ `security/watermark.py` (watermarking pipeline)
- ⚠️ `audit_logger.py` (audit logging system)

**Why:** Complex interdependencies, extensive testing needed

**Safe to modify:**
- ✅ Lesson JSON files (`AnimateDiff/lessons/*.json`)
- ✅ Background music (`assets/bgm/`)
- ✅ Test scripts (`tests/*.py`)
- ✅ Documentation (`Documentation/**/*.md`)
- ✅ Your own feature modules

**Before modifying ANY file:**
1. Read the module docstring
2. Check if it's tested (`tests/test_*.py`)
3. Create a backup
4. Test changes locally
5. Run full test suite

---

## 🔐 Security Questions

### Q33: Are the security keys committed to Git?

**A:** **NO!** Security keys are in `.gitignore`.

**What's in Git:**
- ✅ Public keys (`security/keys/signing_key.pub`)
- ✅ Watermark logo (`security/watermark_logo/BHI_logo.png`)
- ✅ Security scripts (encryption, signing, verification)

**What's NOT in Git:**
- ❌ Private keys (`security/keys/signing_key.priv`) - LOCAL ONLY
- ❌ KSML encryption keys - Environment variables
- ❌ Supabase JWT secrets - Environment variables

**Setup new keys:**
```powershell
python security/generate_keys.py
```

---

### Q30: How long are runtime keys valid?

**A:** **12-24 hours** (Task 10 design).

**Why time-limited:**
- Prevents indefinite key compromise
- Forces periodic key rotation
- Restricted demo mode after expiration

**Check key expiration:**
```python
from security.runtime_validator import RuntimeValidator

validator = RuntimeValidator()
expiration = validator.get_key_expiration()
print(f"Key expires in: {expiration}")
```

---

## 🎓 Learning Resources

### Q31: Where can I learn more about specific components?

**A:** Read the Task README files in `Documentation/Tasks/`:

- **Task 1:** LoRA basics (learning exercise)
- **Task 2:** Motion animation development
- **Task 3:** Core TTV engine (START HERE!)
- **Task 4:** Adaptive intelligence system
- **Task 5:** Production API development
- **Task 6:** Production hardening
- **Task 7:** Quality leap sprint
- **Task 8:** Microservice architecture
- **Task 9:** Indigenous adapters
- **Task 10:** Security implementation
- **Task 11:** Intelligence stack

**Each README contains:**
- Detailed implementation notes
- Code examples
- Testing results
- Lessons learned
- Production issues and fixes

---

### Q32: What if I'm stuck and need help?

**A:** Follow this escalation path:

**1. Check this FAQ first** (most common questions)

**2. Read relevant Task README:**
- `Documentation/Tasks/Task-{N}-README.md`

**3. Check handover documents:**
- `Documentation/Handover/TTV_HANDOVER_MASTER.md`
- `Documentation/Handover/ARCHITECTURE_DIAGRAMS.md`

**4. Check audit logs:**
```powershell
Get-Content logs/audit/audit_YYYYMMDD.jsonl | Select-Object -Last 10
```

**5. Run tests to verify:**
```powershell
pytest tests/ -v
```

**6. Check code comments:**
- All critical modules have detailed docstrings
- Complex logic has inline comments

**If still stuck:** Contact previous engineer (but this should be rare if documentation is followed).

---

## 🎯 Quick Reference

### Essential Commands

```powershell
# Activate environment
.\gurukul-lora-env\Scripts\Activate.ps1

# Generate video
cd AnimateDiff
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1

# Run tests
pytest tests/

# Check GPU
nvidia-smi

# View logs
Get-Content logs/audit/audit_20251126.jsonl | Select-Object -Last 1 | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Docker deployment
docker-compose up -d

# Check watermarks
python security/verify_watermark.py video.mp4
```

### Key Files to Know

| File | Purpose |
|------|---------|
| `AnimateDiff/generate_lesson_video_safe.py` | Main entry point |
| `AnimateDiff/unified_video_generator.py` | Core TTV engine |
| `AnimateDiff/adaptive_engine/` | Intelligence modules |
| `security/` | Watermarking, signing, encryption |
| `logs/audit/` | Audit logs with 26 metrics |
| `requirements-runtime.txt` | Production dependencies |

### Important Locations

| Location | Contains |
|----------|----------|
| `AnimateDiff/storage/` | Generated videos |
| `adapters/gurukul_lora/` | Indigenous LoRA models |
| `assets/bgm/` | Background music files |
| `security/keys/` | Cryptographic keys (NOT in Git) |
| `logs/` | Audit and error logs |

---

**Document Status:** ✅ Production Ready  
**Last Updated:** November 27, 2025  
**Total Questions:** 36 (including all required FAQ topics)  
**Questions or Updates?** Add new Q&A sections as needed.
