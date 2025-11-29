# 🎬 TTV Studio - End-to-End Demo Guide

**Document Version:** 1.0  
**Last Updated:** November 29, 2025  
**Purpose:** Complete walkthrough of TTV video generation pipeline  
**Audience:** New engineers, stakeholders, QA team  
**Duration:** 15-20 minutes for full demo

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Demo (5 minutes)](#quick-demo-5-minutes)
3. [Complete Demo (20 minutes)](#complete-demo-20-minutes)
4. [Expected Outputs](#expected-outputs)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Demos](#advanced-demos)

---

## Prerequisites

### System Requirements

- **OS:** Windows 10/11 or Linux
- **GPU:** NVIDIA RTX 3060 or better (8GB+ VRAM)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 50GB free space (for models and cache)
- **Python:** 3.10.x (verified working version)

### Environment Setup

**1. Clone Repository:**
```powershell
cd C:\Shashank
git clone https://github.com/shashankpc7746/LoRA_TextToVision.git
cd LoRA_TextToVision
```

**2. Create Virtual Environment:**
```powershell
python -m venv gurukul-lora-env
.\gurukul-lora-env\Scripts\Activate.ps1
```

**3. Install Dependencies:**
```powershell
pip install -r requirements-runtime.txt
```

**4. Set Environment Variables:**
```powershell
# Create .env file
cp .env.example .env

# Edit .env and add:
# GEMINI_API_KEY=your_gemini_api_key_here
# KSML_TOKEN=ksml_production
# RUNTIME_KEY=<base64-runtime-key>
# PEXELS_API_KEY=your_pexels_key (optional, for LoRA training)
```

**5. Verify GPU:**
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

**Expected Output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

---

## Quick Demo (5 minutes)

### Generate Your First Video

**Step 1: Navigate to AnimateDiff directory**
```powershell
cd AnimateDiff
```

**Step 2: Run the safe video generator**
```powershell
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1
```

**What This Does:**
- Reads lesson content from `lesson_comprehensive_1.json`
- Uses "realistic" render style
- Generates video #1 (seed=1 for reproducibility)
- Complete pipeline: Text → Motion → Audio → Subtitles → Watermark

**Step 3: Watch Progress**
```
[2025-11-29 14:30:15] 📋 Lesson: Introduction to Pythagorean Theorem
[2025-11-29 14:30:16] 🎨 Style: realistic
[2025-11-29 14:30:17] 🔧 Optimizing prompts with Gemini...
[2025-11-29 14:30:25] ✅ Prompts optimized (8.2s)
[2025-11-29 14:30:26] 🎬 Generating motion with AnimateDiff...
[2025-11-29 14:32:14] ✅ Motion generated (108.5s)
[2025-11-29 14:32:15] 🎤 Generating audio with multi-voice TTS...
[2025-11-29 14:32:28] ✅ Audio generated (13.1s)
[2025-11-29 14:32:29] 📝 Creating subtitles...
[2025-11-29 14:32:32] ✅ Subtitles created (2.8s)
[2025-11-29 14:32:33] 🔒 Adding watermark...
[2025-11-29 14:32:36] ✅ Watermark added (3.2s)
[2025-11-29 14:32:37] ✅ VIDEO COMPLETE!
[2025-11-29 14:32:37] 📹 Output: storage/lesson_comprehensive_1_realistic_v1.mp4
[2025-11-29 14:32:37] ⏱️  Total time: 135.7 seconds
```

**Step 4: View the Output**
```powershell
# Open video in default player
Start-Process storage\lesson_comprehensive_1_realistic_v1.mp4
```

**Expected Result:**
- 6-second video (720p, 25 FPS)
- Realistic visual style
- Multi-voice narration (male/female)
- Burned-in subtitles
- Watermark in metadata

---

## Complete Demo (20 minutes)

### Demo 1: Different Render Styles (5 min)

**Generate 3 videos with different styles:**

```powershell
# Realistic style (photorealistic, human subjects)
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1

# Anime style (illustrated, artistic)
python generate_lesson_video_safe.py lesson_comprehensive_1.json anime 2

# Artistic style (painterly, creative)
python generate_lesson_video_safe.py lesson_comprehensive_1.json artistic 3
```

**Compare Outputs:**
- `storage/lesson_comprehensive_1_realistic_v1.mp4` - Natural, documentary-style
- `storage/lesson_comprehensive_1_anime_v2.mp4` - Illustrated, vibrant colors
- `storage/lesson_comprehensive_1_artistic_v3.mp4` - Painterly, creative interpretation

**Key Observation:** Same content, different visual aesthetics based on target audience

---

### Demo 2: Custom Lesson Content (8 min)

**Step 1: Create custom lesson JSON**

```powershell
# Create new lesson file
notepad lessons\my_first_lesson.json
```

**Step 2: Add lesson content**

```json
{
  "lesson_id": "demo_001",
  "title": "Understanding Photosynthesis",
  "script": "Photosynthesis is the process by which plants convert sunlight into energy. Chlorophyll in the leaves absorbs light, which drives the chemical reactions that produce glucose and oxygen.",
  "narration": {
    "text": "Today we'll learn about photosynthesis, the amazing process that keeps our planet green and provides the oxygen we breathe.",
    "voice": "female",
    "pace": "medium"
  },
  "visuals": {
    "scene_type": "educational",
    "camera_movement": "slow_pan",
    "focus_elements": ["leaf", "sunlight", "chlorophyll", "chemical_diagram"]
  },
  "duration": 6
}
```

**Step 3: Generate video**

```powershell
python generate_lesson_video_safe.py lessons\my_first_lesson.json realistic 1
```

**Step 4: Validate output**

```powershell
# Check video exists
Test-Path storage\demo_001_realistic_v1.mp4

# View video properties
ffprobe storage\demo_001_realistic_v1.mp4
```

---

### Demo 3: API-Based Generation (4 min)

**Step 1: Start API server**

```powershell
cd AnimateDiff_API
python start_server.py
```

**Expected Output:**
```
[2025-11-29 14:35:00] INFO: Starting TTV API server...
[2025-11-29 14:35:01] INFO: Runtime key validated ✅
[2025-11-29 14:35:02] INFO: Models loaded successfully
[2025-11-29 14:35:03] INFO: Uvicorn running on http://0.0.0.0:8000
```

**Step 2: Send API request (new PowerShell window)**

```powershell
# Test API health
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Submit video generation job
$body = @{
    lesson_content = "Albert Einstein developed the theory of relativity, which revolutionized our understanding of space and time."
    style = "realistic"
    quality = "desktop_720p"
    priority = "normal"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/ttv/generate" -Method POST -Body $body -ContentType "application/json"

Write-Output "Job ID: $($response.job_id)"
```

**Step 3: Check job status**

```powershell
# Poll job status
$jobId = $response.job_id
while ($true) {
    $status = Invoke-RestMethod -Uri "http://localhost:8000/ttv/status/$jobId" -Method GET
    Write-Output "Status: $($status.status) | Progress: $($status.progress)%"
    
    if ($status.status -eq "completed") {
        Write-Output "Video URL: $($status.video_url)"
        break
    }
    
    Start-Sleep -Seconds 5
}
```

**Step 4: Download result**

```powershell
# Download generated video
Invoke-WebRequest -Uri $status.video_url -OutFile "api_generated_video.mp4"

# Play video
Start-Process api_generated_video.mp4
```

---

### Demo 4: Adaptive Engine Intelligence (3 min)

**Step 1: Stress test with concurrent requests**

```powershell
cd AnimateDiff_API
python test_adaptive_day1.py
```

**What This Tests:**
- Device capability detection (GPU probing)
- Tier routing (Local → Office → Cloud escalation)
- Queue management (handles 50 concurrent users)
- Intelligent caching (backgrounds, poses, seeds)
- Fallback to Yotta Cloud (if overloaded)

**Expected Output:**
```
🧪 Adaptive Engine Day 1 Tests
================================

Test 1: Device Capability Detection
✅ GPU detected: NVIDIA GeForce RTX 3060 Ti
✅ VRAM: 8192 MB
✅ Tier: OFFICE (capable of 720p production)

Test 2: Budget Planning
✅ Lesson cost estimated: $0.45
✅ Tier selected: LOCAL (within budget)
✅ Caching enabled: 60% cost reduction

Test 3: Concurrent Request Handling
✅ 50 concurrent users submitted
✅ Queue depth: 47 (3 processing)
✅ Success rate: 97.1%
✅ Average latency: 2.3 minutes

Test 4: Intelligent Caching
✅ Cache hit rate: 42%
✅ Speedup: 58% faster with cache
✅ Storage saved: 1.2 GB

Test 5: Yotta Fallback
⚠️  Yotta GPU access pending (95% functional)
✅ Fallback logic working (placeholder API)

Overall: 4/5 tests PASSED ✅
```

---

## Expected Outputs

### File Structure After Demo

```
LoRA_TextToVision/
├── AnimateDiff/
│   ├── storage/
│   │   ├── lesson_comprehensive_1_realistic_v1.mp4  ✅ Generated
│   │   ├── lesson_comprehensive_1_anime_v2.mp4      ✅ Generated
│   │   ├── lesson_comprehensive_1_artistic_v3.mp4   ✅ Generated
│   │   └── demo_001_realistic_v1.mp4                ✅ Generated
│   ├── cache/
│   │   ├── backgrounds/   (cached image assets)
│   │   ├── poses/         (cached character poses)
│   │   └── seeds/         (cached random seeds)
│   └── lessons/
│       └── my_first_lesson.json  ✅ Created
├── logs/
│   ├── production.log     (generation logs)
│   └── audit/             (encrypted audit logs)
└── api_generated_video.mp4  ✅ Downloaded from API
```

### Video Quality Metrics

**Expected Specifications:**
- **Resolution:** 1280x720 (720p)
- **Frame Rate:** 25 FPS
- **Duration:** 6 seconds (configurable)
- **Codec:** H.264 (widely compatible)
- **Audio:** AAC 44.1kHz stereo
- **Subtitles:** Burned-in (not removable)
- **Watermark:** In FFmpeg metadata (invisible)

**Quality Checklist:**
- [ ] Video plays smoothly (no stuttering)
- [ ] Audio synchronized with visuals
- [ ] Subtitles readable and timed correctly
- [ ] No visible artifacts or corruption
- [ ] Watermark verifiable with `python -m security.watermark verify`

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0; 8.00 GiB total capacity)
```

**Solution:**
```powershell
# Reduce quality tier
python generate_lesson_video_safe.py lesson.json realistic 1 --quality mobile_480p

# Or clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

---

#### Issue 2: Gemini API Key Error

**Symptom:**
```
Error: GEMINI_API_KEY not found in environment
```

**Solution:**
```powershell
# Add to .env file
echo "GEMINI_API_KEY=your_actual_api_key_here" >> .env

# Or set temporarily
$env:GEMINI_API_KEY = "your_actual_api_key_here"
python generate_lesson_video_safe.py ...
```

---

#### Issue 3: No Audio in Output

**Symptom:**
Video plays but no sound

**Solution:**
```powershell
# Check TTS installation
python -c "from bark import SAMPLE_RATE; print('Bark TTS installed')"

# Reinstall if needed
pip install git+https://github.com/suno-ai/bark.git

# Verify audio file exists
Test-Path AnimateDiff\temp\audio_*.wav
```

---

#### Issue 4: Watermark Verification Fails

**Symptom:**
```
Warning: Watermark not found in video metadata
```

**Solution:**
```powershell
# Check FFmpeg version (must be 4.4+)
ffmpeg -version

# Re-add watermark manually
python -m security.watermark add storage\video.mp4 --job-id test-001
```

---

#### Issue 5: API Server Won't Start

**Symptom:**
```
Error: Address already in use (port 8000)
```

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F

# Or use different port
python start_server.py --port 8001
```

---

## Advanced Demos

### Demo 5: Multi-Language Support

```powershell
# Generate video with Hindi narration
python generate_lesson_video_safe.py lesson_hindi.json realistic 1 --language hi

# Generate with Spanish subtitles
python generate_lesson_video_safe.py lesson_spanish.json realistic 1 --subtitles es
```

---

### Demo 6: Long-Form Video (60 seconds)

```powershell
# Generate longer video (auto-segments into clips)
python AnimateDiff/multi_clip_generator.py lesson_long.json realistic 1 --duration 60
```

**Note:** Generates 10x 6-second clips and merges them

---

### Demo 7: Batch Processing

```powershell
# Generate 10 videos with different seeds
for ($i=1; $i -le 10; $i++) {
    python generate_lesson_video_safe.py lesson.json realistic $i
}

# All videos saved to storage/ directory
```

---

### Demo 8: Quality Comparison

```powershell
# Generate same lesson at different qualities
python generate_lesson_video_safe.py lesson.json realistic 1 --quality mobile_480p
python generate_lesson_video_safe.py lesson.json realistic 1 --quality desktop_720p
python generate_lesson_video_safe.py lesson.json realistic 1 --quality premium_1080p

# Compare file sizes and visual quality
```

---

### Demo 9: Fallback System Test

```powershell
# Simulate GPU failure (use CPU mode)
$env:CUDA_VISIBLE_DEVICES = ""
python generate_lesson_video_safe.py lesson.json realistic 1

# System should fall back to Yotta Cloud automatically
```

---

### Demo 10: Security Validation

```powershell
# Verify watermark
python -m security.watermark verify storage\video.mp4

# Check artifact signature
python -m security.artifact_signer verify models\sd-v1-5.ckpt

# Test restricted demo mode
Remove-Item Env:\RUNTIME_KEY
python generate_lesson_video_safe.py lesson.json realistic 1
# Should show "DEMO" watermark and limit quality to 480p
```

---

## Demo Recording Checklist

If recording this demo for documentation:

- [ ] Start with clean environment (no cached models)
- [ ] Show GPU detection output
- [ ] Capture entire generation process (2 min)
- [ ] Show final video playback
- [ ] Demonstrate watermark verification
- [ ] Show API request/response flow
- [ ] Display adaptive engine metrics
- [ ] Verify all quality checks pass
- [ ] Demonstrate error recovery (simulate failure)
- [ ] Show logs and monitoring

---

## Success Criteria

**Demo is successful if:**

1. ✅ Video generates without errors
2. ✅ Output quality is 720p, 25 FPS
3. ✅ Audio synchronized with visuals
4. ✅ Subtitles appear correctly
5. ✅ Watermark verifiable in metadata
6. ✅ Generation completes in <3 minutes
7. ✅ API responds within 5 seconds
8. ✅ Adaptive engine handles 50 concurrent users
9. ✅ Fallback system activates on failure
10. ✅ Security validation passes

---

## Next Steps After Demo

**For New Engineers:**
1. Read `TTV_HANDOVER_MASTER.md` (complete technical documentation)
2. Review `FAQ_NEW_ENGINEER.md` (common questions answered)
3. Study `Architecture Diagrams/` (visual system overview)
4. Run full test suite: `pytest tests/ -v`
5. Schedule knowledge transfer session (if available)

**For Stakeholders:**
1. Review generated video samples
2. Assess quality vs requirements
3. Plan production deployment timeline
4. Discuss feature roadmap
5. Approve handover completion

---

**Demo Guide Version:** 1.0  
**Last Updated:** November 29, 2025  
**Status:** ✅ Production-Ready  
**Estimated Demo Duration:** 15-20 minutes (Quick: 5 min, Complete: 20 min)

---

## 🎥 Pre-Recorded Demo (If Available)

**Location:** `Documentation/Videos/TTV_End_to_End_Demo.mp4` (to be recorded)

**Contents:**
- 0:00 - Introduction and system overview
- 1:00 - Environment setup
- 3:00 - Quick demo (single video generation)
- 6:00 - Different render styles comparison
- 9:00 - API-based generation
- 12:00 - Adaptive engine stress test
- 15:00 - Security features demonstration
- 18:00 - Troubleshooting common issues
- 20:00 - Summary and next steps

**To Record Demo:**
```powershell
# Use OBS Studio or similar screen recorder
# Record at 1080p, 30 FPS
# Enable microphone for narration
# Show terminal output clearly
# Pause during long operations (edit out in post)
```

---

**End of Demo Guide**
