# 🎬 TTV Studio - Complete Handover Master Document

**Project:** LoRA_TextToVision (TTV Studio)  
**Version:** 2.1.0  
**Last Updated:** November 27, 2025  
**Prepared By:** Shashank Gupta  
**For:** Next Engineer / Team Continuity  
**Status:** Production Ready ✅  
**Restructured:** November 27, 2025 (Clean A-F Format)

---

## A. System Overview

### What is TTV Studio?

**TTV Studio** (Text-to-Vision) is an enterprise-grade AI video generation platform that transforms text prompts and educational scripts into high-quality, cinematic videos with intelligent story understanding, adaptive optimization, and enterprise security.

**Project Name:** "Gurukul" - **IMPORTANT:** This is just the project brand name. The system handles **ANY educational content** (physics, programming, history, cooking, etc.), NOT limited to traditional Indian Gurukul themes.

### High-Level Flow: Prompt → Video

```
User Input (Text Prompt or Lesson JSON)
         ↓
Story Analysis + Character Resolution (Task 11)
         ↓
Text Optimization via Gemini API (Task 3)
         ↓
Motion Generation via AnimateDiff (Task 3)
         ↓
Audio Synthesis + Subtitle Sync (Task 3)
         ↓
Smart Video Extension (SlowMo + Freeze) (Task 11)
         ↓
Quality Enhancement (RIFE + Upscaling) (Task 7)
         ↓
Security Layer (Watermarking + Signing) (Task 10)
         ↓
Audit Logging with Intelligence Metrics (Task 11)
         ↓
Final Output: 1080p MP4 + Subtitles + Fingerprint
```

### Core Capabilities

**Input:** Text prompt or lesson JSON  
**Output:** Professional 1080p video with audio, subtitles, dual watermarking, fingerprinting, and KSML-compliant audit trail

**What Makes TTV Studio Unique:**
- ✅ **Complete TTV Engine (Task 3):** Full text-to-video pipeline with Gemini API, AnimateDiff, TTS, subtitles, multiple styles
- ✅ **Adaptive Intelligence (Task 4):** 50 concurrent users, device detection, NAS caching (40-60% speedup), GPU queue, Yotta fallback
- ✅ **Story Intelligence (Task 11):** Gender resolution, scene graphs, narrative sequencing, emotion-motion coupling, smart video extension
- ✅ **Production Security (Task 10):** Dual watermarking, Ed25519 signing, content fingerprinting, runtime key validation
- ✅ **Microservice Architecture (Task 8):** FastAPI + Celery GPU workers + multi-backend storage
- ✅ **95%+ Test Coverage:** 152 tests passing, production-ready

### Major Modules and Their Roles

| Module | Primary Role | Key Capabilities | Status |
|--------|-------------|------------------|--------|
| **Task 3 Core Engine** | Complete TTV pipeline | Gemini API, AnimateDiff, TTS, subtitles, fallback | ✅ Production |
| **Task 4 Adaptive Engine** | Scaling + optimization | Device probe, caching, GPU queue, RL, Yotta fallback | ✅ Production |
| **Task 5 Production API** | API endpoints | `/ttv/generate`, quality presets, progressive preview | ✅ Production |
| **Task 6 Production Hardening** | Deployment readiness | Docker, BGM, stress testing, clean dependencies | ✅ Production |
| **Task 7 Quality Leap** | Enterprise-grade quality | RIFE interpolation, Real-ESRGAN upscaling, RL optimization | ✅ Production |
| **Task 8 Microservice** | Service wrapper | FastAPI + Celery, multi-backend storage, monitoring | ✅ Production |
| **Task 9 Indigenous Adapters** | Custom fine-tuning | Gurukul LoRA training, temporal consistency, upscaling | 95% Complete |
| **Task 10 Security** | Provenance + protection | Dual watermarking, Ed25519 signing, KSML encryption | ✅ Production |
| **Task 11 Intelligence Stack** | Story understanding | Gender resolution, scene graphs, emotion-motion coupling | ✅ Production |

### Task Evolution Summary

The TTV Studio evolved through **iterative development sprints** with Task 3 as the core engine, enhanced by subsequent tasks:

| Task | Actual Purpose | Status | Location |
|------|---------------|--------|----------|
| **Task 1** | LoRA learning exercise (text/image fine-tuning) | Learning only | `LoRA_Text/`, `LoRA_StableDiffusion/` |
| **Task 2** | Motion animation system development | Integrated into Task 3 | `AnimateDiff/`, `SadTalker/` |
| **Task 3** | **CORE TTV ENGINE** (complete production pipeline) | ✅ Production | `unified_video_generator.py` |
| **Task 4** | Adaptive intelligence layer (50 users, caching, RL) | ✅ Production | `AnimateDiff/adaptive_engine/` |
| **Task 5** | Production API development | ✅ Production | `AnimateDiff_API/adaptive_api.py` |
| **Task 6** | Production hardening (Docker, stress tests) | ✅ Production | `Dockerfile`, `docker-compose.yml` |
| **Task 7** | Quality leap (RIFE, upscaling, RL optimization) | ✅ Production | `interpolator/`, `upscaler/` |
| **Task 8** | Microservice wrapper (FastAPI, Celery, storage) | ✅ Production | `ttv_service/` |
| **Task 9** | Indigenous adapters (Gurukul LoRA training) | 95% complete | `adapters/gurukul_lora/` |
| **Task 10** | Security layer (watermarking, signing, KSML) | ✅ Production | `security/` |
| **Task 11** | Intelligence stack (story, emotions, smart extension) | ✅ Production | `adaptive_engine/` (extended) |

---

## B. Architecture Diagrams

### Visual Documentation

All architecture diagrams are available in: `Documentation/Handover/Architecture Diagrams of TTV/`

**1. Development Evolution: From Learning to Production**

![Development Evolution](Architecture%20Diagrams%20of%20TTV/Diagram_1_Development%20Evolution.png)

Shows the complete evolution from Task 1 (learning) through Task 11 (intelligence stack). Key insight: Task 3 is the core engine, not a pipeline stage.

---

**2. Layered Enhancement Architecture**

![Layered Enhancement Architecture](Architecture%20Diagrams%20of%20TTV/Diagram_2_Layered%20Enhancement%20Architecture.png)

Visualizes how Tasks 4-11 layer on top of the Task 3 core engine, NOT as sequential pipeline stages.

---

**3. Task 3 Core Engine - Internal Flow**

![Task 3 Core Engine Internal Flow](Architecture%20Diagrams%20of%20TTV/Diagram_3_Task%203%20Core%20Engine%20-%20Internal%20Flow.png)

Complete internal flow of the 5-day Gurukul sprint, showing all 6 major components and their data flow.

---

**4. Evolution of Adaptive Engine (Tasks 4 + 11)**

![Adaptive Engine Evolution](Architecture%20Diagrams%20of%20TTV/Diagram_4_Evolution%20of%20Adaptive%20Enginer%20(Tasks%204%20+%2011).png)

Shows how `adaptive_engine/` was created in Task 4 and extended in Task 11. Critical for understanding file locations.

---

**5. Video Security Enhancement Process (Task 10)**

![Security Enhancement Process](Architecture%20Diagrams%20of%20TTV/Diagram_5_Video%20Security%20Enhancement%20Process%20(Task-10).png)

Complete security pipeline: KSML encryption → dual watermarking → artifact signing → content fingerprinting → audit logging.

---

**6. Task 11 Major Production Fixes**

![Task 11 Production Fixes](Architecture%20Diagrams%20of%20TTV/Diagram_6_Task-11%20Major%20Production%20Fixes.png)

Visual representation of the two critical production problems solved by Task 11: gender confusion and video looping.

---

### System Architecture (Layered View)

The TTV Studio is built as a **layered enhancement architecture**, NOT a linear pipeline:

```
┌──────────────────────────────────────────────────────────┐
│                    LESSON JSON INPUT                     │
│              {text, scenes, metadata}                    │
└────────────────────────┬─────────────────────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │                                         │
┌───▼───────────────────────────────────────────────────────────────────────┐
│  TASK 3: CORE TTV ENGINE (5-Day Gurukul Sprint) ⭐ THE MAIN SYSTEM ⭐    │
│  Location: AnimateDiff/unified_video_generator.py                         │
├───────────────────────────────────────────────────────────────────────────┤
│  1. Gemini API Text Optimization                                          │
│     - Prompt enhancement, story structure analysis                        │
│                                                                            │
│  2. AnimateDiff Motion Generation                                         │
│     - Keyframe → animated video (integrated from Task 2)                  │
│     - ControlNet pose guidance                                            │
│                                                                            │
│  3. Multi-Voice TTS + Audio Integration                                   │
│     - Bark/gTTS text-to-speech                                            │
│     - Gender-specific voice selection                                     │
│                                                                            │
│  4. Subtitle Synchronization                                              │
│     - .srt file generation with precise timing                            │
│                                                                            │
│  5. Multiple Render Styles                                                │
│     - Realistic, anime, artistic presets                                  │
│                                                                            │
│  6. Fallback System                                                       │
│     - 100% reliability with automatic error recovery                      │
│                                                                            │
│  Output: Complete educational videos (6 generated, 9.0/10 score)          │
│                                                                            │
│  Components from Task 2 (motion animation development):                   │
│  - animate_gurukul.py (AnimateDiff wrapper)                               │
│  - multi_clip_generator.py (video clip generation)                        │
│  - SadTalker/ (lip-sync integration)                                      │
│  - audio_video_pipeline/ (TTS integration)                                │
└───────────────────────────┬───────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│  ENHANCEMENT LAYER 1: Adaptive Intelligence (Task 4 - 4-Day Sprint)       │
│  Location: AnimateDiff/adaptive_engine/ (CREATED IN TASK 4, NOT TASK 11!)│
├───────────────────────────────────────────────────────────────────────────┤
│  • Device Capability Detection (RTX 3060 Ti probed)                       │
│  • Budget Planning & Tier Routing (Local → Office → Cloud)               │
│  • Intelligent Caching (backgrounds/poses/seeds) - 40-60% speedup         │
│  • NAS Storage Integration (\\192.168.0.94)                               │
│  • GPU Queue Management (4 GPUs coordinated)                              │
│  • RL Policy Optimization (Q-learning for parameter tuning)               │
│  • 50 Concurrent User Support (97.1% success rate)                        │
│  • Yotta Cloud Fallback (automatic escalation on capacity overflow)       │
│                                                                            │
│  Output: Scalable, cost-efficient generation (86.2% cost efficiency)      │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│  ENHANCEMENT LAYER 2: Production API (Task 5 - 8-Hour Sprint)             │
│  Location: AnimateDiff_API/adaptive_api.py                                │
├───────────────────────────────────────────────────────────────────────────┤
│  • POST /ttv/generate endpoint                                            │
│  • Quality presets (mobile_480p, desktop_720p)                            │
│  • Progressive preview delivery (fast user feedback)                      │
│  • BHIV microservice integration                                          │
│  • Concurrent routing (3 users tested, 100% success)                      │
│                                                                            │
│  Output: Production-ready API (score improved 6/10 → 9/10)                │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│  ENHANCEMENT LAYER 3: Quality & Production (Tasks 6-7)                    │
├───────────────────────────────────────────────────────────────────────────┤
│  TASK 6: Production Hardening (8-Hour Sprint)                             │
│  • Clean dependencies (requirements-runtime.txt, requirements-dev.txt)    │
│  • BGM integration (ffmpeg-based audio mixing)                            │
│  • Lip-sync validation endpoint                                           │
│  • Stress testing (50 concurrent users, 95% success target)               │
│  • Yotta fallback validation                                              │
│  • Docker deployment (multi-stage build)                                  │
│  • Production run script (Gunicorn + Uvicorn workers)                     │
│                                                                            │
│  TASK 7: Quality Leap (4-Day Sprint)                                      │
│  • LoRA adapter + keyframe pipeline                                       │
│  • RIFE interpolation (12fps → 24fps)                                     │
│  • Enhanced SadTalker with micro-expressions                              │
│  • Real-ESRGAN upscaling (1080p cinematic output)                         │
│  • RL policy optimization                                                 │
│  • Complete orchestration + testing                                       │
│                                                                            │
│  Output: Enterprise-grade quality (1080p, 50+ users, 97% success rate)    │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│  ENHANCEMENT LAYER 4: Microservice Architecture (Task 8)                  │
│  Location: ttv_service/                                                   │
├───────────────────────────────────────────────────────────────────────────┤
│  • FastAPI Service with async job management                              │
│  • Celery GPU Worker Queue (Redis-backed distributed tasks)               │
│  • Multi-Backend Storage (S3, Supabase, BHIV bucket, local)               │
│  • Event Emission System (webhooks + Redis pub/sub)                       │
│  • Security & Authentication (Supabase JWT validation)                    │
│  • Monitoring (Sentry error tracking, Prometheus metrics, health checks)  │
│                                                                            │
│  Output: Production microservice ready for BHIV integration               │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│  ENHANCEMENT LAYER 5: Indigenous Adapters (Task 9 - 95% Complete)         │
│  Location: adapters/gurukul_lora/, interpolator/, upscaler/               │
├───────────────────────────────────────────────────────────────────────────┤
│  • Gurukul LoRA Training Pipeline (custom fine-tuning on 500 images)      │
│  • Temporal Consistency Module (de-flicker, stabilization)                │
│  • Two-Pass Upscaling + Denoise                                           │
│  • Motion Controller with Micro-Expressions                               │
│                                                                            │
│  Status: 95% complete (training pending GPU server access)                │
│  Output: Custom-tuned generation with indigenous adapters                 │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│  ENHANCEMENT LAYER 6: Security (Task 10 - BHIV Multi-Layer Security)      │
│  Location: security/                                                      │
├───────────────────────────────────────────────────────────────────────────┤
│  • KSML-Bound Encryption (AES-256-GCM for metadata/audit logs)            │
│  • Dual Watermarking:                                                     │
│    - Invisible: FFmpeg metadata embedding (spread-spectrum, 32-bit)       │
│    - Visible: BHI logo watermark (51x50px, 35% opacity, bottom-right)     │
│  • Artifact Signing (Ed25519 cryptographic signatures for models)         │
│  • Runtime Key Validation (12-24h time-limited keys, restricted mode)     │
│  • Content Fingerprinting (SHA256 + BLAKE2b + perceptual hashing)         │
│  • Build Fingerprint (BUILD_ID seeding for watermarks)                    │
│  • 5 Critical Bugs Fixed (watermarking pipeline hardened)                 │
│                                                                            │
│  Output: Secure, traceable artifacts with provenance checking             │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│  ENHANCEMENT LAYER 7: Intelligence Stack (Task 11 - TTV Studio Core)      │
│  Location: AnimateDiff/adaptive_engine/ (EXTENDED IN TASK 11)             │
├───────────────────────────────────────────────────────────────────────────┤
│  Day 1: Story Context Parser + Identity Memory                            │
│  • Full story NLP analysis (resolves character gender from ALL sentences) │
│  • Character identity tracking (face embeddings, >0.7 similarity)         │
│  • Text condensation (20-30% reduction to minimize video looping)         │
│  • Enhanced prompts with character consistency                            │
│  • Production Fix: Gender confusion SOLVED ✅                              │
│                                                                            │
│  Day 2: Scene Memory Core                                                 │
│  • NetworkX-based scene graph (temporal relationships)                    │
│  • Entity tracking across all scenes                                      │
│  • Scene transitions and co-occurrence detection                          │
│  • 13 query methods for entity history                                    │
│                                                                            │
│  Day 3: Narrative Sequencer v1                                            │
│  • Story beat classification (Setup, Rising Action, Climax, etc.)         │
│  • Character arc tracking (5 stages from Introduction to New Equilibrium) │
│  • Dialogue flow analysis (type, speaker, emotion, subtext)               │
│  • Pacing analysis with tension curve                                     │
│                                                                            │
│  Day 4: Emotion Controller                                                │
│  • 6 core emotions (joy, sadness, anger, fear, neutral, surprise)         │
│  • Motion-emotion coupling (maps emotions → motion intensity/gestures)    │
│  • Cross-scene emotional continuity (smooth transitions)                  │
│  • Micro-expression scheduling with keyframe timing                       │
│                                                                            │
│  Day 5: Smart Video Extension                                             │
│  • SlowMo + Freeze extension (NO RIFE - avoids black screens)             │
│  • Production Fix: Video looping SOLVED ✅                                 │
│  • Cinematic transitions (8 types: fade, dissolve, wipe)                  │
│  • Perfect audio-video sync (<0.5s difference)                            │
│                                                                            │
│  Day 6: TTV Intelligence Metrics                                          │
│  • Extended audit_logger with 26 metrics tracked                          │
│  • Story analysis, scene graph, narrative, emotion, extension, quality    │
│  • KSML-compliant logging (append-only, tamper-evident)                   │
│  • Dashboard backend ready for UI integration                             │
│                                                                            │
│  Output: Intelligent, context-aware videos with production fixes          │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
┌───────▼──────────────────────────────────────────────────────────────────┐
│                        FINAL OUTPUT                                        │
├───────────────────────────────────────────────────────────────────────────┤
│  • Video: 1080p MP4 (H.264, 8000k bitrate, VS Code playable)              │
│  • Subtitles: Synchronized .srt file                                      │
│  • Watermarks: Dual-layer (invisible FFmpeg + visible BHI logo 35%)       │
│  • Fingerprint: SHA256 + BLAKE2b content hash (.json file)                │
│  • Security Metadata: KSML-compliant with BUILD_ID                        │
│  • Audit Log: Complete entry with 26 intelligence metrics                 │
│  • Perfect Sync: Audio-video difference <0.5s                             │
│  • Quality: Professional 8000k bitrate, H.264 encoded                     │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  FOUNDATION LEARNING (Task 1) - NOT PART OF PRODUCTION PIPELINE           │
│  Location: LoRA_Text/, LoRA_StableDiffusion/, VideoMaker/                 │
├───────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Text Fine-Tuning (DistilBERT on IMDb sentiment, 87% accuracy)   │
│  Phase 2: Image Fine-Tuning (Stable Diffusion v1.4, Gurukul themes)       │
│  Phase 3: Visual Sequencing (Frame generation with MoviePy)               │
│                                                                            │
│  Purpose: Learning exercise to understand LoRA fundamentals                │
│  Status: Educational only, NOT used in production videos                  │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  MOTION ANIMATION DEVELOPMENT (Task 2) - INTEGRATED INTO TASK 3           │
│  Location: AnimateDiff/, SadTalker/, audio_video_pipeline/                │
├───────────────────────────────────────────────────────────────────────────┤
│  • AnimateDiff + ControlNet (OpenPose/Depth/Canny) integration            │
│  • SadTalker lip-sync development                                         │
│  • Multi-voice TTS with gender detection                                  │
│  • 20+ generated video samples during development                         │
│                                                                            │
│  Purpose: Motion animation capabilities development                       │
│  Status: Components integrated into Task 3 core engine                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Component Map (Task Purposes)

**Task 1: LoRA Basics - Learning Exercise**
- **Location:** `LoRA_Text/`, `LoRA_StableDiffusion/`, `VideoMaker/`
- **Purpose:** Foundation learning (text/image fine-tuning, visual sequencing)
- **Status:** Educational only, NOT part of production pipeline
- **Deliverables:**
  - Phase 1: DistilBERT fine-tuning (IMDb sentiment, 87% accuracy)
  - Phase 2: Stable Diffusion v1.4 fine-tuning (Gurukul-themed images)
  - Phase 3: MoviePy visual sequencing experiments

**Task 2: Motion-Aware Character Animation**
- **Location:** `AnimateDiff/`, `SadTalker/`, `audio_video_pipeline/`
- **Purpose:** Motion animation system development
- **Status:** Components integrated into Task 3 core engine
- **Deliverables:**
  - AnimateDiff + ControlNet (OpenPose) integration
  - SadTalker lip-sync system
  - Multi-voice TTS (Bark/gTTS) with gender detection
  - 20+ generated video samples
  - Production integration with web UI

**Task 3: 5-Day Gurukul Hyperdrive Sprint - CORE TTV ENGINE ⭐**
- **Location:** `AnimateDiff/unified_video_generator.py`, `generate_lesson_video_safe.py`
- **Purpose:** Complete text-to-video engine for educational lessons
- **Status:** ✅ Production (9.0/10 score, enterprise-grade)
- **Deliverables:**
  - Gemini API text optimization (prompt enhancement)
  - AnimateDiff motion generation (uses Task 2 components)
  - Multi-voice TTS + audio integration
  - Subtitle synchronization (.srt generation)
  - Multiple render styles (realistic, anime, artistic)
  - Fallback system (100% reliability)
  - 6 complete videos generated (exceeded 4 requirement)
  - Team collaboration: Gandhar (TTS), Rishabh (Frontend), Vedant (API), Akash (Text)

**Task 4: 4-Day Adaptive Video Generation System**
- **Location:** `AnimateDiff/adaptive_engine/` **(CREATED IN TASK 4!)**
- **Purpose:** Adaptive intelligence layer for scaling and optimization
- **Status:** ✅ Production (97.1% success rate, 86.2% cost efficiency)
- **Deliverables:**
  - Day 1: Device probe (RTX 3060 Ti), budget planning, tier routing
  - Day 2: Intelligent caching (backgrounds/poses/seeds), RL policy, compression, VMAF
  - Day 3: NAS storage (\\\\192.168.0.94), GPU queue (4 GPUs), mixed precision, lip-sync
  - Day 4: 50 concurrent users, graceful degradation, Yotta fallback, analytics
  - Performance: 97.1% success rate, 86.2% cost efficiency

**Task 5: 8-Hour Adaptive API Sprint**
- **Location:** `AnimateDiff_API/adaptive_api.py`
- **Purpose:** Production API development
- **Status:** ✅ Production (score 6/10 → 9/10)
- **Deliverables:**
  - `/ttv/generate` endpoint (main API)
  - Quality presets (mobile_480p, desktop_720p)
  - Progressive preview delivery
  - BHIV microservice integration
  - Concurrent routing (3 users tested, 100% success)

**Task 6: Production Hardening Sprint (8 Hours)**
- **Location:** `requirements-runtime.txt`, `Dockerfile`, `docker-compose.yml`, `run-prod.sh`
- **Purpose:** Production deployment readiness
- **Status:** ✅ Complete
- **Deliverables:**
  - Clean dependencies (runtime/dev split)
  - BGM integration (ffmpeg-based)
  - Lip-sync validation endpoint
  - Stress testing (50 concurrent users)
  - Yotta fallback validation
  - Docker deployment (multi-stage build)

**Task 7: Quality Leap - Cinematic Video Generation Sprint (4 Days)**
- **Location:** `adapters/`, `interpolator/`, `upscaler/`, `motion_controller/`, `orchestrator.py`
- **Purpose:** Enterprise-grade quality improvements
- **Status:** ✅ Complete (97% success rate, <3 min generation)
- **Deliverables:**
  - LoRA adapter training + keyframe pipeline
  - RIFE interpolation (12fps → 24fps)
  - Enhanced SadTalker with micro-expressions + VASA-1
  - Real-ESRGAN upscaling (1080p)
  - RL policy optimization
  - Complete orchestration + cloud fallback + testing

**Task 8: TTV Service Integration - Production Microservice**
- **Location:** `ttv_service/`
- **Purpose:** Complete microservice wrapper for BHIV integration
- **Status:** ✅ Complete (production-ready)
- **Deliverables:**
  - FastAPI service with async job management
  - Celery GPU worker queue (Redis-backed)
  - Multi-backend storage (S3, Supabase, BHIV bucket, local)
  - Event emission system (webhooks + Redis pub/sub)
  - Security & authentication (Supabase JWT validation)
  - Production deployment (Docker)
  - Monitoring (Sentry, Prometheus, health checks)

**Task 9: TTV-Studio Quality Harden - Indigenous Image Adapter**
- **Location:** `adapters/gurukul_lora/`, `interpolator/temporal_consistency.py`, `upscaler/`
- **Purpose:** Indigenous keyframe generation + quality hardening
- **Status:** 95% complete (training pending GPU access)
- **Deliverables:**
  - Gurukul LoRA training pipeline (500 curated images ready)
  - Temporal consistency module (de-flicker, stabilization)
  - Two-pass upscaling + denoise
  - Motion controller with micro-expressions
  - Training script tested (1-epoch validation successful)

**Task 10: BHIV Multi-Layer Security**
- **Location:** `security/`
- **Purpose:** Complete security implementation
- **Status:** ✅ Complete (9/9 required tasks, 5 bugs fixed)
- **Deliverables:**
  - KSML-bound encryption (AES-256-GCM)
  - Dual watermarking (invisible FFmpeg + visible BHI logo)
  - Artifact signing (Ed25519 cryptographic signatures)
  - Runtime key validation (12-24h time-limited keys)
  - Content fingerprinting (SHA256 + BLAKE2b)
  - Build fingerprinting (BUILD_ID seeding)
  - Detection pipeline for provenance checking
  - Audit logging with security metadata
  - CI/CD security gates

**Task 11: Phase III - TTV Studio Core - Intelligence Stack**
- **Location:** `AnimateDiff/adaptive_engine/` **(EXTENDED IN TASK 11)**
- **Purpose:** Story intelligence + production problem fixes
- **Status:** ✅ Complete (100% of production problems solved)
- **Deliverables:**
  - Day 1: Story context parser + identity memory + text condensation
  - Day 2: Scene memory core (NetworkX scene graph)
  - Day 3: Narrative sequencer (story beats, character arcs)
  - Day 4: Emotion controller (motion-emotion coupling)
  - Day 5: Smart video extension (SlowMo + Freeze, NO RIFE)
  - Day 6: TTV intelligence metrics (26 metrics tracked)
  - Day 7: Final deliverables (audit report, user guide)
  - **Production Fixes:** Gender confusion solved, video looping solved

### File Structure

```
LoRA_TextToVision/
├── AnimateDiff/                          # Task 3 core + Task 2 components
│   ├── unified_video_generator.py        # ⭐ Main orchestrator (Task 3)
│   ├── generate_lesson_video_safe.py     # ⭐ Entry point
│   ├── animate_gurukul.py                # AnimateDiff wrapper (Task 2)
│   ├── multi_clip_generator.py           # Clip generation (Task 2)
│   ├── subtitle_sync_engine.py           # Subtitle generation (Task 3)
│   ├── adaptive_engine/                  # Task 4 (created) + Task 11 (extended)
│   │   ├── story_context_parser.py       # Task 11 Day 1
│   │   ├── identity_memory.py            # Task 11 Day 1
│   │   ├── scene_memory_core.py          # Task 11 Day 2
│   │   ├── narrative_sequencer_v1.py     # Task 11 Day 3
│   │   ├── emotion_controller.py         # Task 11 Day 4
│   │   ├── smart_video_extender.py       # Task 11 Day 5
│   │   ├── cinematic_transition_core.py  # Task 11 Day 5
│   │   └── [Task 4 modules]              # Device probe, caching, RL, etc.
│   └── analytics/                        # Task 4 analytics
│
├── AnimateDiff_API/                      # Task 5 production API
│   ├── adaptive_api.py                   # Main API endpoints
│   └── api_clean.py                      # Clean API implementation
│
├── SadTalker/                            # Task 2 lip-sync integration
│
├── audio_video_pipeline/                 # Task 2 TTS integration
│
├── adapters/                             # Task 7 + Task 9
│   ├── adapter_manager.py                # Task 7 (with signature verification)
│   ├── gurukul_lora/                     # Task 9 indigenous adapters
│   │   ├── train_adapter.py              # Training script
│   │   ├── dataset_curator.py            # 500 images curated
│   │   └── checkpoint.pt                 # LoRA checkpoint (89MB)
│   └── lora_adapter.py                   # Task 7 base LoRA wrapper
│
├── interpolator/                         # Task 7 + Task 9
│   ├── rife_interpolator.py              # Task 7 RIFE (12fps→24fps)
│   └── temporal_consistency.py           # Task 9 de-flicker
│
├── upscaler/                             # Task 7 + Task 9
│   ├── esrgan_upscaler.py                # Task 7 Real-ESRGAN (1080p)
│   └── tile_upscale.py                   # Task 9 two-pass upscaling
│
├── motion_controller/                    # Task 7 + Task 9
│   └── policy.py                         # RL policy optimization
│
├── security/                             # Task 10
│   ├── ksml_encryption.py                # KSML encryption (370 lines)
│   ├── artifact_signer.py                # Ed25519 signing (450 lines)
│   ├── runtime_validator.py              # Runtime key validation (380 lines)
│   ├── watermark.py                      # Invisible watermarking (420 lines)
│   ├── visible_watermark.py              # Visible BHI logo (450 lines)
│   ├── keys/
│   │   └── signing_key.pub               # Ed25519 public key
│   └── watermark_logo/
│       └── BHI_logo.png                  # 51x50px logo with transparency
│
├── ttv_service/                          # Task 8 microservice
│   ├── main.py                           # FastAPI application
│   ├── job_manager.py                    # Celery job orchestration
│   ├── tasks.py                          # Celery task definitions
│   ├── storage.py                        # Multi-backend storage
│   ├── events.py                         # Event emission system
│   ├── security.py                       # Authentication
│   └── monitoring.py                     # Health checks + metrics
│
├── orchestrator.py                       # Task 7 main orchestrator
├── yotta_fallback.py                     # Task 4 + Task 7 cloud fallback
├── audit_logger.py                       # Task 10 + Task 11 extended logging
├── insightflow_client.py                 # Task 10 telemetry
│
├── LoRA_Text/                            # Task 1 learning (NOT production)
├── LoRA_StableDiffusion/                 # Task 1 learning (NOT production)
├── VideoMaker/                           # Task 1 learning (NOT production)
│
├── Dockerfile                            # Task 6 production deployment
├── docker-compose.yml                    # Task 6 orchestration
├── run-prod.sh                           # Task 6 production run script
├── requirements-runtime.txt              # Task 6 clean dependencies
├── requirements-dev.txt                  # Task 6 dev dependencies
│
└── tests/                                # Comprehensive test suite
    ├── test_task9_integration.py         # Task 9 tests
    ├── test_task10_integration.py        # Task 10 tests (5/5 passing)
    ├── test_day6_ttv_metrics.py          # Task 11 Day 6 tests (14/14 passing)
    └── [152 total tests, 100% passing]
```

---

## C. Important Concepts

### 1. Gurukul is NOT a Theme - It's a Project Name

**CRITICAL CLARIFICATION:**

❌ **WRONG:** "Gurukul" means ancient Indian school aesthetic with sages and traditional themes  
✅ **CORRECT:** "Gurukul" is just the project brand name, like "YouTube" or "Coursera"

**What This Means:**
- The system handles **ANY educational content**: physics, programming, history, cooking, science, arts, sports, etc.
- There are **NO thematic limitations** or style constraints
- Users can learn **any concept** they search for
- LoRA adapters should be trained on **diverse datasets**, not locked to one visual style
- Focus: **Educational effectiveness**, NOT specific cultural aesthetics

### 2. Task 3 is the Core Engine, Not Task 1

**Previous Misunderstanding:** Tasks 1-5 were sequential pipeline stages

**Reality:**
- **Task 1** = Learning exercise (LoRA basics)
- **Task 2** = Component development (motion animation)
- **Task 3** = **Complete TTV engine** (the main production system)
- **Task 4** = Adaptive enhancements (scaling, caching, RL)
- **Task 5** = API layer (production endpoints)

### 3. adaptive_engine/ Was Created in Task 4, NOT Task 11

**Critical File Location Error:**

❌ **WRONG:** `adaptive_engine/` is from Task 11  
✅ **CORRECT:** `adaptive_engine/` was **created in Task 4**, **extended in Task 11**

**Task 4 Created:**
- Device probe, tier routing
- Intelligent caching system
- NAS storage integration
- GPU queue management
- RL policy optimization
- Yotta fallback system

**Task 11 Extended:**
- Story context parser
- Scene memory core
- Narrative sequencer
- Emotion controller
- Smart video extender
- TTV metrics

### 4. Layered Architecture, Not Linear Pipeline

**The system evolved through iterative enhancements:**

```
Layer 0: Foundation Learning (Task 1) - Educational only
Layer 1: Component Development (Task 2) - Integrated into Task 3
Layer 2: Core TTV Engine (Task 3) - Main production system ⭐
Layer 3: Adaptive Intelligence (Task 4) - Scaling + optimization
Layer 4: Production API (Task 5) - API endpoints
Layer 5: Quality & Production (Tasks 6-7) - Hardening + quality
Layer 6: Microservice Architecture (Task 8) - Service wrapper
Layer 7: Indigenous Adapters (Task 9) - Custom fine-tuning
Layer 8: Security (Task 10) - Watermarking + signing
Layer 9: Intelligence Stack (Task 11) - Story understanding
```

### 5. Production Problems Solved by Task 11

**Problem 1: Gender Confusion (SOLVED ✅)**
- **Before:** "seeker" → assumed male, then "She" → switched to female
- **Solution:** Full story NLP analysis resolves gender from ALL sentences
- **Module:** `story_context_parser.py` (Task 11 Day 1)

**Problem 2: Video Looping (SOLVED ✅)**
- **Before:** 2s clip looped 3x to match 6s audio = repetitive
- **Solution:** Smart extension (SlowMo + Freeze), NO repetitive looping
- **Module:** `smart_video_extender.py` (Task 11 Day 5)

**Problem 3: RIFE Black Screens (AVOIDED ✅)**
- **Solution:** Don't use RIFE for extension, use frame duplication + freeze
- **Module:** `smart_video_extender.py` (Task 11 Day 5)

### 6. Security is Production-Critical

**All videos now include:**
- Dual watermarks (invisible FFmpeg metadata + visible BHI logo 35%)
- Content fingerprint (SHA256 + BLAKE2b)
- Build fingerprint (BUILD_ID seeding)
- Artifact signatures (Ed25519)
- KSML-compliant audit trail
- Security metadata in every log entry

**5 Critical Bugs Fixed:**
1. LSB watermarking not working (switched to FFmpeg metadata)
2. FFmpeg audio restoration stripping metadata
3. -map_metadata not copying custom tags
4. -c copy stripping custom MP4 metadata
5. H.264 encoding stripping custom tags

---

### 7. KSML Tokens

**What is KSML?**

KSML (Karma State Machine Language) is a security framework for tracking provenance and lineage of AI-generated artifacts.

**KSML Token Structure:**

```json
{
  "ksml_token": "ksml_production",
  "intent": "video_generation",
  "karma_state": "authorized",
  "lineage": {
    "lesson": "Photosynthesis Basics",
    "style": "realistic",
    "build_id": "build_20251127_001"
  }
}
```

**How KSML Tokens Are Used:**

1. **Token Creation** (at video generation start):
   ```python
   # unified_video_generator.py
   ksml_token_data = {
       "ksml_token": os.getenv('KSML_TOKEN', 'ksml_production'),
       "intent": "video_generation",
       "karma_state": "authorized",
       "lineage": {
           "lesson": lesson_title,
           "style": style,
           "build_id": build_id
       }
   }
   ```

2. **Token Binding to Encryption** (security/ksml_encryption.py):
   ```python
   encrypted = ksml_encrypt_json(metadata, ksml_token="ksml_production")
   # Metadata is cryptographically bound to KSML token
   ```

3. **Token in Audit Logs** (every log entry):
   ```python
   audit_logger.log_video_generation(
       prompt=text,
       output_path=video_path,
       ksml_token=ksml_token_data,  # Token included
       security_metadata={...}
   )
   ```

4. **Token Verification** (at runtime):
   ```python
   # Runtime validator checks KSML token validity
   if not validate_ksml_token(ksml_token):
       raise SecurityError("Invalid KSML token")
   ```

**KSML Token Environments:**

- `ksml_development` - Development environment (lenient validation)
- `ksml_staging` - Staging environment (strict validation)
- `ksml_production` - Production environment (full security)

**Why KSML Matters:**

- ✅ Ensures all artifacts are traceable to authorized operations
- ✅ Prevents unauthorized video generation
- ✅ Enables audit trail reconstruction
- ✅ Supports compliance requirements (data lineage)

---

### 8. Lineage Tracking

**What is Lineage Tracking?**

Complete traceability of every video from prompt input to final artifact, including all transformations, models used, and security operations.

**Lineage Data Captured:**

```python
# Example lineage data in audit log
{
  "video_lineage": {
    # Input
    "input_prompt": "Explain photosynthesis",
    "input_timestamp": "2025-11-27T10:30:00Z",
    
    # Models Used
    "models": {
      "text_optimizer": "gemini-pro-1.5",
      "motion_generator": "animatediff-v3",
      "lora_adapter": "gurukul_lora_v1",
      "tts_engine": "bark-v1",
      "upscaler": "realesrgan-x4plus"
    },
    
    # Processing Steps
    "pipeline_stages": [
      {"stage": "text_optimization", "duration_sec": 2.3},
      {"stage": "keyframe_generation", "duration_sec": 15.7},
      {"stage": "motion_generation", "duration_sec": 45.2},
      {"stage": "interpolation", "duration_sec": 12.1},
      {"stage": "upscaling", "duration_sec": 23.4},
      {"stage": "audio_synthesis", "duration_sec": 8.9},
      {"stage": "watermarking", "duration_sec": 1.2}
    ],
    
    # Security Operations
    "security": {
      "watermark_build_id": "build_20251127_001",
      "fingerprint_sha256": "abc123...",
      "signature_algorithm": "Ed25519",
      "ksml_token": "ksml_production"
    },
    
    # Output
    "output_path": "AnimateDiff/storage/2025-11-27/lesson_001.mp4",
    "output_size_mb": 125.3,
    "output_duration_sec": 45.0
  }
}
```

**Lineage Chain Verification:**

```python
# tools/verify_lineage.py
def verify_video_lineage(video_path: str) -> bool:
    """Verify complete lineage chain of a video"""
    
    # 1. Extract watermark
    watermark = detect_watermark(video_path)
    if not watermark:
        return False
    
    # 2. Find audit log entry
    build_id = watermark['build_id']
    audit_entry = find_audit_entry_by_build_id(build_id)
    if not audit_entry:
        return False
    
    # 3. Verify fingerprint matches
    actual_fingerprint = compute_fingerprint(video_path)
    logged_fingerprint = audit_entry['security']['fingerprint_sha256']
    if actual_fingerprint != logged_fingerprint:
        return False
    
    # 4. Verify KSML token
    if not validate_ksml_token(audit_entry['ksml_token']):
        return False
    
    return True
```

**Use Cases for Lineage Tracking:**

- 🔍 **Debugging:** Trace back to exact models/parameters that generated a video
- 🔒 **Security:** Detect tampered or unauthorized videos
- 📊 **Analytics:** Analyze which models/settings produce best results
- ⚖️ **Compliance:** Prove data provenance for regulatory requirements

---

### 9. Telemetry Integration

**What is Telemetry?**

Automated collection and transmission of performance, usage, and quality metrics to InsightFlow analytics platform.

**InsightFlow Integration:**

```python
# insightflow_client.py
class InsightFlowClient:
    def track_video_generation(self, metadata: Dict):
        """Send video generation event to InsightFlow"""
        
        event = {
            "event_type": "video_generation",
            "timestamp": datetime.now().isoformat(),
            "user_id": metadata.get("user_id"),
            "video_id": metadata.get("build_id"),
            "duration_sec": metadata.get("generation_time"),
            "resolution": metadata.get("resolution"),
            "style": metadata.get("style"),
            "quality_score": metadata.get("quality_score"),
            "gpu_model": metadata.get("gpu_model"),
            "success": metadata.get("success")
        }
        
        # Send to InsightFlow API
        response = requests.post(
            f"{self.endpoint}/events",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=event
        )
        
        return response.status_code == 200
```

**Metrics Tracked (26 Total):**

**Story Analysis Metrics (6):**
- Character count
- Gender resolution success rate
- Text condensation percentage
- Prompt enhancement length
- Story complexity score
- Narrative structure type

**Scene Memory Metrics (5):**
- Total scenes
- Total entities tracked
- Scene transitions
- Entity co-occurrence count
- Scene graph depth

**Narrative Metrics (4):**
- Story beats detected
- Character arcs tracked
- Tension curve variance
- Dialogue flow score

**Emotion Metrics (3):**
- Emotion changes per scene
- Dominant emotion distribution
- Motion intensity variance

**Video Extension Metrics (3):**
- Clips extended count
- Extension method used (SlowMo/Freeze)
- Extension ratio (extended_duration / original_duration)

**Quality Metrics (5):**
- Audio-video sync difference (seconds)
- Video duration (seconds)
- Video FPS
- Video bitrate (kbps)
- VMAF quality score (if available)

**Telemetry Configuration:**

```bash
# .env
INSIGHTFLOW_ENABLED=true
INSIGHTFLOW_API_KEY=your_api_key
INSIGHTFLOW_ENDPOINT=https://api.insightflow.io/v1
INSIGHTFLOW_BATCH_SIZE=100  # Send in batches
INSIGHTFLOW_FLUSH_INTERVAL_SEC=60  # Flush every 60 seconds
```

**Querying Telemetry Data:**

```python
# Query InsightFlow for analytics
client = InsightFlowClient()

# Get average generation time by GPU model
stats = client.query_metrics(
    metric="generation_time_sec",
    group_by="gpu_model",
    time_range="last_7_days"
)

# Example output:
# {
#   "RTX 3060 Ti": {"avg": 180.5, "count": 234},
#   "RTX 3070": {"avg": 120.3, "count": 156},
#   "RTX 3080": {"avg": 85.2, "count": 89}
# }
```

---

### 10. Scene Memory (Phase III / Task 11)

**What is Scene Memory?**

A NetworkX-based graph database that tracks all entities, their relationships, and temporal evolution across all scenes in a story.

**Scene Memory Architecture:**

```python
# AnimateDiff/adaptive_engine/scene_memory_core.py
class SceneMemoryCore:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for temporal relationships
        self.entities = {}         # Entity registry
        self.scenes = []           # Scene sequence
    
    def add_scene(self, scene_id: str, entities: List[str], timestamp: float):
        """Add a scene with its entities"""
        
        # Add scene node
        self.graph.add_node(scene_id, type="scene", timestamp=timestamp)
        
        # Add entity nodes
        for entity in entities:
            if entity not in self.entities:
                self.graph.add_node(entity, type="entity")
                self.entities[entity] = {"appearances": []}
            
            # Link entity to scene
            self.graph.add_edge(entity, scene_id, relation="appears_in")
            
            # Track appearance
            self.entities[entity]["appearances"].append(scene_id)
    
    def get_entity_history(self, entity: str) -> List[str]:
        """Get all scenes where entity appeared"""
        return self.entities.get(entity, {}).get("appearances", [])
    
    def get_co_occurring_entities(self, entity: str) -> Set[str]:
        """Get all entities that appeared in same scenes"""
        co_occurring = set()
        
        for scene in self.get_entity_history(entity):
            # Get all entities in this scene
            scene_entities = [
                node for node in self.graph.predecessors(scene)
                if self.graph.nodes[node].get("type") == "entity"
            ]
            co_occurring.update(scene_entities)
        
        co_occurring.discard(entity)  # Remove self
        return co_occurring
```

**Scene Memory Queries (13 Methods):**

```python
# 1. Get entity first appearance
memory.get_entity_first_appearance("teacher")
# Returns: "scene_001"

# 2. Get entity last appearance
memory.get_entity_last_appearance("teacher")
# Returns: "scene_005"

# 3. Get scene entities
memory.get_scene_entities("scene_003")
# Returns: ["teacher", "student", "classroom"]

# 4. Get scene transitions
memory.get_scene_transitions()
# Returns: [("scene_001", "scene_002"), ("scene_002", "scene_003"), ...]

# 5. Get entity co-occurrences
memory.get_co_occurring_entities("teacher")
# Returns: {"student", "classroom", "book"}

# 6. Get entity relationship strength
memory.get_relationship_strength("teacher", "student")
# Returns: 0.85 (appeared together in 85% of scenes)

# 7. Get scene temporal distance
memory.get_temporal_distance("scene_001", "scene_005")
# Returns: 4 (number of scenes between)

# 8. Get entity arc
memory.get_entity_arc("student")
# Returns: ["introduced", "learning", "struggling", "understanding", "mastery"]

# 9. Check entity presence in scene
memory.is_entity_in_scene("teacher", "scene_003")
# Returns: True

# 10. Get scene context
memory.get_scene_context("scene_003", window=1)
# Returns: {"prev": "scene_002", "current": "scene_003", "next": "scene_004"}
```

**Why Scene Memory Matters:**

- ✅ **Consistency:** Ensures character consistency across scenes (same face embeddings)
- ✅ **Context:** Provides narrative context for each scene
- ✅ **Transitions:** Enables smooth scene transitions based on entity continuity
- ✅ **Intelligence:** Powers smart decisions (e.g., which characters to include in a scene)

---

### 11. Identity Embeddings

**What are Identity Embeddings?**

Face embedding vectors (512-dimensional) that uniquely identify characters across all scenes, ensuring visual consistency.

**How Identity Embeddings Work:**

```python
# AnimateDiff/adaptive_engine/identity_memory.py
class IdentityMemory:
    def __init__(self):
        self.face_recognition = FaceRecognition()
        self.identities = {}  # character_name -> face_embedding
        self.similarity_threshold = 0.7
    
    def register_identity(self, character_name: str, reference_image: np.ndarray):
        """Register a character's face embedding"""
        
        # Extract face embedding (512-dimensional vector)
        embedding = self.face_recognition.get_embedding(reference_image)
        
        # Store embedding
        self.identities[character_name] = {
            "embedding": embedding,
            "reference_image": reference_image,
            "appearances": 0
        }
    
    def verify_identity(self, character_name: str, test_image: np.ndarray) -> bool:
        """Verify if test image matches character's identity"""
        
        if character_name not in self.identities:
            return False
        
        # Get test embedding
        test_embedding = self.face_recognition.get_embedding(test_image)
        
        # Compare with stored embedding
        stored_embedding = self.identities[character_name]["embedding"]
        similarity = self._cosine_similarity(stored_embedding, test_embedding)
        
        return similarity >= self.similarity_threshold
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

**Identity Embedding Pipeline:**

```
1. Story Analysis
   ↓
   Identify Characters: ["teacher", "student"]
   
2. First Appearance
   ↓
   Generate Reference Frame → Extract Face Embedding → Store as Identity
   
3. Subsequent Scenes
   ↓
   Generate Frame → Extract Face → Compare to Stored Identity
   ↓
   If Similarity < 0.7: Regenerate with Different Seed
   ↓
   If Similarity >= 0.7: Accept Frame
   
4. All Scenes
   ↓
   Same Character = Same Face (>70% similarity)
```

**Identity Consistency Enforcement:**

```python
# During video generation
def generate_scene_with_identity_consistency(scene_data, character_name):
    """Generate scene ensuring character identity consistency"""
    
    max_retries = 5
    
    for attempt in range(max_retries):
        # Generate scene frame
        frame = generate_frame(scene_data)
        
        # Verify identity
        if identity_memory.verify_identity(character_name, frame):
            # Identity matches, use this frame
            return frame
        
        # Identity doesn't match, try different seed
        scene_data['seed'] = random.randint(0, 2**32)
    
    # If all retries fail, use best match
    return frame
```

**Why Identity Embeddings Matter:**

- ✅ **Visual Consistency:** Same character looks the same across all scenes
- ✅ **Quality:** Prevents jarring changes in character appearance
- ✅ **Storytelling:** Maintains viewer immersion
- ✅ **Professional Output:** Enterprise-grade video quality

---

### 12. Motion Controllers

**What are Motion Controllers?**

Systems that control animation motion parameters based on narrative context, emotions, and scene requirements.

**Motion Controller Architecture:**

```python
# AnimateDiff/adaptive_engine/emotion_controller.py
class EmotionController:
    """Controls motion based on emotions"""
    
    EMOTION_MOTION_MAP = {
        "joy": {
            "intensity": 0.8,       # High motion intensity
            "gesture_frequency": 1.2,  # Frequent gestures
            "motion_speed": 1.3,    # Faster motion
            "expression": "smile"
        },
        "sadness": {
            "intensity": 0.3,       # Low motion intensity
            "gesture_frequency": 0.5,  # Infrequent gestures
            "motion_speed": 0.7,    # Slower motion
            "expression": "frown"
        },
        "anger": {
            "intensity": 0.9,       # Very high intensity
            "gesture_frequency": 1.5,  # Very frequent gestures
            "motion_speed": 1.4,    # Fast, sharp motions
            "expression": "angry"
        },
        # ... 6 core emotions total
    }
    
    def get_motion_params(self, emotion: str) -> Dict:
        """Get motion parameters for an emotion"""
        return self.EMOTION_MOTION_MAP.get(emotion, {})
```

**Motion Controller Types:**

**1. Emotion Controller** (Task 11 Day 4):
- Maps emotions → motion parameters
- 6 core emotions: joy, sadness, anger, fear, surprise, neutral
- Controls: intensity, gesture frequency, motion speed, facial expressions

**2. Motion Intensity Controller** (Task 7):
- RL-based parameter optimization
- Learns optimal motion intensity from quality feedback
- Q-learning algorithm with reward signal

**3. Micro-Expression Controller** (Task 7):
- Subtle facial expressions (eyebrow raises, eye movements)
- Keyframe-based scheduling
- Enhances realism and emotional depth

**4. Gesture Controller** (motion_controller/policy.py):
- Hand gestures, body language
- Context-aware gesture selection
- Timing and amplitude control

**Motion Parameter Example:**

```python
# Example motion parameters for a scene
motion_params = {
    # Base parameters
    "motion_intensity": 0.7,        # 0.0 (static) to 1.0 (very dynamic)
    "motion_speed": 1.0,            # Speed multiplier
    "motion_smoothness": 0.8,       # Temporal smoothness
    
    # Gestures
    "gesture_frequency": 1.0,       # Gestures per second
    "gesture_amplitude": 0.6,       # Size of gestures
    
    # Facial expressions
    "expression_type": "smile",     # Primary expression
    "expression_intensity": 0.7,    # Expression strength
    "micro_expressions": [          # Subtle expressions
        {"type": "eyebrow_raise", "timing": 2.3, "intensity": 0.3},
        {"type": "eye_movement", "timing": 4.1, "intensity": 0.5}
    ],
    
    # Camera motion
    "camera_motion": "subtle_pan",  # Camera movement type
    "camera_speed": 0.5,            # Camera motion speed
    
    # RL optimization
    "rl_policy_version": "v2.1",    # Which RL policy to use
    "rl_confidence": 0.85           # Policy confidence
}
```

**Why Motion Controllers Matter:**

- ✅ **Emotional Depth:** Motion matches narrative emotion
- ✅ **Quality:** Prevents robotic or unnatural motion
- ✅ **Storytelling:** Enhances narrative through body language
- ✅ **Optimization:** RL continuously improves motion quality

---

### 13. Two-Pass Upscaling

**What is Two-Pass Upscaling?**

A two-stage upscaling process that first denoises at original resolution, then upscales to target resolution, producing sharper results than single-pass upscaling.

**Two-Pass Upscaling Pipeline:**

```
Input: 512x512 video frame
       ↓
┌──────────────────────────┐
│   PASS 1: DENOISE        │
│   at Original Resolution │
└──────────────────────────┘
       ↓
   Real-ESRGAN (denoise mode)
   • Remove compression artifacts
   • Fix color banding
   • Preserve details
       ↓
   Clean 512x512 frame
       ↓
┌──────────────────────────┐
│   PASS 2: UPSCALE        │
│   to Target Resolution   │
└──────────────────────────┘
       ↓
   Real-ESRGAN (upscale mode)
   • Upscale to 1920x1080 (4x)
   • Enhance edges
   • Add fine details
       ↓
   Final 1920x1080 frame
```

**Implementation:**

```python
# upscaler/tile_upscale.py
class TwoPassUpscaler:
    def __init__(self):
        self.esrgan = RealESRGAN()
    
    def upscale_two_pass(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Two-pass upscaling for maximum quality"""
        
        # PASS 1: Denoise at original resolution
        denoised = self.esrgan.process(
            frame,
            mode="denoise",
            scale=1,  # Same resolution
            tile_size=512  # Process in tiles to save VRAM
        )
        
        # PASS 2: Upscale to target resolution
        upscaled = self.esrgan.process(
            denoised,
            mode="upscale",
            scale=4,  # 4x upscaling (512 → 2048)
            tile_size=512
        )
        
        # Resize to exact target if needed
        if upscaled.shape[:2] != target_size:
            upscaled = cv2.resize(upscaled, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)
        
        return upscaled
```

**Quality Comparison:**

| Method | Quality (VMAF) | Processing Time | VRAM Usage |
|--------|---------------|-----------------|------------|
| Single-pass upscale | 72.3 | 8 sec | 4.2 GB |
| Two-pass upscale | 85.7 | 15 sec | 5.8 GB |
| **Quality Gain** | **+18.5%** | **+87.5%** | **+38.1%** |

**When to Use Two-Pass:**

- ✅ **Production videos** (quality is critical)
- ✅ **1080p or higher** (4K output)
- ✅ **Final renders** (not previews)
- ❌ **Real-time previews** (too slow)
- ❌ **Low-end GPUs** (VRAM constraints)

**Tile-Based Processing:**

```python
# Process large frames in tiles to avoid OOM
def upscale_large_frame(frame, tile_size=512, overlap=32):
    """Upscale large frame using tiled processing"""
    
    height, width = frame.shape[:2]
    tiles = []
    
    # Split into overlapping tiles
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            tile = frame[y:y+tile_size, x:x+tile_size]
            upscaled_tile = two_pass_upscale(tile)
            tiles.append((y, x, upscaled_tile))
    
    # Merge tiles with blending
    result = blend_tiles(tiles, target_size=(height*4, width*4))
    
    return result
```

---

### 14. RL Reward Loop

**What is RL Reward Loop?**

Reinforcement Learning system that continuously optimizes motion generation parameters based on quality feedback (VMAF scores, lip-sync accuracy).

**RL Architecture:**

```python
# motion_controller/policy.py
class RLPolicyOptimizer:
    """Q-Learning based parameter optimization"""
    
    def __init__(self):
        self.q_table = {}  # State-action value table
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2  # Exploration rate
        
        # Parameter ranges
        self.param_ranges = {
            "motion_intensity": (0.0, 1.0),
            "motion_speed": (0.5, 1.5),
            "gesture_frequency": (0.0, 2.0)
        }
    
    def select_params(self, state: Dict) -> Dict:
        """Select parameters using epsilon-greedy policy"""
        
        # Exploration: Random parameters
        if random.random() < self.epsilon:
            return self._random_params()
        
        # Exploitation: Best known parameters
        return self._best_params(state)
    
    def update_policy(self, state: Dict, action: Dict, reward: float, next_state: Dict):
        """Update Q-table based on reward"""
        
        # Q-learning update rule
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        
        current_q = self.q_table.get((state_key, action_key), 0.0)
        max_next_q = max([
            self.q_table.get((self._state_to_key(next_state), a), 0.0)
            for a in self._possible_actions()
        ])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state_key, action_key)] = new_q
```

**RL Reward Signal:**

```python
def calculate_reward(video_path: str) -> float:
    """Calculate reward based on video quality metrics"""
    
    reward = 0.0
    
    # Component 1: VMAF quality score (0-100)
    vmaf_score = measure_vmaf(video_path)
    reward += (vmaf_score / 100.0) * 0.5  # Weight: 50%
    
    # Component 2: Lip-sync accuracy (0.0-1.0)
    lipsync_score = measure_lipsync_accuracy(video_path)
    reward += lipsync_score * 0.3  # Weight: 30%
    
    # Component 3: Motion smoothness (0.0-1.0)
    smoothness = measure_temporal_consistency(video_path)
    reward += smoothness * 0.2  # Weight: 20%
    
    return reward  # Range: 0.0 to 1.0
```

**RL Training Loop:**

```python
# Training the RL policy
for episode in range(1000):
    # 1. Get current state (scene context)
    state = {
        "emotion": "joy",
        "scene_type": "conversation",
        "character_count": 2
    }
    
    # 2. Select parameters using policy
    params = rl_optimizer.select_params(state)
    
    # 3. Generate video with these parameters
    video = generate_video(params)
    
    # 4. Calculate reward
    reward = calculate_reward(video)
    
    # 5. Update policy
    next_state = get_next_state()
    rl_optimizer.update_policy(state, params, reward, next_state)
    
    # 6. Log progress
    logger.log_rl_episode(episode, params, reward)
```

**RL Performance Gains:**

| Metric | Before RL | After RL (100 episodes) | After RL (1000 episodes) |
|--------|-----------|------------------------|--------------------------|
| VMAF Score | 72.3 | 78.5 (+8.6%) | 85.7 (+18.5%) |
| Lip-Sync Accuracy | 0.65 | 0.82 (+26.2%) | 0.91 (+40.0%) |
| Temporal Smoothness | 0.70 | 0.81 (+15.7%) | 0.88 (+25.7%) |
| **Overall Quality** | **69.3%** | **80.6%** | **88.2%** |

**Why RL Reward Loop Matters:**

- ✅ **Continuous Improvement:** System learns from every video generated
- ✅ **Automated Optimization:** No manual parameter tuning
- ✅ **Quality Gains:** Measurable improvement in VMAF/lip-sync scores
- ✅ **Adaptation:** Adjusts to different content types automatically


---

## D. Complete Setup Guide

### Environment Setup

**Step 1: Python Environment**

```powershell
# Create virtual environment (if not exists)
python -m venv gurukul-lora-env

# Activate environment
.\gurukul-lora-env\Scripts\Activate.ps1

# Verify Python version
python --version
# Expected: Python 3.10.11
```

**Step 2: Install Dependencies**

```powershell
# Install runtime dependencies
pip install -r requirements-runtime.txt

# Install development dependencies (for testing)
pip install -r requirements-dev.txt

# Verify critical packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: 2.7.1+cu126

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Expected: True
```

**Step 3: Verify GPU**

```powershell
# Check GPU availability
nvidia-smi

# Expected output:
# GPU 0: NVIDIA GeForce RTX 3060 Ti
# CUDA Version: 12.6
```

---

### GPU Requirements

**Minimum Requirements:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU Model** | RTX 3060 (12GB VRAM) | RTX 3060 Ti / 3070 / 3080 |
| **VRAM** | 8GB | 12GB+ |
| **CUDA Compute** | 7.5+ | 8.6+ |
| **CUDA Version** | 11.8+ | 12.1+ |
| **Driver Version** | 520.00+ | 535.00+ |
| **System RAM** | 16GB | 32GB+ |
| **Storage** | 50GB free | 100GB+ SSD |

**Performance Expectations:**

| GPU | 720p Video | 1080p Video | Concurrent Users |
|-----|------------|-------------|------------------|
| RTX 3060 (12GB) | 2-3 min | 4-5 min | 10-15 |
| RTX 3060 Ti | 1.5-2 min | 3-4 min | 20-30 |
| RTX 3070 | 1-1.5 min | 2-3 min | 30-40 |
| RTX 3080 | <1 min | 1.5-2 min | 50+ |

**GPU Detection:**

The system automatically detects GPU capabilities:

```python
# AnimateDiff/adaptive_engine/device_probe.py
from adaptive_engine.device_probe import DeviceProbe

probe = DeviceProbe()
capabilities = probe.get_capabilities()

print(f"GPU: {capabilities['gpu_name']}")
print(f"VRAM: {capabilities['vram_total_gb']} GB")
print(f"CUDA Compute: {capabilities['cuda_compute_capability']}")
```

---

### Required Secrets & Environment Variables

**Critical Environment Variables:**

Create a `.env` file in the project root:

```bash
# ===== GEMINI API (Task 3 - Text Optimization) =====
GEMINI_API_KEY=your_gemini_api_key_here
# Get from: https://makersuite.google.com/app/apikey

# ===== KSML SECURITY (Task 10) =====
KSML_TOKEN=ksml_production
# Provided by Core team, format: ksml_<environment>

# ===== RUNTIME KEY VALIDATION (Task 10) =====
RUNTIME_KEY=<base64_encoded_runtime_key>
# Request from Core/Build server (12-24h validity)
# Format: base64-encoded JSON with Ed25519 signature

WORKER_ID=worker-prod-001
# Unique identifier for this worker instance

RUNTIME_MODE=production
# Options: production (strict) | development (warnings only)

# ===== ENCRYPTION (Task 10) =====
ENCRYPTION_KEY=<32_byte_hex_key>
# Auto-generated on first use if not provided
# Format: 64-character hexadecimal string

# ===== NAS STORAGE (Task 4) =====
NAS_PATH=\\\\192.168.0.94\\shared
# Network storage path for caching

NAS_USERNAME=storage_user
# NAS authentication (optional, if required)

NAS_PASSWORD=storage_pass
# NAS authentication (optional, if required)

# ===== YOTTA CLOUD FALLBACK (Task 4) =====
YOTTA_API_KEY=your_yotta_api_key
# Cloud fallback API key (optional)

YOTTA_ENDPOINT=https://api.yotta.cloud/v1
# Cloud endpoint URL

# ===== MONITORING (Task 8) =====
SENTRY_DSN=https://xxx@sentry.io/xxx
# Error tracking (optional)

REDIS_URL=redis://localhost:6379/0
# Redis for Celery queue (Task 8)

# ===== SUPABASE (Task 8) =====
SUPABASE_URL=https://xxx.supabase.co
# Supabase project URL (optional)

SUPABASE_KEY=your_supabase_anon_key
# Supabase anon key (optional)

# ===== INSIGHTFLOW TELEMETRY (Task 10) =====
INSIGHTFLOW_ENABLED=true
# Enable telemetry tracking

INSIGHTFLOW_API_KEY=your_insightflow_key
# Telemetry API key (optional)
```

**Secret Storage Best Practices:**

1. **NEVER commit `.env` file to Git** (already in `.gitignore`)
2. **Use secret manager in production:**
   - Azure Key Vault
   - AWS Secrets Manager
   - HashiCorp Vault
3. **Rotate secrets regularly:**
   - Runtime keys: Every 12-24 hours (auto-expire)
   - Encryption keys: Every 30 days
   - Signing keys: Every 90 days
   - API keys: Every 180 days

**Loading Secrets in Code:**

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access secrets
gemini_key = os.getenv('GEMINI_API_KEY')
ksml_token = os.getenv('KSML_TOKEN', 'ksml_development')
runtime_key = os.getenv('RUNTIME_KEY')
```

---

### CI/CD Overview

**GitHub Actions Workflows:**

The project includes automated CI/CD pipelines:

**1. Artifact Signing Workflow** (Task 10)

```yaml
# .github/workflows/security-artifact-signing.yml
name: Sign Model Artifacts

on:
  push:
    paths:
      - 'adapters/**/*.pt'
      - 'AnimateDiff/models/**/*.safetensors'

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Sign artifacts
        env:
          SIGNING_KEY: ${{ secrets.SIGNING_PRIVATE_KEY }}
        run: |
          python -m security.artifact_signer sign adapters/gurukul_lora/checkpoint.pt
      
      - name: Upload signatures
        uses: actions/upload-artifact@v3
        with:
          name: signatures
          path: '**/*.sig'
```

**2. Testing Workflow**

```yaml
# .github/workflows/test.yml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest tests/ -v --cov=./ --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**3. Docker Build Workflow** (Task 6)

```yaml
# .github/workflows/docker-build.yml
name: Build Docker Image

on:
  push:
    branches: [main, task_quality_harden_secure]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t ttv-studio:latest .
      
      - name: Push to registry
        run: docker push ttv-studio:latest
```

**Setting Up CI/CD:**

```powershell
# 1. Add GitHub Secrets
# Go to: Settings → Secrets and variables → Actions

# Required secrets:
SIGNING_PRIVATE_KEY      # Ed25519 private key (base64)
GEMINI_API_KEY          # Gemini API key
KSML_TOKEN              # KSML production token
DOCKER_USERNAME         # Docker registry username
DOCKER_PASSWORD         # Docker registry password

# 2. Enable workflows
# Go to: Actions → Enable workflows

# 3. Verify workflows run
git push origin task_quality_harden_secure
# Check: Actions tab for workflow status
```

---

### Complete End-to-End Run

**Method 1: Direct Script (Task 3 Core)**

```powershell
cd AnimateDiff
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1
```

**Method 2: API Endpoint (Task 5)**

```powershell
# Start API server
cd AnimateDiff_API
uvicorn adaptive_api:adaptive_app --host 0.0.0.0 --port 8000

# Make request (in another terminal)
curl -X POST http://localhost:8000/ttv/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lesson_data": {
      "text": "Explain photosynthesis process",
      "title": "Photosynthesis Basics"
    },
    "style": "realistic",
    "quality": "desktop_720p"
  }'
```

**Method 3: Microservice (Task 8)**

```powershell
# Start Redis
redis-server

# Start Celery worker
cd ttv_service
celery -A tasks worker --loglevel=info --pool=solo

# Start FastAPI service
uvicorn main:app --host 0.0.0.0 --port 8001

# Submit job
curl -X POST http://localhost:8001/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum mechanics",
    "style": "realistic",
    "quality": "hd_1080p"
  }'
```

**Method 4: Docker (Task 6)**

```powershell
# Build and run
docker-compose up --build

# Access API
# http://localhost:8000/docs (Swagger UI)
```

---

## E. How to Extend / Modify the System

### 1. Adding New Modules

**Scenario:** You want to add a new video effect module

**Step-by-Step:**

```powershell
# 1. Create module directory
mkdir AnimateDiff/effects

# 2. Create module file
New-Item AnimateDiff/effects/my_effect.py
```

**Module Template:**

```python
# AnimateDiff/effects/my_effect.py

"""
My Effect Module
================
Description: Applies custom effect to video frames
Author: Your Name
Date: 2025-11-27
Task: Task 12 (or custom extension)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

class MyEffect:
    """Custom video effect processor"""
    
    def __init__(self, intensity: float = 0.5):
        """
        Initialize effect processor
        
        Args:
            intensity: Effect intensity (0.0 to 1.0)
        """
        self.intensity = intensity
    
    def apply(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply effect to video frames
        
        Args:
            frames: List of video frames (numpy arrays)
        
        Returns:
            List of processed frames
        """
        processed_frames = []
        
        for frame in frames:
            # Apply your effect here
            processed_frame = self._process_frame(frame)
            processed_frames.append(processed_frame)
        
        return processed_frames
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame"""
        # Your effect logic here
        return frame

# Export main class
__all__ = ['MyEffect']
```

**3. Integrate into Pipeline:**

```python
# AnimateDiff/unified_video_generator.py

# Add import at top
from effects.my_effect import MyEffect

# Add to generate_video function
def generate_video(...):
    # ... existing code ...
    
    # After motion generation, before upscaling
    if apply_custom_effect:
        effect = MyEffect(intensity=0.7)
        frames = effect.apply(frames)
    
    # ... continue pipeline ...
```

**4. Add Tests:**

```python
# tests/effects/test_my_effect.py

import pytest
import numpy as np
from AnimateDiff.effects.my_effect import MyEffect

def test_my_effect_initialization():
    """Test effect initialization"""
    effect = MyEffect(intensity=0.5)
    assert effect.intensity == 0.5

def test_my_effect_applies_to_frames():
    """Test effect application"""
    effect = MyEffect()
    
    # Create test frames
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    
    # Apply effect
    processed = effect.apply(frames)
    
    # Verify output
    assert len(processed) == 5
    assert processed[0].shape == (720, 1280, 3)
```

**5. Add Audit Logging:**

```python
# In your effect module
from audit_logger import AuditLogger

logger = AuditLogger()

def apply(self, frames):
    start_time = time.time()
    
    processed = self._process_frames(frames)
    
    # Log effect application
    logger.log_custom_event(
        event_type="custom_effect_applied",
        metadata={
            "effect_name": "my_effect",
            "intensity": self.intensity,
            "frame_count": len(frames),
            "processing_time_sec": time.time() - start_time
        }
    )
    
    return processed
```

---

### 2. Updating LoRA Models

**Scenario:** Train and deploy new LoRA adapter

**Step 1: Prepare Dataset**

```powershell
# Create dataset directory
mkdir -p adapters/my_lora/dataset

# Organize images
# dataset/
# ├── class_1/
# │   ├── image_001.jpg
# │   ├── image_002.jpg
# ├── class_2/
# │   └── ...
```

**Step 2: Train Adapter**

```python
# adapters/my_lora/train.py

from adapters.adapter_trainer import AdapterTrainer

trainer = AdapterTrainer(
    dataset_path="adapters/my_lora/dataset",
    output_path="adapters/my_lora/checkpoint.pt",
    num_epochs=100,
    learning_rate=1e-4,
    lora_rank=4
)

# Train
trainer.train()

# Validate
trainer.validate()
```

**Step 3: Sign Artifact (Production)**

```powershell
# Sign the checkpoint
python -m security.artifact_signer sign adapters/my_lora/checkpoint.pt

# Verify signature
python -m security.artifact_signer verify adapters/my_lora/checkpoint.pt

# Expected output:
# ✅ Signature valid
# Build ID: build_20251127_001
# Signed at: 2025-11-27T10:30:00Z
```

**Step 4: Integrate into Adapter Manager**

```python
# adapters/adapter_manager.py

class AdapterManager:
    def load_my_lora(self) -> MyLoRA:
        """Load custom LoRA adapter"""
        
        checkpoint_file = Path("adapters/my_lora/checkpoint.pt")
        signature_file = Path(str(checkpoint_file) + '.sig')
        
        # Verify signature (production mode)
        if signature_file.exists():
            signer = ArtifactSigner(public_key_path='security/keys/signing_key.pub')
            is_valid = signer.verify_signature(str(checkpoint_file))
            
            if not is_valid and os.getenv('RUNTIME_MODE') == 'production':
                raise ValueError("Cannot load unsigned model in production")
        
        # Load adapter
        adapter = MyLoRA()
        adapter.load_state_dict(torch.load(checkpoint_file))
        
        return adapter
```

**Step 5: Add to Pipeline**

```python
# AnimateDiff/unified_video_generator.py

from adapters.adapter_manager import AdapterManager

def generate_video(..., use_my_lora=False):
    if use_my_lora:
        manager = AdapterManager()
        adapter = manager.load_my_lora()
        
        # Apply adapter to pipeline
        pipe.load_lora_weights(adapter)
```

---

### 3. Integrating New Telemetry Metrics

**Scenario:** Track custom performance metrics

**Step 1: Extend Audit Logger**

```python
# audit_logger.py

class AuditLogger:
    def log_custom_metric(
        self,
        metric_name: str,
        metric_value: float,
        unit: str,
        metadata: Optional[Dict] = None
    ):
        """
        Log custom telemetry metric
        
        Args:
            metric_name: Name of metric (e.g., "gpu_utilization")
            metric_value: Numeric value
            unit: Unit of measurement (e.g., "percent", "seconds")
            metadata: Additional context
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "custom_metric",
            "metric_name": metric_name,
            "metric_value": metric_value,
            "unit": unit,
            "metadata": metadata or {},
            "ksml_token": os.getenv('KSML_TOKEN', 'ksml_development')
        }
        
        self._write_log_entry(entry)
        
        # Send to InsightFlow
        if self.insightflow_enabled:
            self.insightflow_client.track_metric(
                name=metric_name,
                value=metric_value,
                unit=unit,
                tags=metadata
            )
```

**Step 2: Use in Your Module**

```python
# Your custom module
from audit_logger import AuditLogger

logger = AuditLogger()

def process_video():
    start_time = time.time()
    
    # Your processing logic
    result = do_processing()
    
    processing_time = time.time() - start_time
    
    # Log custom metric
    logger.log_custom_metric(
        metric_name="video_processing_time",
        metric_value=processing_time,
        unit="seconds",
        metadata={
            "video_length_sec": result.duration,
            "resolution": "1080p",
            "gpu_model": "RTX 3060 Ti"
        }
    )
    
    return result
```

**Step 3: Query Metrics**

```python
# Query audit logs for metrics
import json
from pathlib import Path

def get_metric_statistics(metric_name: str, days: int = 7):
    """Get statistics for a custom metric"""
    
    values = []
    
    # Read recent audit logs
    for log_file in Path("logs/audit").glob("audit_*.jsonl"):
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("metric_name") == metric_name:
                    values.append(entry["metric_value"])
    
    # Calculate statistics
    if values:
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }
    
    return None

# Usage
stats = get_metric_statistics("video_processing_time", days=7)
print(f"Average processing time: {stats['mean']:.2f} seconds")
```

---

### 4. Updating Security Lineage

**Scenario:** Modify watermarking process

**Step 1: Update Watermark Module**

```python
# security/watermark.py

def embed_watermark(
    video_path: str,
    build_id: Optional[str] = None,
    output_path: Optional[str] = None,
    custom_metadata: Optional[Dict] = None  # NEW PARAMETER
) -> str:
    """
    Embed watermark with custom metadata
    
    Args:
        video_path: Input video path
        build_id: Build identifier
        output_path: Output video path
        custom_metadata: Additional metadata to embed
    """
    
    # Build watermark data
    watermark_data = {
        "build_id": build_id or generate_build_id(),
        "worker_id": os.getenv('WORKER_ID', 'worker-unknown'),
        "timestamp": datetime.now().isoformat(),
        "framework": "BHIV",
        "creator": "TTV_Studio"
    }
    
    # Add custom metadata
    if custom_metadata:
        watermark_data.update(custom_metadata)
    
    # Embed using FFmpeg
    watermark_b64 = base64.b64encode(
        json.dumps(watermark_data).encode()
    ).decode()
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-metadata', f'BHIV_WATERMARK={watermark_b64}',
        # ... rest of FFmpeg command
    ]
    
    # Execute
    subprocess.run(cmd, check=True)
    
    return output_path
```

**Step 2: Update Detection**

```python
# security/watermark.py

def detect_watermark(video_path: str) -> Optional[Dict]:
    """Detect and decode watermark"""
    
    # Extract metadata
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    
    # Decode watermark
    tags = metadata.get('format', {}).get('tags', {})
    watermark_b64 = tags.get('BHIV_WATERMARK')
    
    if watermark_b64:
        watermark_json = base64.b64decode(watermark_b64).decode()
        watermark_data = json.loads(watermark_json)
        
        # Validate custom fields
        if 'custom_field' in watermark_data:
            # Your validation logic
            pass
        
        return watermark_data
    
    return None
```

**Step 3: Update Audit Logging**

```python
# AnimateDiff/unified_video_generator.py

# Log watermark with custom metadata
audit_logger.log_video_generation(
    prompt=text,
    output_path=final_path,
    ksml_token=ksml_token_data,
    security_metadata={
        "watermark_embedded": True,
        "build_id": build_id,
        "custom_metadata": custom_metadata,  # NEW
        "watermark_version": "2.0"  # Track version changes
    }
)
```

**Step 4: Test Changes**

```python
# tests/security/test_custom_watermark.py

def test_custom_metadata_embedding():
    """Test custom metadata in watermark"""
    
    custom_data = {
        "project_id": "proj_123",
        "user_id": "user_456",
        "license_type": "commercial"
    }
    
    # Embed watermark
    watermarked = embed_watermark(
        "test_video.mp4",
        custom_metadata=custom_data
    )
    
    # Detect watermark
    detected = detect_watermark(watermarked)
    
    # Verify custom fields
    assert detected["project_id"] == "proj_123"
    assert detected["user_id"] == "user_456"
    assert detected["license_type"] == "commercial"
```

---

## F. Best Practices

### 1. Naming Conventions

**File Naming:**

```
# Python modules (lowercase, underscores)
story_context_parser.py       ✅
StoryContextParser.py          ❌
story-context-parser.py        ❌

# Classes (PascalCase)
class StoryContextParser:     ✅
class story_context_parser:    ❌
class storyContextParser:      ❌

# Functions (lowercase, underscores)
def parse_story_context():    ✅
def ParseStoryContext():       ❌
def parseStoryContext():       ❌

# Constants (UPPERCASE)
MAX_VIDEO_LENGTH = 300         ✅
max_video_length = 300         ❌
MaxVideoLength = 300           ❌

# Private methods (leading underscore)
def _internal_helper():        ✅
def internal_helper():         ❌ (if truly private)
```

**Directory Naming:**

```
AnimateDiff/adaptive_engine/   ✅ (lowercase, underscores)
AnimateDiff/AdaptiveEngine/    ❌
AnimateDiff/adaptive-engine/   ❌
```

**Variable Naming:**

```python
# Descriptive names
video_generation_time_sec = 45.2    ✅
vgt = 45.2                          ❌
t = 45.2                            ❌

# Boolean flags (is_, has_, can_, should_)
is_watermark_embedded = True        ✅
watermark_embedded = True           ❌
watermark = True                    ❌

# Collections (plural)
frames = [frame1, frame2]           ✅
frame = [frame1, frame2]            ❌
```

---

### 2. Logging Standards

**Use Audit Logger for Production Events:**

```python
from audit_logger import AuditLogger

logger = AuditLogger()

# ✅ GOOD: Structured logging with context
logger.log_video_generation(
    prompt=user_prompt,
    output_path=video_path,
    ksml_token=ksml_token,
    security_metadata={
        "watermark_embedded": True,
        "build_id": build_id
    }
)

# ❌ BAD: Print statements
print(f"Video generated: {video_path}")

# ❌ BAD: Unstructured logging
logging.info(f"Generated video at {video_path}")
```

**Log Levels:**

```python
import logging

# DEBUG: Detailed diagnostic information
logging.debug(f"Processing frame {frame_idx}/{total_frames}")

# INFO: Confirmation that things are working
logging.info("Video generation started")

# WARNING: Something unexpected, but not an error
logging.warning("GPU utilization at 95%, may slow down")

# ERROR: Serious problem, operation failed
logging.error(f"Failed to load model: {error}")

# CRITICAL: System-level failure
logging.critical("Out of GPU memory, cannot continue")
```

**Structured Logging Example:**

```python
# ✅ GOOD: All context in one log entry
logger.info(
    "video_generation_complete",
    extra={
        "duration_sec": 45.2,
        "resolution": "1080p",
        "file_size_mb": 125.3,
        "gpu_model": "RTX 3060 Ti",
        "build_id": build_id
    }
)

# ❌ BAD: Multiple log lines
logger.info(f"Video complete in {duration} seconds")
logger.info(f"Resolution: {resolution}")
logger.info(f"File size: {file_size} MB")
```

---

### 3. Commit Structuring

**Follow Conventional Commits:**

```bash
# Format: <type>(<scope>): <subject>

# Types:
feat     # New feature
fix      # Bug fix
docs     # Documentation only
style    # Code style (formatting, no logic change)
refactor # Code restructuring (no feature/fix)
perf     # Performance improvement
test     # Adding/updating tests
chore    # Build/tooling changes

# Examples:

✅ feat(task11): add emotion controller with 6 core emotions
✅ fix(security): resolve watermark stripping during audio restoration
✅ docs(handover): complete setup guide with GPU requirements
✅ refactor(adapters): consolidate LoRA loading logic
✅ perf(interpolation): optimize RIFE frame processing (2x speedup)
✅ test(task10): add watermark detection integration tests
✅ chore(deps): update PyTorch to 2.7.1+cu126

❌ "updated files"
❌ "fixes"
❌ "Task 11 work"
❌ "changes"
```

**Commit Body (Optional but Recommended):**

```bash
git commit -m "fix(security): resolve watermark stripping during audio restoration

Bug #2 from ERRORS_AND_BUGS_LOG.md - FFmpeg audio restoration was stripping
all metadata including watermarks. Added -map_metadata 0 flag to preserve
metadata through the audio restoration step.

Fixes: Bug #2
Related: Task 10 security implementation
Tested: 10 videos, 100% watermark detection rate
"
```

**Pull Request Guidelines:**

```markdown
## PR Title
fix(security): resolve 5 cascading watermark bugs

## Description
Fixes all 5 watermark bugs discovered during Task 10 validation:
- Bug #1: LSB watermarking not working (switched to FFmpeg metadata)
- Bug #2: Audio restoration stripping metadata
- Bug #3: -map_metadata ignoring custom tags
- Bug #4: -c copy stripping MP4 metadata
- Bug #5: H.264 encoding stripping tags

## Testing
- [x] All 5 security integration tests passing
- [x] 100% watermark detection on 10+ generated videos
- [x] Manual validation with detect_provenance.py

## Related Issues
- Closes #42 (watermark detection failing)
- Related to Task 10 requirements

## Checklist
- [x] Code follows project naming conventions
- [x] Added tests for all bug fixes
- [x] Updated ERRORS_AND_BUGS_LOG.md
- [x] Signed commits with GPG key
```

---

### 4. Testing Workflow

**Test Organization:**

```
tests/
├── task10/                    # Task-specific tests
│   ├── test_task10_integration.py
│   ├── test_watermarking.py
│   └── test_artifact_signing.py
├── components/                # Component tests
│   ├── audio/
│   ├── interpolation/
│   └── upscaler/
├── integration/               # End-to-end tests
│   └── test_end_to_end.py
└── conftest.py                # Shared fixtures
```

**Writing Tests:**

```python
# tests/task11/test_emotion_controller.py

import pytest
from AnimateDiff.adaptive_engine.emotion_controller import EmotionController

@pytest.fixture
def emotion_controller():
    """Shared fixture for emotion controller"""
    return EmotionController()

def test_emotion_controller_initialization(emotion_controller):
    """Test emotion controller initializes correctly"""
    assert emotion_controller is not None
    assert len(emotion_controller.emotions) == 6

def test_emotion_detection_from_text(emotion_controller):
    """Test emotion detection from text"""
    text = "I am so happy today!"
    emotion = emotion_controller.detect_emotion(text)
    
    assert emotion == "joy"
    assert emotion_controller.confidence > 0.7

def test_motion_emotion_coupling(emotion_controller):
    """Test motion parameters adjust based on emotion"""
    # Joy should increase motion intensity
    joy_params = emotion_controller.get_motion_params("joy")
    assert joy_params["intensity"] > 0.5
    
    # Sadness should decrease motion intensity
    sad_params = emotion_controller.get_motion_params("sadness")
    assert sad_params["intensity"] < 0.5

@pytest.mark.slow
def test_emotion_controller_end_to_end(emotion_controller):
    """Full emotion controller workflow test"""
    story = "She was happy at first, but then became sad."
    
    # Analyze story
    emotions = emotion_controller.analyze_story(story)
    
    # Should detect emotion transition
    assert len(emotions) == 2
    assert emotions[0]["emotion"] == "joy"
    assert emotions[1]["emotion"] == "sadness"
```

**Running Tests:**

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/task11/test_emotion_controller.py

# Run with coverage
pytest --cov=AnimateDiff --cov-report=html

# Run only fast tests (skip @pytest.mark.slow)
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test
pytest tests/task11/test_emotion_controller.py::test_emotion_detection_from_text
```

**Test Coverage Standards:**

| Module Type | Minimum Coverage | Recommended |
|-------------|------------------|-------------|
| Core Engine (Task 3) | 80% | 90%+ |
| Security (Task 10) | 95% | 100% |
| Intelligence (Task 11) | 70% | 85% |
| API Endpoints (Task 5) | 80% | 90% |
| Utilities | 60% | 75% |

**Integration Test Example:**

```python
# tests/integration/test_full_pipeline.py

def test_complete_video_generation_pipeline():
    """Test full pipeline from JSON to video"""
    
    # Input
    lesson_data = {
        "text": "Explain gravity",
        "title": "Understanding Gravity"
    }
    
    # Generate video
    result = generate_lesson_video(
        lesson_data=lesson_data,
        style="realistic",
        quality="720p"
    )
    
    # Verify output exists
    assert result["video_path"].exists()
    assert result["subtitle_path"].exists()
    
    # Verify watermark
    watermark = detect_watermark(result["video_path"])
    assert watermark is not None
    assert watermark["build_id"] is not None
    
    # Verify quality
    assert result["resolution"] == "1280x720"
    assert result["fps"] == 24
    assert result["duration_sec"] > 0
    
    # Verify audit log
    audit_entry = get_last_audit_entry()
    assert audit_entry["event_type"] == "video_generation"
    assert audit_entry["ksml_token"] is not None
```

---

## 🚀 Quick Start Guide

### Prerequisites

```powershell
# Activate Python environment
.\gurukul-lora-env\Scripts\Activate.ps1

# Verify Python version
python --version  # Should be 3.10.11

# Verify GPU
nvidia-smi  # Should show RTX 3060 Ti or better
```

### Generate Your First Video

```powershell
# Navigate to AnimateDiff directory
cd AnimateDiff

# Generate video (Task 3 core engine)
python generate_lesson_video_safe.py lesson_comprehensive_1.json realistic 1
```

**What Happens:**
1. **Story Analysis (Task 11):** Analyzes full story, resolves character genders, builds scene graph
2. **Text Optimization (Task 3):** Gemini API enhances prompts
3. **Motion Generation (Task 3):** AnimateDiff creates animated clips
4. **Audio Integration (Task 3):** Multi-voice TTS with subtitle sync
5. **Smart Extension (Task 11):** SlowMo + Freeze for perfect audio-video sync
6. **Security (Task 10):** Dual watermarking + fingerprinting
7. **Audit Logging (Task 11):** 26 intelligence metrics tracked

**Output Files:**
```
AnimateDiff/storage/2025-11-26/
├── Lesson_Title_realistic_complete.mp4        # Final video (H.264, 8000k)
├── Lesson_Title_realistic_complete.srt        # Subtitles
├── Lesson_Title_realistic_complete_fingerprint.json  # SHA256 + BLAKE2b
└── [intermediate files]

logs/audit/audit_20251126.jsonl               # Audit log entry
```

### Check Intelligence Metrics

```powershell
# View audit log (JSONL format)
Get-Content logs/audit/audit_20251126.jsonl | Select-Object -Last 1 | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**Metrics Tracked (26 total):**
- Story analysis: character count, gender resolved, text condensation %
- Scene graph: total scenes, entities, transitions
- Narrative: story beats, character arcs, tension levels
- Emotion: emotion changes, motion intensity
- Extension: clips extended, method used
- Quality: audio-video sync, duration, FPS, bitrate

---

## 📞 Getting Help

**If You See Errors:**

1. **Check Task README files first:** `Documentation/Tasks/Task-{N}-README.md`
2. **Review this handover:** Sections match actual implementation
3. **Check audit logs:** `logs/audit/audit_YYYYMMDD.jsonl`
4. **Verify file locations:** Use corrected paths from this document

---

**Document Status:** ✅ Production Ready - Restructured A-F Format (November 27, 2025)  
**Based on:** Actual implementation through Task 11 (November 26, 2025)