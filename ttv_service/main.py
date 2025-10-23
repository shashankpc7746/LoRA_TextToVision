#!/usr/bin/env python3
"""
TTV (Text-to-Video) Service - Task 8 Production Integration
FastAPI service wrapper around LoRA_TextToVision system
"""

import os
import sys
import time
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our existing TTV system
from AnimateDiff.unified_video_generator import UnifiedVideoGenerator
from AnimateDiff.orchestrator import get_orchestrator
from AnimateDiff.generate_lesson_video_safe import generate_lesson_video_safe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# Pydantic Models for API
# ================================

class TTVGenerateRequest(BaseModel):
    """Request model for TTV generation matching BHIV system schema"""
    script: str = Field(..., description="Text script for video generation")
    style: Optional[str] = Field(default="realistic", description="Video style: realistic, anime, artistic")
    quality: Optional[str] = Field(default="balanced", description="Quality preset: ultra_fast, fast, balanced, quality, ultra_quality")
    duration: Optional[int] = Field(default=None, description="Target duration in seconds")
    voice_settings: Optional[Dict[str, Any]] = Field(default=None, description="Voice and audio settings")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    user_id: Optional[str] = Field(default=None, description="User ID for tracking and security")
    priority: Optional[str] = Field(default="normal", description="Job priority: low, normal, high, urgent")

class TTVStoryboardScene(BaseModel):
    """Individual scene in the storyboard"""
    scene_id: str
    text: str
    duration: float
    visual_description: str
    audio_cues: Optional[List[str]] = []
    timestamp_start: float
    timestamp_end: float

class TTVStoryboard(BaseModel):
    """Storyboard structure matching BHIV expectations"""
    total_duration: float
    total_scenes: int
    scenes: List[TTVStoryboardScene]
    version: str = "1.0"
    generation_method: str = "lora_animatediff"
    metadata: Optional[Dict[str, Any]] = None

class TTVGenerateResponse(BaseModel):
    """Response model for TTV generation"""
    job_id: str
    status: str  # queued, processing, completed, failed
    message: str
    storyboard: Optional[TTVStoryboard] = None
    video_url: Optional[str] = None
    preview_url: Optional[str] = None
    duration: Optional[float] = None
    fps: Optional[int] = None
    scenes: Optional[List[Dict[str, Any]]] = None
    estimated_completion: Optional[str] = None
    created_at: str
    
class TTVStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: float  # 0.0 to 1.0
    current_step: str
    estimated_remaining: Optional[int] = None  # seconds
    video_url: Optional[str] = None
    storyboard: Optional[TTVStoryboard] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]] = None

class TTVHealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "TTV Service"
    version: str = "1.0.0"
    gpu_available: bool
    queue_status: Dict[str, Any]
    system_metrics: Dict[str, Any]
    timestamp: str

class ContentModerationRequest(BaseModel):
    """Content moderation request"""
    content: str
    content_type: str = "text"
    user_id: Optional[str] = None

class ContentModerationResponse(BaseModel):
    """Content moderation response"""
    allowed: bool
    confidence: float
    reasons: List[str] = []
    category: Optional[str] = None
    message: str

# ================================
# Job Queue and Status Management
# ================================

class TTVJobManager:
    """Simple in-memory job management (will be replaced with Celery/RQ)"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.orchestrator = get_orchestrator()
        
    def create_job(self, request: TTVGenerateRequest) -> str:
        """Create a new video generation job"""
        job_id = f"ttv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_step": "initializing",
            "request": request.dict(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "video_url": None,
            "storyboard": None,
            "error_message": None,
            "metadata": {}
        }
        
        logger.info(f"Created TTV job: {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, **updates):
        """Update job status"""
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
    
    async def process_job(self, job_id: str):
        """Process a video generation job asynchronously"""
        try:
            job = self.jobs[job_id]
            request = TTVGenerateRequest(**job["request"])
            
            # Update status to processing
            self.update_job_status(job_id, 
                status="processing", 
                progress=0.1, 
                current_step="preparing_generation"
            )
            
            # Create temporary lesson file for our existing system
            lesson_data = self._create_lesson_from_script(request.script, request.style)
            
            self.update_job_status(job_id, 
                progress=0.3, 
                current_step="generating_video"
            )
            
            # Use our existing video generation system
            result = await self._generate_video_async(lesson_data, request)
            
            if result["success"]:
                # Create storyboard from result
                storyboard = self._create_storyboard_from_result(result, request)
                
                # Update job with completion
                self.update_job_status(job_id,
                    status="completed",
                    progress=1.0,
                    current_step="completed",
                    video_url=result.get("video_url"),
                    storyboard=storyboard
                )
                
                # Emit completion event (Task 5 requirement)
                await self._emit_completion_event(job_id, result)
                
            else:
                self.update_job_status(job_id,
                    status="failed",
                    progress=0.0,
                    current_step="failed",
                    error_message=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.update_job_status(job_id,
                status="failed",
                progress=0.0,
                current_step="failed",
                error_message=str(e)
            )
    
    def _create_lesson_from_script(self, script: str, style: str = "realistic") -> Dict[str, Any]:
        """Convert script to lesson format for our existing system"""
        # Split script into scenes
        sentences = [s.strip() for s in script.split('.') if s.strip()]
        
        lesson_data = {
            "title": "TTV Generated Video",
            "style": style,
            "segments": []
        }
        
        for i, sentence in enumerate(sentences[:10]):  # Limit to 10 scenes
            lesson_data["segments"].append({
                "id": i + 1,
                "text": sentence,
                "duration": 3.0,
                "audio_cues": [],
                "visual_elements": [f"Scene {i+1}: {sentence[:50]}..."]
            })
        
        return lesson_data
    
    async def _generate_video_async(self, lesson_data: Dict[str, Any], request: TTVGenerateRequest) -> Dict[str, Any]:
        """Generate video using our existing system"""
        try:
            # Create temporary lesson file
            temp_lesson_file = f"temp_lesson_{int(time.time())}.json"
            lesson_path = f"AnimateDiff/lessons/{temp_lesson_file}"
            
            with open(lesson_path, 'w') as f:
                json.dump(lesson_data, f, indent=2)
            
            # Use our existing generation function
            result = await asyncio.to_thread(
                generate_lesson_video_safe,
                temp_lesson_file,
                lesson_data.get("style", "realistic"),
                1  # speech_rate
            )
            
            # Clean up temp file
            if os.path.exists(lesson_path):
                os.remove(lesson_path)
            
            return {
                "success": True,
                "video_url": f"/videos/{result.get('video_filename', 'generated.mp4')}",
                "duration": result.get("duration", 30.0),
                "fps": 12,
                "scenes": len(lesson_data["segments"]),
                "style": lesson_data.get("style", "realistic")
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_storyboard_from_result(self, result: Dict[str, Any], request: TTVGenerateRequest) -> TTVStoryboard:
        """Create storyboard object from generation result"""
        scenes = []
        total_duration = result.get("duration", 30.0)
        scene_count = result.get("scenes", 3)
        scene_duration = total_duration / max(scene_count, 1)
        
        # Split script into scenes for storyboard
        sentences = [s.strip() for s in request.script.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences[:scene_count]):
            scene = TTVStoryboardScene(
                scene_id=f"scene_{i+1}",
                text=sentence,
                duration=scene_duration,
                visual_description=f"Visual representation of: {sentence[:100]}...",
                timestamp_start=i * scene_duration,
                timestamp_end=(i + 1) * scene_duration,
                audio_cues=["background_music", "narration"]
            )
            scenes.append(scene)
        
        return TTVStoryboard(
            total_duration=total_duration,
            total_scenes=len(scenes),
            scenes=scenes,
            generation_method="lora_animatediff_unified",
            metadata={
                "style": request.style,
                "quality": request.quality,
                "fps": result.get("fps", 12)
            }
        )
    
    async def _emit_completion_event(self, job_id: str, result: Dict[str, Any]):
        """Emit completion event for backend notification"""
        event_data = {
            "event_type": "ttv_generation_completed",
            "job_id": job_id,
            "video_url": result.get("video_url"),
            "duration": result.get("duration"),
            "timestamp": datetime.now().isoformat(),
            "service": "ttv_service"
        }
        
        # Log event (will be enhanced with actual event system)
        logger.info(f"TTV completion event: {json.dumps(event_data)}")
        
        # TODO: Integrate with actual event system to notify Ashmit's backend

# ================================
# Content Moderation
# ================================

class ContentModerator:
    """Simple content moderation (stub implementation)"""
    
    FORBIDDEN_KEYWORDS = [
        "violence", "hate", "explicit", "illegal", "harmful"
    ]
    
    def moderate_content(self, content: str, content_type: str = "text") -> ContentModerationResponse:
        """Basic content moderation"""
        content_lower = content.lower()
        
        # Check for forbidden keywords
        found_keywords = [kw for kw in self.FORBIDDEN_KEYWORDS if kw in content_lower]
        
        if found_keywords:
            return ContentModerationResponse(
                allowed=False,
                confidence=0.9,
                reasons=[f"Contains forbidden content: {', '.join(found_keywords)}"],
                category="inappropriate_content",
                message="Content violates community guidelines"
            )
        
        # Check content length
        if len(content) > 10000:  # 10k character limit
            return ContentModerationResponse(
                allowed=False,
                confidence=1.0,
                reasons=["Content too long"],
                category="content_length",
                message="Content exceeds maximum length limit"
            )
        
        return ContentModerationResponse(
            allowed=True,
            confidence=0.95,
            reasons=["Content passes basic moderation"],
            message="Content approved for processing"
        )

# ================================
# FastAPI Application
# ================================

app = FastAPI(
    title="TTV Service - Text to Video Generation",
    description="Production-ready Text-to-Video service for BHIV integration",
    version="1.0.0",
    docs_url="/ttv/docs",
    redoc_url="/ttv/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
job_manager = TTVJobManager()
content_moderator = ContentModerator()

# ================================
# API Endpoints
# ================================

@app.post("/ttv/generate", response_model=TTVGenerateResponse)
async def generate_video(
    request: TTVGenerateRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Generate video from script - Main TTV endpoint
    Accepts script and returns storyboard JSON and video_url
    """
    try:
        # Content moderation
        moderation_result = content_moderator.moderate_content(request.script)
        if not moderation_result.allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Content moderation failed: {moderation_result.message}"
            )
        
        # Create job
        job_id = job_manager.create_job(request)
        
        # Start background processing
        background_tasks.add_task(job_manager.process_job, job_id)
        
        # Create initial storyboard for immediate response
        initial_storyboard = TTVStoryboard(
            total_duration=30.0,  # Estimate
            total_scenes=len([s for s in request.script.split('.') if s.strip()]),
            scenes=[],  # Will be populated when processing completes
            generation_method="lora_animatediff_unified"
        )
        
        return TTVGenerateResponse(
            job_id=job_id,
            status="queued",
            message="Video generation started. Use /ttv/status/{job_id} to check progress.",
            storyboard=initial_storyboard,
            estimated_completion=datetime.now().isoformat(),
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Video generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ttv/status/{job_id}", response_model=TTVStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a video generation job"""
    job = job_manager.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return TTVStatusResponse(**job)

@app.get("/ttv/health", response_model=TTVHealthResponse)
async def health_check():
    """Health check endpoint for service monitoring"""
    try:
        import psutil
        import torch
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # Get system metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "active_jobs": len([j for j in job_manager.jobs.values() if j["status"] in ["queued", "processing"]])
        }
        
        # Queue status
        queue_status = {
            "total_jobs": len(job_manager.jobs),
            "queued": len([j for j in job_manager.jobs.values() if j["status"] == "queued"]),
            "processing": len([j for j in job_manager.jobs.values() if j["status"] == "processing"]),
            "completed": len([j for j in job_manager.jobs.values() if j["status"] == "completed"]),
            "failed": len([j for j in job_manager.jobs.values() if j["status"] == "failed"])
        }
        
        return TTVHealthResponse(
            status="healthy",
            gpu_available=gpu_available,
            queue_status=queue_status,
            system_metrics=system_metrics,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return TTVHealthResponse(
            status="unhealthy",
            gpu_available=False,
            queue_status={"error": str(e)},
            system_metrics={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )

@app.post("/ttv/moderate", response_model=ContentModerationResponse)
async def moderate_content(request: ContentModerationRequest):
    """Content moderation endpoint"""
    return content_moderator.moderate_content(request.content, request.content_type)

@app.get("/ttv/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """List recent jobs (for debugging/monitoring)"""
    jobs = list(job_manager.jobs.values())
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "jobs": jobs[:limit],
        "total": len(jobs),
        "timestamp": datetime.now().isoformat()
    }

# ================================
# Error Handlers
# ================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ================================
# Startup/Shutdown Events
# ================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("TTV Service starting up...")
    logger.info("Initializing LoRA_TextToVision system...")
    
    # Test our existing system
    try:
        orchestrator = get_orchestrator()
        logger.info("✅ TTV system initialized successfully")
    except Exception as e:
        logger.error(f"❌ TTV system initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("TTV Service shutting down...")

# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,  # Different port from main BHIV system (9000)
        reload=True,
        log_level="info"
    )