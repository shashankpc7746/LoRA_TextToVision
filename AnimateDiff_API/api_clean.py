#!/usr/bin/env python3
"""
Clean API for AnimateDiff - No heavy imports
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import subprocess
import sys
from pathlib import Path

app = FastAPI(title="AnimateDiff Clean API", description="Clean API without heavy imports")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class VideoRequest(BaseModel):
    lesson_filename: str = "lesson_man_carpenter.json"
    style: str = "realistic"
    speech_rate: float = 1.0

# Production team's format
class ProductionVideoRequest(BaseModel):
    explanation: Optional[str] = None
    title: str = "Generated Video"
    level: str = "Advanced"
    duration: Optional[str] = None
    tts_enabled: bool = True
    scenes: Optional[List[Dict[str, Any]]] = None
    prompts: Optional[List[str]] = None
    text: Optional[str] = None
    video_style: str = "realistic"
    style_modifiers: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AnimateDiff Clean API is running",
        "status": "healthy",
        "version": "clean-v1.0",
        "endpoints": {
            "health": "/health",
            "generate_video": "/generate-video",
            "proxy_vision": "/proxy/vision (PRODUCTION TEAM FORMAT)",
            "generate_video_production": "/generate-video-production",
            "debug": "/debug-request",
            "test": "/test"
        },
        "production_team_info": {
            "backend_proxy_url": "http://localhost:8001/proxy/vision",
            "ngrok_url": "https://5d805a066cfd.ngrok-free.app/generate-video",
            "local_api_url": "http://localhost:8002/proxy/vision",
            "supported_format": "explanation + video_style fields"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Clean API is working",
        "animatediff_path": str(Path(__file__).parent.parent / "AnimateDiff")
    }

# Debug endpoint
@app.post("/debug-request")
async def debug_request(request: Request):
    """Debug endpoint to see what JSON is being sent"""
    try:
        body = await request.body()
        json_data = await request.json()
        return {
            "success": True,
            "received_json": json_data,
            "content_type": request.headers.get("content-type"),
            "body_size": len(body),
            "message": "Request received successfully"
        }
    except Exception as e:
        return {
            "error": str(e),
            "body": body.decode() if 'body' in locals() else "Could not read body"
        }

# Video generation endpoint (original format)
@app.post("/generate-video")
async def generate_video(request: Request):
    """
    Flexible video generation endpoint - handles both formats
    """
    try:
        data = await request.json()

        # Check if it's the production team's format (has explanation field)
        if "explanation" in data or "text" in data:
            # Use the same logic as /proxy/vision
            explanation = data.get("explanation", "") or data.get("text", "")
            if not explanation:
                raise HTTPException(status_code=400, detail="No explanation or text provided")

            style = data.get("video_style", "realistic")
            if style not in ["realistic", "anime", "artistic"]:
                style = "realistic"

            # Create lesson content
            sentences = [s.strip() for s in explanation.split('.') if s.strip()]

            # Create prompts for each sentence
            prompts = []
            for sentence in sentences:
                if sentence:
                    prompts.append(f"{sentence}, spiritual journey, educational content, meditation theme")

            lesson_content = {
                "title": data.get("title", "Generated Video"),
                "level": data.get("level", "Advanced"),
                "duration": f"{len(sentences) * 6}-{len(sentences) * 8} seconds",
                "tts_enabled": data.get("tts_enabled", True),
                "scenes": [],
                "prompts": prompts,
                "text": explanation,
                "metadata": {
                    "theme": "spiritual journey and meditation",
                    "setting": "educational content",
                    "character": "spiritual seeker",
                    "lesson": "meditation and wisdom",
                    "mood": "profound and transformative",
                    "visual_style": "cinematic spiritual journey",
                    "duration_target": f"{len(sentences) * 7} seconds",
                    "created": "2025-08-04",
                    "story_type": "educational_lesson",
                    "production_generated": True
                }
            }

            for sentence in sentences:
                if sentence:
                    lesson_content["scenes"].append({
                        "text": sentence,  # Use "text" not "description" to match your format
                        "duration": max(4.0, min(8.0, len(sentence) * 0.08))
                    })

            if data.get("tts_enabled", True):
                lesson_content["tts"] = True

            # Create lesson file
            import time
            timestamp = int(time.time())
            lesson_filename = f"production_lesson_{timestamp}.json"

            animatediff_path = Path(__file__).parent.parent / "AnimateDiff"
            lessons_path = animatediff_path / "lessons"
            lesson_file_path = lessons_path / lesson_filename

            lessons_path.mkdir(exist_ok=True)

            with open(lesson_file_path, 'w', encoding='utf-8') as f:
                json.dump(lesson_content, f, indent=2, ensure_ascii=False)

            print(f"📝 Created lesson file: {lesson_filename}")
            print(f"🎬 Style: {style}")

        else:
            # Original format with lesson_filename
            lesson_filename = data.get("lesson_filename", "lesson_man_carpenter.json")
            style = data.get("style", "realistic")
            speech_rate = data.get("speech_rate", 1.0)

            animatediff_path = Path(__file__).parent.parent / "AnimateDiff"
            lesson_path = animatediff_path / "lessons" / lesson_filename
            if not lesson_path.exists():
                raise HTTPException(status_code=404, detail=f"Lesson file not found: {lesson_filename}")

        # Start video generation
        cmd = [
            sys.executable,
            "generate_lesson_video_safe.py",
            lesson_filename,
            style,
            str(data.get("speech_rate", 1))
        ]

        print(f"🎬 Starting video generation process...")
        print(f"📁 Working directory: {animatediff_path}")
        print(f"⚡ Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            cwd=str(animatediff_path),
            stdout=None,  # Let output go to terminal
            stderr=None,  # Let errors go to terminal
            text=True
        )

        return {
            "success": True,
            "message": "Video generation started successfully",
            "lesson_filename": lesson_filename,
            "style": style,
            "process_id": process.pid,
            "status": "processing",
            "note": "Video generation is running in background. Check storage folder for output."
        }

    except Exception as e:
        print(f"❌ Error in /generate-video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

# Production team's endpoint - handles their complex format
@app.post("/proxy/vision")
async def proxy_vision(request: Request):
    """
    Handle production team's complex video format and convert to lesson file
    """
    try:
        data = await request.json()

        # Extract the explanation text
        explanation = data.get("explanation", "")
        if not explanation:
            explanation = data.get("text", "")

        if not explanation:
            raise HTTPException(status_code=400, detail="No explanation or text provided")

        # Get style from video_style or default
        style = data.get("video_style", "realistic")
        if style not in ["realistic", "anime", "artistic"]:
            style = "realistic"

        # Create lesson content from explanation
        sentences = [s.strip() for s in explanation.split('.') if s.strip()]

        # Create prompts for each sentence
        prompts = []
        for sentence in sentences:
            if sentence:
                prompts.append(f"{sentence}, spiritual journey, educational content, meditation theme")

        # Create lesson file format (matching your current format)
        lesson_content = {
            "title": data.get("title", "Generated Video"),
            "level": data.get("level", "Advanced"),
            "duration": f"{len(sentences) * 6}-{len(sentences) * 8} seconds",
            "tts_enabled": data.get("tts_enabled", True),
            "scenes": [],
            "prompts": prompts,
            "text": explanation,
            "metadata": {
                "theme": "spiritual journey and meditation",
                "setting": "educational content",
                "character": "spiritual seeker",
                "lesson": "meditation and wisdom",
                "mood": "profound and transformative",
                "visual_style": "cinematic spiritual journey",
                "duration_target": f"{len(sentences) * 7} seconds",
                "created": "2025-08-04",
                "story_type": "educational_lesson",
                "production_generated": True
            }
        }

        # Add scenes with correct format (text, not description)
        for i, sentence in enumerate(sentences):
            if sentence:
                lesson_content["scenes"].append({
                    "text": sentence,  # Use "text" not "description" to match your format
                    "duration": max(4.0, min(8.0, len(sentence) * 0.08))
                })

        # Add TTS if specified
        if data.get("tts_enabled", True):
            lesson_content["tts"] = True

        # Create temporary lesson file
        import time
        timestamp = int(time.time())
        lesson_filename = f"production_lesson_{timestamp}.json"

        # Path to lessons directory
        animatediff_path = Path(__file__).parent.parent / "AnimateDiff"
        lessons_path = animatediff_path / "lessons"
        lesson_file_path = lessons_path / lesson_filename

        # Ensure lessons directory exists
        lessons_path.mkdir(exist_ok=True)

        # Write lesson file
        with open(lesson_file_path, 'w', encoding='utf-8') as f:
            json.dump(lesson_content, f, indent=2, ensure_ascii=False)

        print(f"📝 Created lesson file: {lesson_filename}")
        print(f"🎬 Style: {style}")
        print(f"📄 Content: {len(sentences)} sentences")

        # Start video generation
        cmd = [
            sys.executable,
            "generate_lesson_video_safe.py",
            lesson_filename,
            style,
            "1"
        ]

        print(f"🎬 Starting video generation: {' '.join(cmd)}")

        # Execute video generation in subprocess with visible output
        print(f"🎬 Starting video generation process...")
        print(f"📁 Working directory: {animatediff_path}")
        print(f"⚡ Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            cwd=str(animatediff_path),
            stdout=None,  # Let output go to terminal
            stderr=None,  # Let errors go to terminal
            text=True
        )

        return {
            "success": True,
            "message": "Video generation started successfully",
            "lesson_filename": lesson_filename,
            "style": style,
            "process_id": process.pid,
            "status": "processing",
            "scenes_count": len(sentences),
            "estimated_duration": f"{sum(max(4.0, min(8.0, len(s) * 0.08)) for s in sentences):.1f} seconds",
            "note": "Video generation is running in background. Check storage folder for output."
        }

    except Exception as e:
        print(f"❌ Error in proxy/vision: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

# Alternative endpoint for ngrok direct access
@app.post("/generate-video-production")
async def generate_video_production(production_request: ProductionVideoRequest):
    """
    Handle production team's format with Pydantic validation
    """
    try:
        # Use explanation or text
        explanation = production_request.explanation or production_request.text
        if not explanation:
            raise HTTPException(status_code=400, detail="No explanation or text provided")

        # Get style
        style = production_request.video_style or "realistic"
        if style not in ["realistic", "anime", "artistic"]:
            style = "realistic"

        # Create lesson content
        sentences = [s.strip() for s in explanation.split('.') if s.strip()]

        lesson_content = {
            "title": production_request.title,
            "level": production_request.level,
            "text": explanation,
            "scenes": []
        }

        # Add scenes
        for sentence in sentences:
            if sentence:
                lesson_content["scenes"].append({
                    "description": sentence,
                    "duration": max(4.0, min(8.0, len(sentence) * 0.08))
                })

        if production_request.tts_enabled:
            lesson_content["tts"] = True

        # Create lesson file
        import time
        timestamp = int(time.time())
        lesson_filename = f"production_lesson_{timestamp}.json"

        animatediff_path = Path(__file__).parent.parent / "AnimateDiff"
        lessons_path = animatediff_path / "lessons"
        lesson_file_path = lessons_path / lesson_filename

        lessons_path.mkdir(exist_ok=True)

        with open(lesson_file_path, 'w', encoding='utf-8') as f:
            json.dump(lesson_content, f, indent=2, ensure_ascii=False)

        # Start video generation
        cmd = [sys.executable, "generate_lesson_video_safe.py", lesson_filename, style, "1"]

        process = subprocess.Popen(
            cmd,
            cwd=str(animatediff_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return {
            "success": True,
            "message": "Video generation started successfully",
            "lesson_filename": lesson_filename,
            "style": style,
            "process_id": process.pid,
            "status": "processing"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

# Simple test endpoint
@app.post("/test")
async def test_endpoint(request: Request):
    """Simple test endpoint"""
    try:
        data = await request.json()
        return {
            "success": True,
            "message": "Test endpoint working",
            "received": data,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Clean AnimateDiff API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
