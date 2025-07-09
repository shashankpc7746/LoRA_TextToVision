# main.py
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from animate_generator import generate_video
from dotenv import load_dotenv
import os
import requests
import json
from datetime import datetime
from typing import Optional

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Configuration for main system video transfer
MAIN_SYSTEM_URL = os.getenv("MAIN_SYSTEM_URL", "http://192.168.0.121:8001")  # Adjust this to your main system's IP
MAIN_SYSTEM_ENDPOINT = f"{MAIN_SYSTEM_URL}/receive-video"
ENABLE_VIDEO_TRANSFER = os.getenv("ENABLE_VIDEO_TRANSFER", "true").lower() == "true"

app = FastAPI(title="AnimateDiff Video API", description="Text-to-Video Generation API")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    seed: int = 333
    guidance_scale: float = 15
    steps: int = 25
    num_frames: int = 32
    fps: int = 8
    subject: str = "AnimateDiff Video"
    topic: str = "AI Generated Video"

class ManualVideoTransfer(BaseModel):
    video_path: str
    subject: str = "Manual Upload"
    topic: str = "Manual Upload"
    prompt: str = "Manually uploaded video"

async def send_video_to_main_system(
    video_file_path: str,
    subject: str,
    topic: str,
    prompt: str,
    metadata: Optional[dict] = None
):
    """
    POST endpoint function to send generated video to main system
    Call this after video generation is complete
    """
    # Check if video transfer is enabled
    if not ENABLE_VIDEO_TRANSFER:
        print("ℹ️ Video transfer is disabled")
        return {"status": "disabled", "message": "Video transfer is disabled"}

    try:
        # Prepare metadata
        video_metadata = {
            "subject": subject,
            "topic": topic,
            "prompt": prompt,
            "generated_at": datetime.now().isoformat(),
            "file_size": os.path.getsize(video_file_path),
            "system_info": "AnimateDiff_192.168.0.121:8501",
            **(metadata or {})
        }

        print(f"🎬 Sending video to main system: {MAIN_SYSTEM_ENDPOINT}")
        print(f"🎬 Video file: {video_file_path}")
        print(f"🎬 Metadata: {video_metadata}")

        # Prepare the multipart form data
        with open(video_file_path, 'rb') as video_file:
            files = {
                'video': ('generated_video.mp4', video_file, 'video/mp4')
            }

            data = {
                'metadata': json.dumps(video_metadata)
            }

            headers = {
                'x-api-key': 'shashank_ka_vision786'
            }

            # Send POST request to main system
            response = requests.post(
                MAIN_SYSTEM_ENDPOINT,
                files=files,
                data=data,
                headers=headers,
                timeout=30  # 30 second timeout
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Video successfully sent to main system!")
                print(f"🎬 Video ID: {result.get('video_id')}")
                print(f"🎬 Access URL: {result.get('access_url')}")
                return result
            else:
                error_msg = f"Failed to send video to main system: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error sending video to main system: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    except Exception as e:
        error_msg = f"Error sending video to main system: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

app = FastAPI(title="AnimateDiff Video API", description="Text-to-Video Generation API")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint for health check
@app.get("/")
async def root():
    return {
        "message": "AnimateDiff Video API is running",
        "status": "healthy",
        "endpoints": {
            "generate_video": "/generate-video",
            "generate_video_with_transfer": "/generate-video-with-transfer",
            "send_video_to_main": "/send-video-to-main",
            "test_generate_video": "/test-generate-video",
            "health": "/health",
            "docs": "/docs"
        },
        "main_system_config": {
            "url": MAIN_SYSTEM_URL,
            "endpoint": MAIN_SYSTEM_ENDPOINT
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AnimateDiff Video API",
        "version": "1.0.0"
    }

# Secure endpoint using header-based API key
@app.post("/generate-video")
async def create_video(
    req: VideoRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")

    try:
        # Generate the video
        path = generate_video(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            steps=req.steps,
            num_frames=req.num_frames,
            fps=req.fps
        )

        # After successful video generation, send to main system
        if ENABLE_VIDEO_TRANSFER:
            try:
                transfer_result = await send_video_to_main_system(
                    video_file_path=path,
                    subject=req.subject,
                    topic=req.topic,
                    prompt=req.prompt,
                    metadata={
                        "num_frames": req.num_frames,
                        "guidance_scale": req.guidance_scale,
                        "steps": req.steps,
                        "seed": req.seed,
                        "fps": req.fps,
                        "negative_prompt": req.negative_prompt
                    }
                )

                if transfer_result.get("status") != "disabled":
                    print(f"✅ Video successfully transferred to main system!")
                    print(f"🎬 Video ID: {transfer_result.get('video_id')}")
                    print(f"🎬 Access URL: {transfer_result.get('access_url')}")

            except Exception as transfer_error:
                print(f"⚠️ Warning: Failed to transfer video to main system: {transfer_error}")
                # Continue with local file response even if transfer fails
        else:
            print("ℹ️ Video transfer is disabled - serving local file only")

        # Return the video file as before
        return FileResponse(path, media_type="video/mp4", filename=path.split("/")[-1])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

# Test endpoint without API key for development
@app.post("/test-generate-video")
async def test_create_video(req: VideoRequest):
    """Test endpoint for video generation without API key authentication"""
    path = generate_video(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        steps=req.steps,
        num_frames=req.num_frames,
        fps=req.fps
    )
    return FileResponse(path, media_type="video/mp4", filename=path.split("/")[-1])

# Manual video transfer endpoint
@app.post("/send-video-to-main")
async def manual_send_video(
    req: ManualVideoTransfer,
    x_api_key: str = Header(None)
):
    """
    Manual endpoint to send an existing video file to main system
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")

    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {req.video_path}")

    try:
        result = await send_video_to_main_system(
            video_file_path=req.video_path,
            subject=req.subject,
            topic=req.topic,
            prompt=req.prompt
        )

        return {
            "success": True,
            "message": "Video sent to main system",
            "video_id": result.get("video_id"),
            "access_url": result.get("access_url"),
            "local_path": req.video_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send video: {str(e)}")

# Enhanced generate-video endpoint that returns JSON with transfer info
@app.post("/generate-video-with-transfer")
async def create_video_with_transfer_info(
    req: VideoRequest,
    x_api_key: str = Header(None)
):
    """
    Generate video and return both local path and main system transfer information
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")

    try:
        # Generate the video
        path = generate_video(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            steps=req.steps,
            num_frames=req.num_frames,
            fps=req.fps
        )

        # After successful video generation, send to main system
        transfer_result = None
        transfer_error = None

        try:
            transfer_result = await send_video_to_main_system(
                video_file_path=path,
                subject=req.subject,
                topic=req.topic,
                prompt=req.prompt,
                metadata={
                    "num_frames": req.num_frames,
                    "guidance_scale": req.guidance_scale,
                    "steps": req.steps,
                    "seed": req.seed,
                    "fps": req.fps,
                    "negative_prompt": req.negative_prompt
                }
            )
        except Exception as e:
            transfer_error = str(e)
            print(f"⚠️ Warning: Failed to transfer video to main system: {e}")

        # Return comprehensive response
        response = {
            "success": True,
            "message": "Video generated successfully",
            "local_path": path,
            "filename": path.split("/")[-1],
            "transfer_success": transfer_result is not None,
            "transfer_error": transfer_error
        }

        if transfer_result:
            response.update({
                "video_id": transfer_result.get("video_id"),
                "access_url": transfer_result.get("access_url"),
                "main_system_response": transfer_result
            })

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
