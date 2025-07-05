# main.py
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from animate_generator import generate_video
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

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

# Root endpoint for health check
@app.get("/")
async def root():
    return {
        "message": "AnimateDiff Video API is running",
        "status": "healthy",
        "endpoints": {
            "generate_video": "/generate-video",
            "health": "/health",
            "docs": "/docs"
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
