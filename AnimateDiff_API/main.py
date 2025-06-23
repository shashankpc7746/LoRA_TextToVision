# main.py
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from animate_generator import generate_video
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="AnimateDiff Video API")

class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    seed: int = 333
    guidance_scale: float = 15
    steps: int = 25
    num_frames: int = 32
    fps: int = 8

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
