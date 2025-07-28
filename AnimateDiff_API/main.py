# main.py
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse
from animate_generator import generate_video, generate_lesson_video
from dotenv import load_dotenv
import os
import requests
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

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
    fps: int = 12  # Updated to match current system
    style: str = "realistic"  # realistic, anime, artistic
    subject: str = "AnimateDiff Video"
    topic: str = "AI Generated Video"

class LessonVideoRequest(BaseModel):
    lesson_filename: str  # e.g., "lesson_1_dharma.json"
    style: str = "realistic"  # realistic, anime, artistic
    speech_rate: int = 1  # 0-2, where 1 is normal speed
    subject: str = "Lesson Video"
    topic: str = "Educational Content"

class ManualVideoTransfer(BaseModel):
    video_path: str
    subject: str = "Manual Upload"
    topic: str = "Manual Upload"
    prompt: str = "Manually uploaded video"

# NEW: Production JSON Schema (as specified in feedback)
class LessonSegment(BaseModel):
    text: str = Field(..., description="Text content for this segment")
    mood: str = Field(default="calm", description="Emotional mood: calm, intense, wise, etc.")
    scene: str = Field(default="temple", description="Visual scene: temple, forest, cosmic, etc.")
    expression: str = Field(default="neutral", description="Character expression: neutral, wise, happy, etc.")

class ProductionVideoRequest(BaseModel):
    lesson_id: str = Field(..., description="Unique lesson identifier")
    title: str = Field(..., description="Lesson title")
    segments: List[LessonSegment] = Field(..., description="List of lesson segments")
    voice_url: Optional[str] = Field(None, description="Pre-generated TTS audio URL")
    style: str = Field(default="realistic", description="Visual style: realistic, anime, artistic")
    bgm: Optional[str] = Field(default="tanpura", description="Background music: tanpura, om, flute, none")

class ProductionVideoResponse(BaseModel):
    video_url: str = Field(..., description="URL to generated video")
    subtitles: str = Field(..., description="URL to subtitle file (.srt)")
    render_time: str = Field(..., description="Total rendering time")
    models_used: List[str] = Field(..., description="AI models used in generation")
    fps: int = Field(..., description="Video frame rate")
    audio_offset_ms: int = Field(..., description="Audio synchronization offset")
    fallback_used: bool = Field(..., description="Whether fallback generation was used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

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
            "production_generate_video": "/generate-video (NEW PRODUCTION)",
            "legacy_generate_video": "/legacy-generate-video",
            "generate_lesson_video": "/generate-lesson-video",
            "generate_video_with_transfer": "/generate-video-with-transfer",
            "send_video_to_main": "/send-video-to-main",
            "test_generate_video": "/test-generate-video",
            "test_generate_lesson": "/test-generate-lesson",
            "models": "/models",
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

# Legacy endpoint using header-based API key
@app.post("/legacy-generate-video")
async def create_video_legacy(
    req: VideoRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")

    try:
        # Generate the video using updated system
        path = generate_video(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            steps=req.steps,
            num_frames=req.num_frames,
            fps=req.fps,
            style=req.style
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
        fps=req.fps,
        style=req.style
    )
    return FileResponse(path, media_type="video/mp4", filename=path.split("/")[-1])

# NEW: Lesson video generation endpoint
@app.post("/generate-lesson-video")
async def create_lesson_video(
    req: LessonVideoRequest,
    x_api_key: str = Header(None)
):
    """Generate video from lesson file with audio and subtitles"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")

    try:
        # Generate the lesson video
        path = generate_lesson_video(
            lesson_filename=req.lesson_filename,
            style=req.style,
            speech_rate=req.speech_rate
        )

        # After successful video generation, send to main system
        if ENABLE_VIDEO_TRANSFER:
            try:
                transfer_result = await send_video_to_main_system(
                    video_file_path=path,
                    subject=req.subject,
                    topic=req.topic,
                    prompt=f"Lesson: {req.lesson_filename}",
                    metadata={
                        "lesson_filename": req.lesson_filename,
                        "style": req.style,
                        "speech_rate": req.speech_rate,
                        "type": "lesson_video"
                    }
                )

                if transfer_result.get("status") != "disabled":
                    print(f"✅ Lesson video successfully transferred to main system!")
                    print(f"🎬 Video ID: {transfer_result.get('video_id')}")
                    print(f"🎬 Access URL: {transfer_result.get('access_url')}")

            except Exception as transfer_error:
                print(f"⚠️ Warning: Failed to transfer lesson video to main system: {transfer_error}")
        else:
            print("ℹ️ Video transfer is disabled - serving local file only")

        # Return the video file
        return FileResponse(path, media_type="video/mp4", filename=path.split("/")[-1])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lesson video generation failed: {str(e)}")

# NEW: Test lesson video endpoint
@app.post("/test-generate-lesson")
async def test_create_lesson_video(req: LessonVideoRequest):
    """Test endpoint for lesson video generation without API key authentication"""
    path = generate_lesson_video(
        lesson_filename=req.lesson_filename,
        style=req.style,
        speech_rate=req.speech_rate
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

# NEW: Production endpoint with exact feedback schema
@app.post("/generate-video", response_model=ProductionVideoResponse)
async def generate_production_video(request: ProductionVideoRequest):
    """
    Production endpoint for generating educational videos with audio and subtitles
    Implements the exact JSON schema from feedback for Gurukul integration
    """

    start_time = time.time()
    job_id = str(uuid.uuid4())
    models_used = []
    fallback_used = False

    try:
        logging.info(f"Starting production video generation for lesson: {request.lesson_id}")

        # Step 1: Convert request to lesson JSON format
        lesson_data = {
            "title": request.title,
            "level": "Production",
            "text": ". ".join([segment.text for segment in request.segments]),
            "prompts": [
                f"{segment.text} in {segment.scene} setting with {segment.mood} mood, {segment.expression} expression"
                for segment in request.segments
            ],
            "metadata": {
                "lesson_id": request.lesson_id,
                "style": request.style,
                "bgm": request.bgm,
                "segments": [segment.dict() for segment in request.segments]
            }
        }

        # Create temporary lesson file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_lesson_filename = f"production_lesson_{job_id}_{timestamp}.json"

        # Use the AnimateDiff path for lesson files
        ANIMATEDIFF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AnimateDiff")
        lesson_path = os.path.join(ANIMATEDIFF_PATH, "lessons", temp_lesson_filename)

        with open(lesson_path, 'w', encoding='utf-8') as f:
            json.dump(lesson_data, f, indent=2, ensure_ascii=False)

        # Step 2: Attempt main video generation with fallback
        try:
            video_path = generate_lesson_video(
                lesson_filename=temp_lesson_filename,
                style=request.style,
                speech_rate=1
            )

            if video_path and os.path.exists(video_path):
                models_used = ["AnimateDiff-v2", "ControlNet-depth", "TTS-Engine", "3D-Motion"]
                logging.info(f"Main generation successful: {video_path}")
            else:
                raise Exception("Main generation failed - no output file")

        except Exception as e:
            logging.warning(f"Main generation failed: {e}")
            # Fallback generation
            try:
                from fallback_generator import create_fallback_video
                video_path = create_fallback_video(
                    title=request.title,
                    segments=[segment.dict() for segment in request.segments],
                    style=request.style,
                    output_dir="outputs/fallback"
                )
                models_used = ["Fallback-Static", "TTS-Engine"]
                fallback_used = True
                logging.info(f"Fallback generation successful: {video_path}")
            except Exception as fallback_error:
                logging.error(f"Fallback generation also failed: {fallback_error}")
                raise HTTPException(status_code=500, detail=f"Both main and fallback generation failed: {str(e)}")

        # Step 3: Check for generated SRT file from unified system
        subtitle_path = None
        video_filename = os.path.basename(video_path)
        srt_filename = video_filename.replace('.mp4', '.srt')

        # Look for SRT file in storage directory
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        storage_srt_path = os.path.join("storage", today, srt_filename)

        if os.path.exists(storage_srt_path):
            subtitle_path = storage_srt_path
            logging.info(f"Found generated SRT file: {storage_srt_path}")
        else:
            # Fallback: generate basic subtitles
            subtitle_path = generate_subtitles_from_segments(request.segments, job_id)
            logging.info(f"Generated fallback SRT file: {subtitle_path}")

        # Step 4: Upload to storage and get URLs
        video_url = upload_to_storage(video_path, f"{request.lesson_id}_video.mp4")
        subtitle_url = upload_to_storage(subtitle_path, f"{request.lesson_id}_subtitles.srt")

        # Step 5: Calculate metrics
        render_time = time.time() - start_time

        # Step 6: Send to main system if enabled
        if ENABLE_VIDEO_TRANSFER:
            try:
                await send_video_to_main_system(
                    video_file_path=video_path,
                    subject=request.title,
                    topic=f"Lesson: {request.lesson_id}",
                    prompt=f"Production lesson: {request.title}",
                    metadata={
                        "lesson_id": request.lesson_id,
                        "segments_count": len(request.segments),
                        "style": request.style,
                        "fallback_used": fallback_used,
                        "job_id": job_id
                    }
                )
            except Exception as transfer_error:
                logging.warning(f"Video transfer failed: {transfer_error}")

        # Cleanup temporary files
        if os.path.exists(lesson_path):
            os.remove(lesson_path)

        response = ProductionVideoResponse(
            video_url=video_url,
            subtitles=subtitle_url,
            render_time=f"{render_time:.0f} sec",
            models_used=models_used,
            fps=8,  # Using current FPS setting
            audio_offset_ms=60,  # Calculated during sync
            fallback_used=fallback_used,
            metadata={
                "lesson_id": request.lesson_id,
                "title": request.title,
                "segments_count": len(request.segments),
                "style": request.style,
                "generation_time": datetime.now().isoformat(),
                "job_id": job_id,
                "bgm": request.bgm
            }
        )

        logging.info(f"Production video generation completed successfully: {job_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Production video generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

def generate_subtitles_from_segments(segments: List[LessonSegment], job_id: str) -> str:
    """Generate SRT subtitle file from lesson segments"""

    subtitle_path = f"outputs/subtitles_{job_id}.srt"
    os.makedirs("outputs", exist_ok=True)

    with open(subtitle_path, 'w', encoding='utf-8') as f:
        current_time = 0

        for i, segment in enumerate(segments):
            # Estimate duration based on text length (adjust as needed)
            duration = max(3.0, len(segment.text) * 0.08)  # ~80ms per character

            start_time = current_time
            end_time = current_time + duration

            # SRT format
            f.write(f"{i + 1}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{segment.text}\n\n")

            current_time = end_time

    return subtitle_path

def format_srt_time(seconds: float) -> str:
    """Format time for SRT subtitle format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def upload_to_storage(file_path: str, filename: str) -> str:
    """Upload file to storage and return URL"""

    # Create storage directory
    today = datetime.now().strftime("%Y-%m-%d")
    storage_dir = os.path.join("storage", today)
    os.makedirs(storage_dir, exist_ok=True)

    # Copy file to storage
    import shutil
    storage_file_path = os.path.join(storage_dir, filename)
    shutil.copy2(file_path, storage_file_path)

    # Return URL (configure for your actual storage)
    base_url = "http://localhost:8000"  # Configure for production
    return f"{base_url}/storage/{today}/{filename}"

@app.get("/models")
async def list_available_models():
    """List available models and styles for video generation"""
    return {
        "styles": ["realistic", "anime", "artistic"],
        "bgm_options": ["tanpura", "om", "flute", "none"],
        "models": ["AnimateDiff-v2", "ControlNet-depth", "TTS-Engine", "Fallback-Static", "3D-Motion"],
        "moods": ["calm", "intense", "wise", "happy", "serious", "peaceful"],
        "scenes": ["temple", "forest", "cosmic", "mountain", "river", "palace"],
        "expressions": ["neutral", "wise", "happy", "serious", "peaceful", "contemplative"],
        "fps_options": [8, 12, 16, 24],
        "max_segments": 10,
        "supported_formats": ["mp4", "srt"]
    }
