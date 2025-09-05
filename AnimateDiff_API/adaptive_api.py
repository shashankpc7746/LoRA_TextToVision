#!/usr/bin/env python3
"""
Adaptive API Module - Task-4 Day-1
New API endpoint that integrates device probe, budget planner, tier router, and workload analyzer
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Import adaptive engine modules
# Add AnimateDiff path to sys.path for adaptive_engine imports
animatediff_path = Path(__file__).parent.parent / "AnimateDiff"
sys.path.insert(0, str(animatediff_path))

try:
    from adaptive_engine import (  # type: ignore
        get_device_capabilities,  # type: ignore
        plan_video_quality,  # type: ignore
        BudgetConstraints,  # type: ignore
        route_generation_task,  # type: ignore
        analyze_generation_task,  # type: ignore
        device_probe,  # type: ignore
        budget_planner,  # type: ignore
        tier_router,  # type: ignore
        workload_analyzer,  # type: ignore
        # Day 2 Components
        get_cache_manager,  # type: ignore
        get_rl_policy,  # type: ignore
        get_compression_engine,  # type: ignore
        get_quality_assessor,  # type: ignore
        get_adaptive_pipeline,  # type: ignore
        process_adaptive_request  # type: ignore
    )
except ImportError as e:
    print(f"[ERROR] Failed to import adaptive_engine package: {e}")
    print(f"[INFO] AnimateDiff path: {animatediff_path}")
    print("[INFO] Please ensure adaptive_engine package is properly installed")
    raise

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


class AdaptiveVideoRequest(BaseModel):
    """Task-4 Adaptive Video Generation Request"""
    prompt: str
    style: Optional[str] = "realistic"
    target_quality: Optional[str] = "balanced"
    max_cost_usd: Optional[float] = 0.10
    max_latency_sec: Optional[float] = 300
    prefer_local: Optional[bool] = True
    user_device_info: Optional[Dict[str, Any]] = None
    additional_params: Optional[Dict[str, Any]] = None


class AdaptiveVideoResponse(BaseModel):
    """Task-4 Adaptive Video Generation Response"""
    request_id: str
    status: str
    selected_tier: str
    quality_settings: Dict[str, Any]
    estimated_cost: float
    estimated_latency: int
    device_capabilities: Dict[str, Any]
    task_analysis: Dict[str, Any]
    routing_decision: Dict[str, Any]
    video_url: Optional[str] = None
    error_message: Optional[str] = None


class AdaptiveAPIManager:
    """Manages adaptive video generation requests"""

    def __init__(self):
        self.active_requests = {}
        self.completed_requests = {}

    def create_adaptive_request(self, request: AdaptiveVideoRequest) -> str:
        """Create a new adaptive video generation request"""
        request_id = f"adaptive_{int(time.time())}_{hash(request.prompt) % 10000}"

        # Store request for processing
        self.active_requests[request_id] = {
            "request": request,
            "status": "analyzing",
            "created_at": datetime.now(),
            "progress": {}
        }

        return request_id

    def process_adaptive_request(self, request_id: str) -> Dict[str, Any]:
        """Process an adaptive video generation request"""
        if request_id not in self.active_requests:
            raise HTTPException(status_code=404, detail="Request not found")

        request_data = self.active_requests[request_id]
        original_request = request_data["request"]

        try:
            # Step 1: Get device capabilities
            print(f"[ADAPTIVE] Step 1: Analyzing device capabilities...")
            device_caps = get_device_capabilities()

            # Step 2: Analyze task complexity
            print(f"[ADAPTIVE] Step 2: Analyzing task complexity...")
            task_analysis = analyze_generation_task(
                prompt=original_request.prompt,
                style=original_request.style or "realistic",
                target_quality=original_request.target_quality or "balanced",
                additional_params=original_request.additional_params
            )

            # Step 3: Plan quality settings
            print(f"[ADAPTIVE] Step 3: Planning quality settings...")
            constraints = BudgetConstraints(
                max_cost_usd=original_request.max_cost_usd or 0.10,
                max_latency_ms=(original_request.max_latency_sec or 300) * 1000
            )

            quality_settings = plan_video_quality(
                device_capabilities=device_caps,
                task_complexity=task_analysis.complexity,
                user_preferences={
                    "target_quality": original_request.target_quality or "balanced",
                    "max_cost_usd": original_request.max_cost_usd or 0.10,
                    "max_latency_sec": original_request.max_latency_sec or 300,
                    "prefer_local": original_request.prefer_local
                }
            )

            # Step 4: Route to optimal tier
            print(f"[ADAPTIVE] Step 4: Routing to optimal tier...")
            routing_decision = route_generation_task(
                device_capabilities=device_caps,
                quality_settings=quality_settings.__dict__,
                task_complexity=task_analysis.complexity,
                user_preferences={
                    "prefer_local": original_request.prefer_local,
                    "max_cost_usd": original_request.max_cost_usd or 0.10,
                    "max_latency_sec": original_request.max_latency_sec or 300
                }
            )

            # Step 5: Generate video (simplified for demo)
            print(f"[ADAPTIVE] Step 5: Generating video on {routing_decision.tier} tier...")

            # For now, simulate video generation
            video_url = self._simulate_video_generation(
                request_id, routing_decision.tier, quality_settings
            )

            # Prepare response
            response = {
                "request_id": request_id,
                "status": "completed",
                "selected_tier": routing_decision.tier,
                "quality_settings": {
                    "resolution": quality_settings.resolution,
                    "num_frames": quality_settings.num_frames,
                    "fps": quality_settings.fps,
                    "steps": quality_settings.steps,
                    "guidance_scale": quality_settings.guidance_scale,
                    "style": quality_settings.style,
                    "estimated_vram_gb": quality_settings.estimated_vram_gb,
                    "estimated_time_sec": quality_settings.estimated_time_sec,
                    "estimated_cost_usd": quality_settings.estimated_cost_usd
                },
                "estimated_cost": routing_decision.estimated_cost,
                "estimated_latency": routing_decision.estimated_latency,
                "device_capabilities": device_caps,
                "task_analysis": {
                    "complexity": task_analysis.complexity,
                    "confidence": task_analysis.confidence,
                    "estimated_vram_gb": task_analysis.estimated_vram_gb,
                    "estimated_time_sec": task_analysis.estimated_time_sec,
                    "recommended_tier": task_analysis.recommended_tier,
                    "reasoning": task_analysis.reasoning,
                    "factors": task_analysis.factors
                },
                "routing_decision": {
                    "tier": routing_decision.tier,
                    "reason": routing_decision.reason,
                    "estimated_cost": routing_decision.estimated_cost,
                    "estimated_latency": routing_decision.estimated_latency,
                    "confidence": routing_decision.confidence,
                    "fallback_options": routing_decision.fallback_options
                },
                "video_url": video_url,
                "processing_time_sec": time.time() - request_data["created_at"].timestamp()
            }

            # Move to completed
            self.completed_requests[request_id] = response
            del self.active_requests[request_id]

            return response

        except Exception as e:
            print(f"[ADAPTIVE] Error processing request {request_id}: {e}")
            self.active_requests[request_id]["status"] = "failed"
            self.active_requests[request_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=f"Adaptive processing failed: {str(e)}")

    def _simulate_video_generation(self, request_id: str, tier: str, quality_settings) -> str:
        """Simulate video generation (replace with actual generation)"""
        # In real implementation, this would call the actual video generation
        # For now, return a placeholder URL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"/videos/adaptive_{request_id}_{timestamp}.mp4"

    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a request"""
        if request_id in self.completed_requests:
            return self.completed_requests[request_id]
        elif request_id in self.active_requests:
            return {
                "request_id": request_id,
                "status": self.active_requests[request_id]["status"],
                "created_at": self.active_requests[request_id]["created_at"].isoformat(),
                "progress": self.active_requests[request_id]["progress"]
            }
        else:
            raise HTTPException(status_code=404, detail="Request not found")


# Global API manager instance
api_manager = AdaptiveAPIManager()

# FastAPI app for adaptive endpoints
adaptive_app = FastAPI(title="Adaptive Video Generation API", version="2.0.0")


@adaptive_app.post("/ttv/generate", response_model=AdaptiveVideoResponse)
async def generate_adaptive_video(request: AdaptiveVideoRequest, background_tasks: BackgroundTasks):
    """
    Task-4 Adaptive Video Generation Endpoint

    This endpoint automatically:
    - Detects device capabilities
    - Analyzes task complexity
    - Plans optimal quality settings
    - Routes to best processing tier
    - Generates video with adaptive intelligence
    """
    try:
        # Create request
        request_id = api_manager.create_adaptive_request(request)

        # Process request (in real implementation, this might be async)
        result = api_manager.process_adaptive_request(request_id)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptive generation failed: {str(e)}")


@adaptive_app.get("/ttv/status/{request_id}")
async def get_adaptive_status(request_id: str):
    """Get status of adaptive video generation request"""
    return api_manager.get_request_status(request_id)


@adaptive_app.get("/ttv/capabilities")
async def get_system_capabilities():
    """Get current system capabilities and tier status"""
    device_caps = get_device_capabilities()
    tier_status = tier_router.tier_router.get_tier_status()

    return {
        "device_capabilities": device_caps,
        "tier_status": tier_status,
        "available_quality_presets": budget_planner.budget_planner.get_available_presets(),
        "timestamp": int(time.time())
    }


@adaptive_app.get("/ttv/analyze")
async def analyze_prompt(
    prompt: str,
    style: str = "realistic",
    target_quality: str = "balanced"
):
    """Analyze a prompt without generating video"""
    analysis = analyze_generation_task(prompt, style, target_quality)

    return {
        "analysis": {
            "complexity": analysis.complexity,
            "confidence": analysis.confidence,
            "estimated_vram_gb": analysis.estimated_vram_gb,
            "estimated_time_sec": analysis.estimated_time_sec,
            "recommended_tier": analysis.recommended_tier,
            "reasoning": analysis.reasoning,
            "factors": analysis.factors
        }
    }


# Day 2 Endpoints: Caching, RL, Compression, Quality Assessment

@adaptive_app.get("/ttv/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    cache = get_cache_manager()
    return cache.get_stats()


@adaptive_app.post("/ttv/cache/clear")
async def clear_cache(cache_type: str = "all"):
    """Clear cache entries"""
    cache = get_cache_manager()
    if cache_type == "all":
        cache.clear_cache()
    else:
        cache.clear_cache(cache_type)
    return {"message": f"Cleared {cache_type} cache"}


@adaptive_app.get("/ttv/rl/stats")
async def get_rl_stats():
    """Get RL policy statistics"""
    rl = get_rl_policy()
    return rl.get_policy_stats()


@adaptive_app.post("/ttv/rl/reset")
async def reset_rl_policy():
    """Reset RL policy"""
    rl = get_rl_policy()
    rl.reset_policy()
    return {"message": "RL policy reset"}


@adaptive_app.get("/ttv/compression/presets")
async def get_compression_presets():
    """Get available compression presets"""
    compressor = get_compression_engine()
    return {
        "presets": list(compressor.presets.keys()),
        "details": {name: {
            "codec": preset.codec,
            "crf": preset.crf,
            "target_vmaf": preset.target_vmaf,
            "description": preset.description
        } for name, preset in compressor.presets.items()}
    }


@adaptive_app.post("/ttv/compress")
async def compress_video(
    input_path: str,
    output_path: str,
    preset: str = "desktop_standard"
):
    """Compress a video file"""
    compressor = get_compression_engine()
    result = compressor.compress_video(input_path, output_path, preset)
    return result


@adaptive_app.post("/ttv/quality/assess")
async def assess_video_quality(
    video_path: str,
    reference_path: str = None,
    sample_rate: float = 0.1
):
    """Assess video quality using VMAF"""
    assessor = get_quality_assessor()
    try:
        metrics = assessor.assess_quality(video_path, reference_path or video_path, sample_rate)
        return {
            "vmaf_score": metrics.vmaf_score,
            "psnr_score": metrics.psnr_score,
            "ssim_score": metrics.ssim_score,
            "bitrate_kbps": metrics.bitrate_kbps,
            "compression_ratio": metrics.compression_ratio,
            "file_size_mb": metrics.file_size_mb,
            "meets_threshold_70": assessor.meets_quality_threshold(metrics, 70.0),
            "meets_threshold_80": assessor.meets_quality_threshold(metrics, 80.0),
            "recommendation": assessor.get_quality_recommendation(metrics)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")


@adaptive_app.post("/ttv/pipeline/process")
async def process_pipeline_request(request: Dict[str, Any]):
    """Process request through complete Day 2 adaptive pipeline"""
    try:
        result = process_adaptive_request(request)
        return {
            "success": result.success,
            "video_path": result.video_path,
            "total_time_seconds": result.total_time_seconds,
            "total_cost_usd": result.total_cost_usd,
            "tier_used": result.tier_used,
            "cache_hits": result.cache_hits,
            "rl_decisions": result.rl_decisions,
            "quality_metrics": result.quality_metrics.__dict__ if result.quality_metrics else None,
            "compression_info": result.compression_info,
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@adaptive_app.get("/ttv/pipeline/stats")
async def get_pipeline_stats():
    """Get comprehensive pipeline statistics"""
    pipeline = get_adaptive_pipeline()
    return pipeline.get_pipeline_stats()


@adaptive_app.get("/ttv/day2/status")
async def get_day2_status():
    """Get Day 2 system status"""
    cache = get_cache_manager()
    rl = get_rl_policy()
    compressor = get_compression_engine()
    assessor = get_quality_assessor()
    pipeline = get_adaptive_pipeline()

    return {
        "cache": {
            "entries": cache.get_stats()["total_entries"],
            "size_mb": cache.get_stats()["total_size_mb"]
        },
        "rl_policy": {
            "experiences": rl.get_policy_stats()["total_experiences"],
            "avg_reward": rl.get_policy_stats()["average_reward"]
        },
        "compression": {
            "presets_available": len(compressor.presets)
        },
        "pipeline": {
            "version": "2.0.0",
            "components": ["cache", "rl", "compression", "quality", "adaptive_pipeline"]
        },
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    print("[INFO] Starting Adaptive Video Generation API...")
    print("[INFO] Task-4 Day-2: Caching + RL Policy + Compression + Quality Assessment")
    print("[INFO] Enhanced with intelligent caching, reinforcement learning, and quality optimization")
    print("[INFO] API will be available at http://localhost:8001")
    print("[INFO] Docs at http://localhost:8001/docs")

    uvicorn.run(
        "adaptive_api:adaptive_app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )