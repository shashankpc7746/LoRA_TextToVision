"""
Adaptive Pipeline for Task 4 Day 2
Integrated pipeline with caching, RL, compression, and quality assessment
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import hashlib

from .device_probe import get_device_capabilities
from .budget_planner import plan_video_quality
from .tier_router import route_generation_task
from .workload_analyzer import analyze_generation_task
from .cache_manager import get_cache_manager
from .rl_policy import get_rl_policy, State, Action
from .compression_engine import get_compression_engine
from .quality_assessor import get_quality_assessor, QualityMetrics


@dataclass
class PipelineResult:
    """Result of adaptive pipeline execution"""
    success: bool
    video_path: Optional[str]
    metadata: Dict[str, Any]
    quality_metrics: Optional[QualityMetrics]
    cache_hits: List[str]
    rl_decisions: List[Dict[str, Any]]
    compression_info: Optional[Dict[str, Any]]
    total_time_seconds: float
    total_cost_usd: float
    tier_used: str


class AdaptivePipeline:
    """Complete adaptive video generation pipeline with Day 2 features"""

    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.rl_policy = get_rl_policy()
        self.compression_engine = get_compression_engine()
        self.quality_assessor = get_quality_assessor()

    def process_request(self, request: Dict[str, Any]) -> PipelineResult:
        """
        Process a complete video generation request with adaptive intelligence

        Args:
            request: Video generation request with prompts, style, constraints, etc.

        Returns:
            PipelineResult with complete processing information
        """
        start_time = time.time()
        cache_hits = []
        rl_decisions = []
        total_cost = 0.0

        try:
            # Step 1: Analyze workload and device capabilities
            device_caps = get_device_capabilities()
            task_analysis = analyze_generation_task(request)

            # Step 2: Check cache for reusable assets
            cache_results = self._check_and_load_cache(request, task_analysis)
            cache_hits.extend(cache_results["hits"])

            # Step 3: Plan quality with caching considerations
            quality_plan = plan_video_quality(
                device_caps,
                task_analysis.complexity,
                request.get("preferences", {})
            )

            # Step 4: Route to optimal tier
            routing_decision = route_generation_task(
                device_caps,
                quality_plan.__dict__,
                task_analysis.complexity,
                request.get("preferences", {})
            )

            # Step 5: Generate video (placeholder - would call actual generation)
            generation_result = self._generate_video_adaptive(
                request, cache_results, quality_plan, routing_decision
            )
            total_cost += generation_result.get("cost_usd", 0.0)

            # Step 6: Assess and potentially retry with RL
            quality_metrics = None
            if generation_result["success"]:
                quality_metrics = self._assess_and_retry_if_needed(
                    generation_result["video_path"],
                    request,
                    rl_decisions
                )

                # Update total cost if retry occurred
                if quality_metrics and "retry_cost" in quality_metrics.__dict__:
                    total_cost += quality_metrics.__dict__["retry_cost"]

            # Step 7: Compress and finalize
            compression_info = None
            final_video_path = generation_result["video_path"]

            if generation_result["success"] and final_video_path:
                compression_info = self._compress_and_finalize(
                    final_video_path,
                    device_caps.get("device_class", "desktop"),
                    quality_plan.quality_preset
                )
                if compression_info["success"]:
                    final_video_path = compression_info["output_path"]

            # Step 8: Cache successful results
            if generation_result["success"]:
                self._cache_generation_results(request, generation_result, task_analysis)

            # Calculate total time
            total_time = time.time() - start_time

            return PipelineResult(
                success=generation_result["success"],
                video_path=final_video_path,
                metadata={
                    "request": request,
                    "task_analysis": task_analysis.__dict__,
                    "quality_plan": quality_plan.__dict__,
                    "routing_decision": {
                        "tier": routing_decision.tier,
                        "reason": routing_decision.reason,
                        "estimated_cost": routing_decision.estimated_cost,
                        "confidence": routing_decision.confidence
                    },
                    "cache_stats": self.cache_manager.get_stats(),
                    "rl_stats": self.rl_policy.get_policy_stats()
                },
                quality_metrics=quality_metrics,
                cache_hits=cache_hits,
                rl_decisions=rl_decisions,
                compression_info=compression_info,
                total_time_seconds=total_time,
                total_cost_usd=total_cost,
                tier_used=routing_decision.tier
            )

        except Exception as e:
            total_time = time.time() - start_time
            return PipelineResult(
                success=False,
                video_path=None,
                metadata={"error": str(e)},
                quality_metrics=None,
                cache_hits=cache_hits,
                rl_decisions=rl_decisions,
                compression_info=None,
                total_time_seconds=total_time,
                total_cost_usd=total_cost,
                tier_used="unknown"
            )

    def _check_and_load_cache(self, request: Dict[str, Any], task_analysis) -> Dict[str, Any]:
        """Check cache for reusable assets"""
        hits = []

        # Check for background cache
        scene_type = request.get("scene_type", "generic")
        style = request.get("style", "realistic")

        cached_bg = self.cache_manager.get_background(scene_type, style)
        if cached_bg:
            hits.append(f"background_{scene_type}_{style}")

        # Check for pose cache
        pose_name = request.get("character_pose", "default")
        cached_pose = self.cache_manager.get_pose(pose_name)
        if cached_pose:
            hits.append(f"pose_{pose_name}")

        # Check for seed cache
        prompt_text = request.get("prompt", "")
        prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]
        cached_seed = self.cache_manager.get_seed(prompt_hash)
        if cached_seed:
            hits.append(f"seed_{prompt_hash}")

        return {
            "hits": hits,
            "background": cached_bg,
            "pose": cached_pose,
            "seed": cached_seed
        }

    def _generate_video_adaptive(self, request: Dict[str, Any], cache_results: Dict[str, Any],
                               quality_plan, routing_decision) -> Dict[str, Any]:
        """Generate video with adaptive parameters"""
        # This is a placeholder - in real implementation, this would call the actual
        # AnimateDiff generation pipeline with cached assets

        # Simulate generation based on cache hits and quality plan
        cache_hit_bonus = len(cache_results["hits"]) * 0.1  # 10% speedup per cache hit
        base_time = quality_plan.estimated_time_sec
        actual_time = base_time * (1 - min(cache_hit_bonus, 0.5))  # Max 50% speedup

        # Simulate cost based on tier
        tier_costs = {
            "local": 0.0,
            "office_gpu": 0.02,
            "yotta": 0.15
        }
        cost_per_minute = tier_costs.get(routing_decision.tier, 0.0)
        cost_usd = (actual_time / 60) * cost_per_minute

        # Simulate success (90% success rate)
        success = True  # For demo purposes

        return {
            "success": success,
            "video_path": f"/tmp/generated_video_{int(time.time())}.mp4" if success else None,
            "generation_time": actual_time,
            "cost_usd": cost_usd,
            "cache_used": cache_results["hits"],
            "quality_preset": quality_plan.quality_preset
        }

    def _assess_and_retry_if_needed(self, video_path: str, request: Dict[str, Any],
                                  rl_decisions: List[Dict[str, Any]]) -> Optional[QualityMetrics]:
        """Assess quality and decide whether to retry using RL"""
        try:
            # Assess quality (this would compare to original if available)
            # For demo, we'll simulate quality metrics
            vmaf_score = 75.0 + (10 * (0.5 - time.time() % 1))  # Random between 65-85
            psnr_score = 25.0 + (5 * (0.5 - time.time() % 1))   # Random between 20-30
            ssim_score = 0.85 + (0.1 * (0.5 - time.time() % 1)) # Random between 0.8-0.9

            quality_metrics = QualityMetrics(
                vmaf_score=vmaf_score,
                psnr_score=psnr_score,
                ssim_score=ssim_score,
                bitrate_kbps=1500.0,
                compression_ratio=0.6,
                encoding_time_seconds=120.0,
                file_size_mb=45.0
            )

            # Create state for RL
            current_state = State(
                vmaf_score=vmaf_score,
                latency_ms=120000,  # 2 minutes
                cost_usd=0.02,
                tier="local",
                quality_preset="balanced",
                device_class="desktop",
                task_complexity="medium"
            )

            # Get RL decision
            should_retry, action, reason = self.rl_policy.should_retry(current_state)

            rl_decision = {
                "state": current_state.__dict__,
                "should_retry": should_retry,
                "action": action.value if action else None,
                "reason": reason,
                "timestamp": time.time()
            }
            rl_decisions.append(rl_decision)

            # If retry needed, simulate retry (in real implementation, this would trigger re-generation)
            if should_retry and action:
                print(f"[RL] Retrying with action: {action.value} - {reason}")
                # Simulate retry cost
                quality_metrics.__dict__["retry_cost"] = 0.01

            return quality_metrics

        except Exception as e:
            print(f"[WARNING] Quality assessment failed: {e}")
            return None

    def _compress_and_finalize(self, video_path: str, device_class: str,
                             quality_preset: str) -> Dict[str, Any]:
        """Compress video using optimal preset"""
        try:
            # Get optimal compression preset
            compression_preset = self.compression_engine.get_optimal_preset(device_class)

            # Compress video
            output_path = video_path.replace(".mp4", "_compressed.mp4")
            result = self.compression_engine.compress_video(
                video_path,
                output_path,
                compression_preset
            )

            return result

        except Exception as e:
            print(f"[WARNING] Compression failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _cache_generation_results(self, request: Dict[str, Any], generation_result: Dict[str, Any],
                                task_analysis):
        """Cache successful generation results"""
        try:
            # Cache background if generated
            scene_type = request.get("scene_type", "generic")
            style = request.get("style", "realistic")
            # In real implementation, extract background from generated video
            # self.cache_manager.cache_background(scene_type, style, background_data)

            # Cache seed/features
            prompt_text = request.get("prompt", "")
            prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]
            seed_data = {
                "prompt": prompt_text,
                "style": style,
                "quality_preset": generation_result.get("quality_preset"),
                "generation_time": generation_result.get("generation_time"),
                "cache_timestamp": time.time()
            }
            self.cache_manager.cache_seed(prompt_hash, seed_data)

        except Exception as e:
            print(f"[WARNING] Caching generation results failed: {e}")

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            "cache_stats": self.cache_manager.get_stats(),
            "rl_stats": self.rl_policy.get_policy_stats(),
            "compression_presets": list(self.compression_engine.presets.keys()),
            "pipeline_version": "2.0.0"
        }


# Global pipeline instance
_adaptive_pipeline = None

def get_adaptive_pipeline() -> AdaptivePipeline:
    """Get global adaptive pipeline instance"""
    global _adaptive_pipeline
    if _adaptive_pipeline is None:
        _adaptive_pipeline = AdaptivePipeline()
    return _adaptive_pipeline

def process_adaptive_request(request: Dict[str, Any]) -> PipelineResult:
    """Convenience function to process adaptive requests"""
    pipeline = get_adaptive_pipeline()
    return pipeline.process_request(request)