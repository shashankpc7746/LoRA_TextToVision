"""
Main Orchestrator for Task-7 Quality Leap
Complete end-to-end video generation pipeline orchestration
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

# Import all pipeline components
from adapters import get_gurukul_lora, get_keyframe_generator, create_keyframe_animation
from interpolator import get_interpolation_pipeline, interpolate_video_from_keyframes
from audio_manager import get_audio_pipeline, process_lip_sync
from upscaler import get_upscale_pipeline, upscale_video_to_1080p
from motion_controller import get_rl_policy, optimize_generation_parameters
from test_tools import get_lip_sync_tester


class GenerationOrchestrator:
    """Main orchestrator for end-to-end video generation"""

    def __init__(self):
        self.gurukul_lora = get_gurukul_lora()
        self.keyframe_gen = get_keyframe_generator()
        self.interpolation_pipeline = get_interpolation_pipeline()
        self.audio_pipeline = get_audio_pipeline()
        self.upscale_pipeline = get_upscale_pipeline()
        self.rl_policy = get_rl_policy()
        self.lip_sync_tester = get_lip_sync_tester()

        # Orchestration configuration
        self.config = {
            "max_generation_time": 1800,  # 30 minutes
            "quality_target": 0.85,       # Target quality score
            "cost_budget": 1.0,          # Max cost in USD
            "enable_rl_optimization": True,
            "enable_fallbacks": True,
            "gpu_allocation": {
                "adapters": "cuda:0",      # RTX 3080
                "interpolation": "cuda:1", # RTX 3060
                "upscaling": "cuda:0",     # RTX 3080
                "audio": "cpu"             # CPU for audio processing
            }
        }

        # Generation statistics
        self.stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_quality": 0.0,
            "average_cost": 0.0,
            "average_time": 0.0
        }

    async def generate_video(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Main video generation orchestration"""

        start_time = time.time()
        generation_id = f"gen_{int(start_time)}_{hash(prompt) % 10000}"

        print(f"\n🎬 Starting video generation: {generation_id}")
        print(f"📝 Prompt: {prompt[:100]}...")

        # Update config with kwargs
        config = self.config.copy()
        config.update(kwargs)

        result = {
            "generation_id": generation_id,
            "prompt": prompt,
            "start_time": datetime.now().isoformat(),
            "pipeline_steps": [],
            "final_result": None,
            "quality_metrics": {},
            "performance_metrics": {},
            "errors": []
        }

        try:
            # Step 1: RL Parameter Optimization
            if config["enable_rl_optimization"]:
                print("🤖 Optimizing parameters with RL policy...")
                rl_result = self._optimize_parameters()
                result["rl_optimization"] = rl_result
                result["pipeline_steps"].append("rl_optimization")

            # Step 2: Keyframe Generation
            print("🎭 Generating keyframes...")
            keyframes_result = await self._generate_keyframes(prompt, config)
            if not keyframes_result["success"]:
                raise Exception(f"Keyframe generation failed: {keyframes_result.get('error')}")

            result["keyframes"] = keyframes_result
            result["pipeline_steps"].append("keyframes")

            # Step 3: Video Animation
            print("🎬 Creating base animation...")
            animation_result = await self._create_animation(keyframes_result, config)
            if not animation_result["success"]:
                raise Exception(f"Animation failed: {animation_result.get('error')}")

            result["animation"] = animation_result
            result["pipeline_steps"].append("animation")

            # Step 4: Interpolation & Stabilization
            print("🔄 Applying interpolation and stabilization...")
            interpolation_result = await self._apply_interpolation(animation_result, config)
            if not interpolation_result["success"]:
                raise Exception(f"Interpolation failed: {interpolation_result.get('error')}")

            result["interpolation"] = interpolation_result
            result["pipeline_steps"].append("interpolation")

            # Step 5: Audio & Lip-sync
            print("🎵 Adding audio and lip-sync...")
            audio_result = await self._add_audio_and_lipsync(interpolation_result, config)
            if not audio_result["success"]:
                raise Exception(f"Audio processing failed: {audio_result.get('error')}")

            result["audio"] = audio_result
            result["pipeline_steps"].append("audio")

            # Step 6: Upscaling & Polish
            print("📈 Applying upscaling and cinematic polish...")
            upscale_result = await self._apply_upscaling_and_polish(audio_result, config)
            if not upscale_result["success"]:
                raise Exception(f"Upscaling failed: {upscale_result.get('error')}")

            result["upscaling"] = upscale_result
            result["pipeline_steps"].append("upscaling")

            # Step 7: Quality Validation
            print("✅ Validating final quality...")
            quality_result = self._validate_final_quality(upscale_result, config)
            result["quality_validation"] = quality_result
            result["pipeline_steps"].append("quality_validation")

            # Success!
            end_time = time.time()
            generation_time = end_time - start_time

            result["success"] = True
            result["final_result"] = upscale_result
            result["performance_metrics"] = {
                "total_time_seconds": generation_time,
                "time_per_step": generation_time / len(result["pipeline_steps"]),
                "peak_memory_usage": "N/A",  # Would track actual memory usage
                "gpu_utilization": "N/A"     # Would track GPU usage
            }

            # Update statistics
            self._update_statistics(result, generation_time)

            print("🎉 Video generation completed successfully!")
            print(".2f")
            print(f"📁 Output: {upscale_result.get('output_path', 'N/A')}")

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["performance_metrics"] = {
                "total_time_seconds": time.time() - start_time,
                "failed_at_step": result["pipeline_steps"][-1] if result["pipeline_steps"] else "initialization"
            }
            print(f"❌ Generation failed: {e}")

        return result

    async def _generate_keyframes(self, prompt: str, config: Dict) -> Dict[str, Any]:
        """Generate keyframes from prompt"""
        try:
            keyframes_result = self.keyframe_gen.generate_keyframes_async(
                prompt, num_keyframes=6
            )

            return await keyframes_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_animation(self, keyframes_result: Dict, config: Dict) -> Dict[str, Any]:
        """Create base animation from keyframes"""
        try:
            keyframes_dir = None
            for kf in keyframes_result.get("results", []):
                if kf.get("success"):
                    # Find keyframes directory
                    kf_path = Path(kf["image_path"])
                    keyframes_dir = str(kf_path.parent)
                    break

            if not keyframes_dir:
                return {"success": False, "error": "No successful keyframes found"}

            # Create animation
            from adapters.animate_diff_bridge import create_keyframe_animation
            output_video = f"temp_animation_{int(time.time())}.mp4"

            animation_result = create_keyframe_animation(keyframes_dir, output_video)

            return animation_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _apply_interpolation(self, animation_result: Dict, config: Dict) -> Dict[str, Any]:
        """Apply interpolation and stabilization"""
        try:
            input_video = animation_result.get("output_path")
            if not input_video:
                return {"success": False, "error": "No animation video found"}

            output_video = f"interpolated_{Path(input_video).name}"

            interpolation_result = self.interpolation_pipeline.process_video_upscale(
                input_video, output_video,
                target_resolution=(1280, 720),  # 720p for interpolation
                apply_denoising=True,
                apply_cinematic_polish=False  # Will apply later
            )

            return interpolation_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _add_audio_and_lipsync(self, interpolation_result: Dict, config: Dict) -> Dict[str, Any]:
        """Add audio and lip-sync"""
        try:
            video_path = interpolation_result.get("output_path")
            if not video_path:
                return {"success": False, "error": "No interpolated video found"}

            # For now, create placeholder audio path
            # In production, this would generate or accept audio input
            audio_path = "placeholder_audio.wav"  # TODO: Implement actual audio generation

            # Apply lip-sync
            lipsync_result = self.audio_pipeline.process_lip_sync(
                video_path, audio_path, method="enhanced_sadtalker"
            )

            return lipsync_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _apply_upscaling_and_polish(self, audio_result: Dict, config: Dict) -> Dict[str, Any]:
        """Apply final upscaling and cinematic polish"""
        try:
            video_path = audio_result.get("output_path")
            if not video_path:
                return {"success": False, "error": "No audio-processed video found"}

            output_video = f"final_1080p_{Path(video_path).stem}.mp4"

            upscale_result = self.upscale_pipeline.process_video_upscale(
                video_path, output_video,
                target_resolution=(1920, 1080),
                apply_denoising=True,
                apply_cinematic_polish=True
            )

            return upscale_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_final_quality(self, upscale_result: Dict, config: Dict) -> Dict[str, Any]:
        """Validate final video quality"""
        try:
            video_path = upscale_result.get("output_path")
            if not video_path:
                return {"valid": False, "error": "No final video found"}

            # Basic validation
            validation = self.upscale_pipeline.validate_upscale_quality(video_path)

            # Lip-sync validation if audio was added
            lipsync_validation = {"score": 0.0, "valid": False}

            return {
                "upscale_validation": validation,
                "lipsync_validation": lipsync_validation,
                "overall_quality_score": validation.get("quality_metrics", {}).get("quality_score", 0.0),
                "meets_target": validation.get("quality_metrics", {}).get("quality_score", 0.0) >= config["quality_target"]
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _optimize_parameters(self) -> Dict[str, Any]:
        """Get RL-based parameter optimization"""
        try:
            # Create current state (placeholder - would use real metrics)
            from motion_controller.rl_policy import QualityState
            current_state = QualityState(
                vmaf_score=0.7,
                lip_sync_score=0.8,
                temporal_consistency=0.6,
                generation_time=120.0,
                cost=0.2
            )

            optimization = self.rl_policy.optimize_parameters(current_state)
            return optimization

        except Exception as e:
            return {"error": str(e), "fallback": "Using default parameters"}

    def _update_statistics(self, result: Dict, generation_time: float):
        """Update generation statistics"""
        self.stats["total_generations"] += 1

        if result["success"]:
            self.stats["successful_generations"] += 1

            # Update averages
            quality = result.get("quality_validation", {}).get("overall_quality_score", 0.0)
            cost = result.get("performance_metrics", {}).get("estimated_cost", 0.0)

            self.stats["average_quality"] = (
                (self.stats["average_quality"] * (self.stats["successful_generations"] - 1)) + quality
            ) / self.stats["successful_generations"]

            self.stats["average_cost"] = (
                (self.stats["average_cost"] * (self.stats["successful_generations"] - 1)) + cost
            ) / self.stats["successful_generations"]

            self.stats["average_time"] = (
                (self.stats["average_time"] * (self.stats["successful_generations"] - 1)) + generation_time
            ) / self.stats["successful_generations"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.stats.copy()

    def save_statistics(self, filepath: str = "generation_stats.json"):
        """Save statistics to file"""
        stats_with_timestamp = self.stats.copy()
        stats_with_timestamp["last_updated"] = datetime.now().isoformat()

        with open(filepath, 'w') as f:
            json.dump(stats_with_timestamp, f, indent=2)


# Global orchestrator instance
_orchestrator = None


def get_orchestrator() -> GenerationOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = GenerationOrchestrator()
    return _orchestrator


async def generate_video(prompt: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for video generation"""
    orchestrator = get_orchestrator()
    return await orchestrator.generate_video(prompt, **kwargs)


def quick_test_orchestrator():
    """Quick test of orchestrator components"""
    print("Testing orchestrator components...")

    try:
        orchestrator = get_orchestrator()

        # Test component initialization
        components = [
            ("Gurukul LoRA", orchestrator.gurukul_lora),
            ("Keyframe Generator", orchestrator.keyframe_gen),
            ("Interpolation Pipeline", orchestrator.interpolation_pipeline),
            ("Audio Pipeline", orchestrator.audio_pipeline),
            ("Upscale Pipeline", orchestrator.upscale_pipeline),
            ("RL Policy", orchestrator.rl_policy),
            ("Lip-sync Tester", orchestrator.lip_sync_tester)
        ]

        for name, component in components:
            if component is not None:
                print(f"✅ {name}: Initialized")
            else:
                print(f"❌ {name}: Failed to initialize")

        print(f"📊 Current statistics: {orchestrator.get_statistics()}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    quick_test_orchestrator()