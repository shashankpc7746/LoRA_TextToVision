"""
TTV Service Celery Tasks
Implements GPU worker tasks for video generation
"""

import os
import json
import logging
import traceback
from typing import Dict, Any
from datetime import datetime, timedelta
import asyncio
import shutil
from pathlib import Path

from celery import Task
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlmodel import SQLModel

from .job_manager import TTVJob, JobStatus
from celery.exceptions import Retry

from .job_manager import celery_app, JobStatus
from .config import settings
from .storage import get_storage_backend
from .events import emit_event


logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task class with callbacks for job progress"""
    
    def on_progress(self, job_id: str, progress: int, message: str = "", current_step: str = ""):
        """Update job progress"""
        from .job_manager import job_manager
        asyncio.create_task(
            job_manager.update_job_progress(
                job_id=job_id,
                progress=progress,
                message=message,
                current_step=current_step,
                total_steps=10  # Standard TTV pipeline steps
            )
        )
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        job_id = task_id
        error_message = str(exc)
        
        from .job_manager import job_manager
        asyncio.create_task(
            job_manager.update_job_progress(
                job_id=job_id,
                progress=0,
                status=JobStatus.FAILED,
                message=f"Task failed: {error_message}"
            )
        )
        
        # Emit failure event
        asyncio.create_task(
            emit_event("ttv.job.failed", {
                "job_id": job_id,
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        logger.error(f"Task {job_id} failed: {error_message}\n{einfo}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        job_id = task_id
        
        # Emit success event
        asyncio.create_task(
            emit_event("ttv.job.completed", {
                "job_id": job_id,
                "result": retval,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        logger.info(f"Task {job_id} completed successfully")


@celery_app.task(bind=True, base=CallbackTask, name='ttv_service.tasks.generate_video')
def generate_video(self, job_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main task for TTV video generation
    Integrates with existing LoRA_TextToVision system
    """
    try:
        # Update job status to processing
        from .job_manager import job_manager
        asyncio.create_task(
            job_manager.update_job_progress(
                job_id=job_id,
                progress=5,
                status=JobStatus.PROCESSING,
                message="Starting video generation",
                current_step="initialization"
            )
        )
        
        # Step 1: Validate and prepare inputs
        self.on_progress(job_id, 10, "Validating inputs", "validation")
        script = request_data.get('script', '')
        video_style = request_data.get('video_style', 'realistic')
        quality = request_data.get('quality', 'balanced')
        
        if not script.strip():
            raise ValueError("Script cannot be empty")
        
        # Step 2: Initialize TTV system components
        self.on_progress(job_id, 20, "Initializing TTV components", "initialization")
        
        # Import existing TTV components
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        
        from orchestrator import TextToVideoOrchestrator
        from AnimateDiff.unified_video_generator import UnifiedVideoGenerator
        
        # Step 3: Create orchestrator instance
        self.on_progress(job_id, 30, "Setting up orchestrator", "setup")
        
        orchestrator = TextToVideoOrchestrator()
        
        # Step 4: Process script and generate scenes
        self.on_progress(job_id, 40, "Processing script", "script_processing")
        
        # Prepare generation parameters
        generation_params = {
            'script': script,
            'video_style': video_style,
            'quality': quality,
            'output_dir': f"temp/{job_id}",
            'num_scenes': min(request_data.get('num_scenes', 5), settings.max_scenes_per_video),
            'fps': request_data.get('fps', settings.default_fps),
            'duration': min(request_data.get('duration', 60), settings.max_video_duration)
        }
        
        # Step 5: Generate video using existing system
        self.on_progress(job_id, 50, "Generating video frames", "generation")
        
        # Create temporary output directory
        temp_dir = Path(f"temp/{job_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use existing unified video generator
            generator = UnifiedVideoGenerator()
            
            # Step 6: Generate scenes
            self.on_progress(job_id, 60, "Processing scenes", "scene_generation")
            
            video_result = generator.generate_from_script(
                script=script,
                style=video_style,
                quality=quality,
                output_path=str(temp_dir / "output.mp4")
            )
            
            # Step 7: Post-processing
            self.on_progress(job_id, 80, "Post-processing", "post_processing")
            
            if not video_result or not os.path.exists(video_result.get('output_path', '')):
                raise Exception("Video generation failed - no output file created")
            
            # Step 8: Upload to storage
            self.on_progress(job_id, 90, "Uploading to storage", "upload")
            
            storage = get_storage_backend()
            video_url = storage.upload_file(
                file_path=video_result['output_path'],
                key=f"ttv/{job_id}/video.mp4",
                content_type="video/mp4"
            )
            
            # Step 9: Generate metadata
            metadata = {
                'job_id': job_id,
                'script': script,
                'video_style': video_style,
                'quality': quality,
                'duration': video_result.get('duration', 0),
                'resolution': video_result.get('resolution', '1080p'),
                'fps': video_result.get('fps', settings.default_fps),
                'file_size': os.path.getsize(video_result['output_path']),
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Upload metadata
            metadata_url = storage.upload_json(
                data=metadata,
                key=f"ttv/{job_id}/metadata.json"
            )
            
            # Step 10: Complete job
            self.on_progress(job_id, 100, "Generation complete", "completed")
            
            result = {
                'job_id': job_id,
                'video_url': video_url,
                'metadata_url': metadata_url,
                'metadata': metadata,
                'status': 'completed'
            }
            
            # Update job status to completed
            asyncio.create_task(
                job_manager.update_job_progress(
                    job_id=job_id,
                    progress=100,
                    status=JobStatus.COMPLETED,
                    message="Video generation completed successfully"
                )
            )
            
            return result
            
        finally:
            # Cleanup temporary files if configured
            if settings.cleanup_temp_files:
                cleanup_job.apply_async(args=[job_id], queue='cleanup', countdown=300)  # 5 minutes delay
    
    except Exception as e:
        logger.error(f"Error in generate_video task {job_id}: {str(e)}\n{traceback.format_exc()}")
        
        # Update job status to failed
        from .job_manager import job_manager
        asyncio.create_task(
            job_manager.update_job_progress(
                job_id=job_id,
                progress=0,
                status=JobStatus.FAILED,
                message=f"Generation failed: {str(e)}"
            )
        )
        
        raise


@celery_app.task(name='ttv_service.tasks.cleanup_job')
def cleanup_job(job_id: str):
    """Clean up temporary files for a job"""
    try:
        temp_dir = Path(f"temp/{job_id}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary files for job {job_id}")
        
        # Clean up any other job-specific files
        cache_dir = Path(f"cache/{job_id}")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
    except Exception as e:
        logger.error(f"Error cleaning up job {job_id}: {str(e)}")


@celery_app.task(name='ttv_service.tasks.health_check')
def health_check() -> Dict[str, Any]:
    """Health check task for monitoring worker status"""
    try:
        import GPUtil
        import psutil
        
        # GPU status
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil,
                'temperature': gpu.temperature,
                'load': gpu.load
            })
        
        # System status
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'gpu_info': gpu_info,
            'system_info': system_info,
            'worker_id': os.environ.get('WORKER_ID', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'worker_id': os.environ.get('WORKER_ID', 'unknown')
        }


@celery_app.task(name='ttv_service.tasks.batch_generate')
def batch_generate(job_ids: list, request_data_list: list) -> Dict[str, Any]:
    """Process multiple TTV generation requests as a batch"""
    results = {}
    
    for job_id, request_data in zip(job_ids, request_data_list):
        try:
            result = generate_video.apply(args=[job_id, request_data])
            results[job_id] = result
        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {str(e)}")
            results[job_id] = {'error': str(e), 'status': 'failed'}
    
    return {
        'batch_id': f"batch_{int(datetime.utcnow().timestamp())}",
        'total_jobs': len(job_ids),
        'results': results,
        'completed_at': datetime.utcnow().isoformat()
    }


@celery_app.task(name='ttv_service.tasks.retry_failed_job')
def retry_failed_job(job_id: str, max_retries: int = 3):
    """Retry a failed job with exponential backoff"""
    from .job_manager import job_manager
    
    try:
        # Get job from database
        with Session(job_manager.db_engine) as session:
            job = session.get(TTVJob, job_id)
            if not job:
                logger.error(f"Job {job_id} not found for retry")
                return
            
            if job.retry_count >= max_retries:
                logger.error(f"Job {job_id} exceeded max retries ({max_retries})")
                return
            
            # Increment retry count
            job.retry_count += 1
            job.status = JobStatus.PENDING
            session.add(job)
            session.commit()
            
            # Resubmit job
            request_data = json.loads(job.request_data)
            generate_video.apply_async(
                args=[job_id, request_data],
                task_id=job_id,
                countdown=2 ** job.retry_count * 60  # Exponential backoff
            )
            
            logger.info(f"Retrying job {job_id} (attempt {job.retry_count + 1})")
    
    except Exception as e:
        logger.error(f"Error retrying job {job_id}: {str(e)}")


# Task monitoring and periodic tasks
@celery_app.task(name='ttv_service.tasks.monitor_jobs')
def monitor_jobs():
    """Monitor job health and handle stuck jobs"""
    from .job_manager import job_manager
    
    try:
        # Find stuck jobs (processing for too long)
        stuck_threshold = datetime.utcnow() - timedelta(minutes=settings.job_timeout_minutes + 5)
        
        with Session(job_manager.db_engine) as session:
            stuck_jobs = session.exec(
                select(TTVJob)
                .where(TTVJob.status == JobStatus.PROCESSING)
                .where(TTVJob.started_at < stuck_threshold)
            ).all()
            
            for job in stuck_jobs:
                logger.warning(f"Found stuck job {job.id}, marking as failed")
                job.status = JobStatus.FAILED
                job.error_message = "Job timeout - exceeded maximum processing time"
                job.completed_at = datetime.utcnow()
                session.add(job)
            
            session.commit()
            
            if stuck_jobs:
                logger.info(f"Marked {len(stuck_jobs)} stuck jobs as failed")
    
    except Exception as e:
        logger.error(f"Error monitoring jobs: {str(e)}")


# Periodic task setup
celery_app.conf.beat_schedule = {
    'monitor-jobs': {
        'task': 'ttv_service.tasks.monitor_jobs',
        'schedule': 300.0,  # Every 5 minutes
    },
    'health-check': {
        'task': 'ttv_service.tasks.health_check',
        'schedule': 60.0,   # Every minute
    },
}