"""
TTV Service Job Queue Implementation
Handles GPU worker queues using Celery for heavy TTV processing
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

from celery import Celery
from celery.result import AsyncResult
from redis import Redis
from sqlmodel import SQLModel, Field, Session, create_engine, select

from .config import settings


# Job Status Enum
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Job Models
class TTVJob(SQLModel, table=True):
    """TTV Job model for database storage"""
    __tablename__ = "ttv_jobs"
    
    id: Optional[str] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    request_data: str = Field()  # JSON string of TTVGenerateRequest
    result_data: Optional[str] = Field(default=None)  # JSON string of results
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    progress: int = Field(default=0)  # 0-100
    worker_id: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    priority: int = Field(default=0)  # Higher = more priority
    estimated_duration: Optional[int] = Field(default=None)  # seconds
    actual_duration: Optional[int] = Field(default=None)  # seconds


@dataclass
class JobProgress:
    """Job progress tracking"""
    job_id: str
    progress: int
    status: JobStatus
    message: str
    current_step: str
    total_steps: int
    estimated_remaining: Optional[int] = None  # seconds


# Celery Configuration
celery_app = Celery(
    'ttv_service',
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=['ttv_service.tasks']
)

celery_app.conf.update(
    **settings.redis_config,
    task_routes={
        'ttv_service.tasks.generate_video': {'queue': 'gpu_high'},
        'ttv_service.tasks.cleanup_job': {'queue': 'cleanup'},
        'ttv_service.tasks.health_check': {'queue': 'monitoring'},
    },
    task_default_queue='default',
    worker_prefetch_multiplier=1,  # Important for GPU workers
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_time_limit=settings.job_timeout_minutes * 60,
    task_soft_time_limit=(settings.job_timeout_minutes - 2) * 60,
)


class JobManager:
    """Manages TTV job lifecycle and worker coordination"""
    
    def __init__(self):
        self.redis_client = Redis.from_url(settings.redis_url)
        self.db_engine = create_engine(settings.database_config['url'])
        self.logger = logging.getLogger(__name__)
        
        # Create tables if they don't exist
        SQLModel.metadata.create_all(self.db_engine)
    
    async def submit_job(
        self, 
        user_id: str, 
        request_data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Submit a new TTV generation job"""
        job_id = f"ttv_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        # Store job in database
        with Session(self.db_engine) as session:
            job = TTVJob(
                id=job_id,
                user_id=user_id,
                request_data=json.dumps(request_data),
                priority=priority,
                estimated_duration=self._estimate_duration(request_data)
            )
            session.add(job)
            session.commit()
        
        # Submit to Celery
        from .tasks import generate_video
        celery_result = generate_video.apply_async(
            args=[job_id, request_data],
            task_id=job_id,
            priority=priority,
            queue='gpu_high' if priority > 5 else 'default'
        )
        
        self.logger.info(f"Submitted job {job_id} for user {user_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and progress"""
        with Session(self.db_engine) as session:
            job = session.get(TTVJob, job_id)
            if not job:
                return None
            
            # Get Celery task status
            celery_result = AsyncResult(job_id, app=celery_app)
            
            # Get progress from Redis
            progress_data = self.redis_client.get(f"job_progress:{job_id}")
            progress = None
            if progress_data:
                progress = json.loads(progress_data)
            
            return {
                'job_id': job.id,
                'status': job.status,
                'progress': job.progress,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'error_message': job.error_message,
                'estimated_duration': job.estimated_duration,
                'actual_duration': job.actual_duration,
                'celery_status': celery_result.status,
                'progress_details': progress
            }
    
    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a job"""
        with Session(self.db_engine) as session:
            job = session.get(TTVJob, job_id)
            if not job or job.user_id != user_id:
                return False
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            # Cancel Celery task
            celery_app.control.revoke(job_id, terminate=True)
            
            # Update database
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            session.add(job)
            session.commit()
            
            self.logger.info(f"Cancelled job {job_id}")
            return True
    
    async def get_user_jobs(
        self, 
        user_id: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get jobs for a specific user"""
        with Session(self.db_engine) as session:
            statement = (
                select(TTVJob)
                .where(TTVJob.user_id == user_id)
                .order_by(TTVJob.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            jobs = session.exec(statement).all()
            
            return [
                {
                    'job_id': job.id,
                    'status': job.status,
                    'progress': job.progress,
                    'created_at': job.created_at.isoformat(),
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'error_message': job.error_message,
                    'estimated_duration': job.estimated_duration,
                    'actual_duration': job.actual_duration
                }
                for job in jobs
            ]
    
    async def update_job_progress(
        self, 
        job_id: str, 
        progress: int, 
        status: JobStatus = None,
        message: str = "",
        current_step: str = "",
        total_steps: int = 1
    ):
        """Update job progress"""
        # Update database
        with Session(self.db_engine) as session:
            job = session.get(TTVJob, job_id)
            if job:
                job.progress = progress
                if status:
                    job.status = status
                    if status == JobStatus.PROCESSING and not job.started_at:
                        job.started_at = datetime.utcnow()
                    elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                        job.completed_at = datetime.utcnow()
                        if job.started_at:
                            job.actual_duration = int((job.completed_at - job.started_at).total_seconds())
                
                session.add(job)
                session.commit()
        
        # Update Redis for real-time progress
        progress_data = JobProgress(
            job_id=job_id,
            progress=progress,
            status=status or JobStatus.PROCESSING,
            message=message,
            current_step=current_step,
            total_steps=total_steps
        )
        
        self.redis_client.setex(
            f"job_progress:{job_id}",
            300,  # 5 minutes TTL
            json.dumps(asdict(progress_data))
        )
    
    async def cleanup_old_jobs(self, days: int = 7):
        """Clean up old completed jobs"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with Session(self.db_engine) as session:
            statement = (
                select(TTVJob)
                .where(TTVJob.completed_at < cutoff_date)
                .where(TTVJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]))
            )
            old_jobs = session.exec(statement).all()
            
            for job in old_jobs:
                # Clean up job files
                await self._cleanup_job_files(job.id)
                
                # Delete from database
                session.delete(job)
            
            session.commit()
            self.logger.info(f"Cleaned up {len(old_jobs)} old jobs")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        inspect = celery_app.control.inspect()
        
        active_tasks = inspect.active() or {}
        scheduled_tasks = inspect.scheduled() or {}
        reserved_tasks = inspect.reserved() or {}
        
        # Redis queue lengths
        queue_lengths = {}
        for queue in ['default', 'gpu_high', 'cleanup', 'monitoring']:
            queue_lengths[queue] = self.redis_client.llen(f"celery:{queue}")
        
        # Database job counts
        with Session(self.db_engine) as session:
            status_counts = {}
            for status in JobStatus:
                count = session.exec(
                    select(TTVJob).where(TTVJob.status == status)
                ).count()
                status_counts[status.value] = count
        
        return {
            'queue_lengths': queue_lengths,
            'active_tasks': sum(len(tasks) for tasks in active_tasks.values()),
            'scheduled_tasks': sum(len(tasks) for tasks in scheduled_tasks.values()),
            'reserved_tasks': sum(len(tasks) for tasks in reserved_tasks.values()),
            'job_status_counts': status_counts,
            'worker_stats': {
                'active_workers': len(active_tasks),
                'total_workers': len(inspect.stats() or {})
            }
        }
    
    def _estimate_duration(self, request_data: Dict[str, Any]) -> int:
        """Estimate job duration based on request parameters"""
        base_duration = 120  # 2 minutes base
        
        # Add time based on script length
        script_length = len(request_data.get('script', ''))
        duration_per_char = 0.1  # seconds per character
        
        # Add time based on quality
        quality_multipliers = {
            'low': 0.5,
            'balanced': 1.0,
            'high': 1.5,
            'ultra': 2.0
        }
        quality = request_data.get('quality', 'balanced')
        quality_multiplier = quality_multipliers.get(quality, 1.0)
        
        # Add time based on video style complexity
        style_multipliers = {
            'simple': 0.8,
            'realistic': 1.0,
            'cinematic': 1.3,
            'anime': 1.1
        }
        style = request_data.get('video_style', 'realistic')
        style_multiplier = style_multipliers.get(style, 1.0)
        
        estimated_duration = int(
            (base_duration + script_length * duration_per_char) * 
            quality_multiplier * style_multiplier
        )
        
        return min(estimated_duration, settings.job_timeout_minutes * 60)
    
    async def _cleanup_job_files(self, job_id: str):
        """Clean up temporary files for a job"""
        from .tasks import cleanup_job
        cleanup_job.apply_async(args=[job_id], queue='cleanup')


# Global job manager instance
job_manager = JobManager()