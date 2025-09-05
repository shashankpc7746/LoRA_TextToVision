"""
GPU Queue Manager for Task 4 Day 3
Manages office GPU pool queue and job scheduling
"""

import time
import threading
import requests
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class JobStatus(Enum):
    """GPU job status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class GPUJob:
    """GPU job representation"""
    job_id: str
    prompt: str
    priority: JobPriority
    estimated_time_sec: int
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_gpu: Optional[str] = None
    progress: float = 0.0
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUStatus:
    """GPU status information"""
    gpu_id: str
    name: str
    memory_total: int
    memory_used: int
    utilization: float
    temperature: float
    is_available: bool
    current_job: Optional[str] = None


class GPUQueueManager:
    """Manages GPU job queue and resource allocation"""

    def __init__(self, office_gpu_endpoint: str = "http://192.168.0.100:8001",
                 max_queue_size: int = 50, max_concurrent_jobs: int = 4):
        self.office_gpu_endpoint = office_gpu_endpoint
        self.max_queue_size = max_queue_size
        self.max_concurrent_jobs = max_concurrent_jobs

        # Job management
        self.job_queue: List[GPUJob] = []
        self.running_jobs: Dict[str, GPUJob] = {}
        self.completed_jobs: Dict[str, GPUJob] = {}

        # GPU status tracking
        self.gpu_status: Dict[str, GPUStatus] = {}
        self.last_status_update = 0
        self.status_update_interval = 30  # seconds

        # Threading
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
        self.monitor_thread.start()

    def submit_job(self, prompt: str, priority: JobPriority = JobPriority.NORMAL,
                  estimated_time_sec: int = 180, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a job to the GPU queue

        Args:
            prompt: Video generation prompt
            priority: Job priority level
            estimated_time_sec: Estimated processing time
            metadata: Additional job metadata

        Returns:
            Job ID
        """
        with self.lock:
            # Check queue size limit
            if len(self.job_queue) + len(self.running_jobs) >= self.max_queue_size:
                raise ValueError(f"Queue is full (max {self.max_queue_size} jobs)")

            # Create job
            job_id = f"gpu_job_{int(time.time())}_{hash(prompt) % 10000}"
            job = GPUJob(
                job_id=job_id,
                prompt=prompt,
                priority=priority,
                estimated_time_sec=estimated_time_sec,
                metadata=metadata or {}
            )

            # Add to queue with priority sorting
            self.job_queue.append(job)
            self.job_queue.sort(key=lambda j: (j.priority.value, j.created_at), reverse=True)

            print(f"[GPU Queue] Job {job_id} submitted with priority {priority.name}")
            return job_id

    def get_job_status(self, job_id: str) -> Optional[GPUJob]:
        """Get job status by ID"""
        with self.lock:
            # Check running jobs
            if job_id in self.running_jobs:
                return self.running_jobs[job_id]

            # Check completed jobs
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]

            # Check queue
            for job in self.job_queue:
                if job.job_id == job_id:
                    return job

        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job"""
        with self.lock:
            # Check queue
            for i, job in enumerate(self.job_queue):
                if job.job_id == job_id:
                    job.status = JobStatus.CANCELLED
                    self.completed_jobs[job_id] = job
                    del self.job_queue[i]
                    return True

            # Check running jobs
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                self.completed_jobs[job_id] = job
                del self.running_jobs[job_id]
                return True

        return False

    def _monitor_jobs(self):
        """Monitor job queue and GPU status"""
        while True:
            try:
                # Update GPU status
                self._update_gpu_status()

                # Process job queue
                self._process_queue()

                # Clean up old completed jobs
                self._cleanup_old_jobs()

            except Exception as e:
                print(f"[GPU Queue] Monitor error: {e}")

            time.sleep(5)  # Check every 5 seconds

    def _update_gpu_status(self):
        """Update GPU status from office endpoint"""
        if time.time() - self.last_status_update < self.status_update_interval:
            return

        try:
            response = requests.get(f"{self.office_gpu_endpoint}/gpu-status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                with self.lock:
                    for gpu_data in status_data.get("gpus", []):
                        gpu = GPUStatus(**gpu_data)
                        self.gpu_status[gpu.gpu_id] = gpu
                self.last_status_update = time.time()
        except Exception as e:
            # Fallback to simulated status
            self._simulate_gpu_status()

    def _simulate_gpu_status(self):
        """Simulate GPU status for testing"""
        gpus = [
            GPUStatus(
                gpu_id="gpu_01",
                name="RTX 4090",
                memory_total=24 * 1024,  # 24GB
                memory_used=8 * 1024,    # 8GB used
                utilization=65.0,
                temperature=72.0,
                is_available=True
            ),
            GPUStatus(
                gpu_id="gpu_02",
                name="RTX 4090",
                memory_total=24 * 1024,
                memory_used=12 * 1024,
                utilization=80.0,
                temperature=75.0,
                is_available=True
            ),
            GPUStatus(
                gpu_id="gpu_03",
                name="RTX 4080",
                memory_total=16 * 1024,
                memory_used=4 * 1024,
                utilization=45.0,
                temperature=68.0,
                is_available=True
            ),
            GPUStatus(
                gpu_id="gpu_04",
                name="RTX 4080",
                memory_total=16 * 1024,
                memory_used=16 * 1024,
                utilization=95.0,
                temperature=82.0,
                is_available=False
            )
        ]

        with self.lock:
            for gpu in gpus:
                self.gpu_status[gpu.gpu_id] = gpu

    def _process_queue(self):
        """Process jobs from queue"""
        with self.lock:
            # Count running jobs
            running_count = len(self.running_jobs)

            # Start new jobs if capacity available
            while (running_count < self.max_concurrent_jobs and
                   self.job_queue and
                   self._has_available_gpu()):

                # Get next job
                job = self.job_queue.pop(0)

                # Find available GPU
                available_gpu = self._get_available_gpu()
                if available_gpu:
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    job.assigned_gpu = available_gpu.gpu_id

                    self.running_jobs[job.job_id] = job
                    running_count += 1

                    # Start job processing (async)
                    threading.Thread(
                        target=self._process_job,
                        args=(job,),
                        daemon=True
                    ).start()

                    print(f"[GPU Queue] Started job {job.job_id} on {available_gpu.name}")

    def _has_available_gpu(self) -> bool:
        """Check if any GPU is available"""
        return any(gpu.is_available and gpu.current_job is None
                  for gpu in self.gpu_status.values())

    def _get_available_gpu(self) -> Optional[GPUStatus]:
        """Get best available GPU"""
        available_gpus = [
            gpu for gpu in self.gpu_status.values()
            if gpu.is_available and gpu.current_job is None
        ]

        if not available_gpus:
            return None

        # Prefer GPU with lowest utilization
        return min(available_gpus, key=lambda g: g.utilization)

    def _process_job(self, job: GPUJob):
        """Process a GPU job"""
        try:
            # Simulate job processing
            print(f"[GPU Job] Processing {job.job_id}: {job.prompt[:50]}...")

            # Update progress
            for progress in range(0, 101, 10):
                job.progress = progress
                time.sleep(job.estimated_time_sec / 10)

            # Complete job
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result_path = f"/results/{job.job_id}.mp4"

            print(f"[GPU Job] Completed {job.job_id}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()
            print(f"[GPU Job] Failed {job.job_id}: {e}")

        finally:
            # Move to completed
            with self.lock:
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
                self.completed_jobs[job.job_id] = job

    def _cleanup_old_jobs(self):
        """Clean up old completed jobs"""
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago

        with self.lock:
            to_remove = []
            for job_id, job in self.completed_jobs.items():
                if job.completed_at and job.completed_at < cutoff_time:
                    to_remove.append(job_id)

            for job_id in to_remove:
                del self.completed_jobs[job_id]

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return {
                "queued_jobs": len(self.job_queue),
                "running_jobs": len(self.running_jobs),
                "completed_jobs": len(self.completed_jobs),
                "available_gpus": sum(1 for gpu in self.gpu_status.values() if gpu.is_available),
                "total_gpus": len(self.gpu_status),
                "queue_utilization": len(self.running_jobs) / self.max_concurrent_jobs if self.max_concurrent_jobs > 0 else 0
            }

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics"""
        with self.lock:
            return {
                "gpus": [
                    {
                        "id": gpu.gpu_id,
                        "name": gpu.name,
                        "memory_used": gpu.memory_used,
                        "memory_total": gpu.memory_total,
                        "utilization": gpu.utilization,
                        "temperature": gpu.temperature,
                        "available": gpu.is_available,
                        "current_job": gpu.current_job
                    }
                    for gpu in self.gpu_status.values()
                ],
                "last_update": self.last_status_update
            }


# Global GPU queue instance
_gpu_queue = None

def get_gpu_queue() -> GPUQueueManager:
    """Get global GPU queue instance"""
    global _gpu_queue
    if _gpu_queue is None:
        _gpu_queue = GPUQueueManager()
    return _gpu_queue