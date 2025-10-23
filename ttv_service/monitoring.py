"""
TTV Service Monitoring and Health Checks
Comprehensive monitoring with Sentry integration, performance metrics, and worker status tracking
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
import json

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session

from .config import settings
from .job_manager import job_manager, JobStatus
from .storage import get_storage_backend
from .events import emit_event, EventType


logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status"""
    service: str
    status: str  # healthy, unhealthy, degraded
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = None


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    active_jobs: int = 0
    queue_length: int = 0
    worker_count: int = 0


class SentryManager:
    """Sentry error tracking and performance monitoring"""
    
    def __init__(self):
        if settings.sentry_dsn and settings.environment != "development":
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment=settings.environment,
                traces_sample_rate=1.0 if settings.environment == "staging" else 0.1,
                profiles_sample_rate=0.1,
                integrations=[
                    FastApiIntegration(auto_enable=True),
                    CeleryIntegration(),
                    RedisIntegration(),
                    SqlalchemyIntegration(),
                ],
                before_send=self._before_send,
                before_send_transaction=self._before_send_transaction,
            )
            logger.info("Sentry monitoring initialized")
    
    def _before_send(self, event, hint):
        """Filter sensitive data before sending to Sentry"""
        # Remove sensitive information
        if 'request' in event:
            if 'headers' in event['request']:
                # Remove authorization headers
                event['request']['headers'].pop('authorization', None)
                event['request']['headers'].pop('x-api-key', None)
        
        return event
    
    def _before_send_transaction(self, event, hint):
        """Filter transaction data"""
        return event
    
    def capture_exception(self, error: Exception, extra: Dict[str, Any] = None):
        """Capture exception with additional context"""
        with sentry_sdk.push_scope() as scope:
            if extra:
                for key, value in extra.items():
                    scope.set_tag(key, value)
            sentry_sdk.capture_exception(error)
    
    def capture_message(self, message: str, level: str = "info", extra: Dict[str, Any] = None):
        """Capture custom message"""
        with sentry_sdk.push_scope() as scope:
            if extra:
                for key, value in extra.items():
                    scope.set_tag(key, value)
            sentry_sdk.capture_message(message, level)


class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'ttv_requests_total',
            'Total TTV requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'ttv_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Job metrics
        self.job_count = Counter(
            'ttv_jobs_total',
            'Total TTV jobs',
            ['status', 'user_id']
        )
        
        self.job_duration = Histogram(
            'ttv_job_duration_seconds',
            'Job duration in seconds',
            ['status']
        )
        
        self.active_jobs = Gauge(
            'ttv_active_jobs',
            'Currently active jobs'
        )
        
        self.queue_length = Gauge(
            'ttv_queue_length',
            'Current queue length',
            ['queue_name']
        )
        
        # System metrics
        self.cpu_usage = Gauge('ttv_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('ttv_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('ttv_disk_usage_percent', 'Disk usage percentage')
        self.gpu_utilization = Gauge('ttv_gpu_utilization_percent', 'GPU utilization percentage')
        self.gpu_memory_usage = Gauge('ttv_gpu_memory_usage_percent', 'GPU memory usage percentage')
        
        # Worker metrics
        self.worker_count = Gauge('ttv_worker_count', 'Number of active workers')
        self.worker_health = Gauge('ttv_worker_health', 'Worker health status', ['worker_id'])
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_job_metrics(self, status: str, duration: float = None, user_id: str = None):
        """Record job metrics"""
        self.job_count.labels(status=status, user_id=user_id or 'unknown').inc()
        if duration:
            self.job_duration.labels(status=status).observe(duration)
    
    def update_system_metrics(self, metrics: SystemMetrics):
        """Update system metrics"""
        self.cpu_usage.set(metrics.cpu_percent)
        self.memory_usage.set(metrics.memory_percent)
        self.disk_usage.set(metrics.disk_percent)
        self.active_jobs.set(metrics.active_jobs)
        self.worker_count.set(metrics.worker_count)
        
        if metrics.gpu_utilization is not None:
            self.gpu_utilization.set(metrics.gpu_utilization)
        if metrics.gpu_memory_percent is not None:
            self.gpu_memory_usage.set(metrics.gpu_memory_percent)
    
    def update_queue_metrics(self, queue_stats: Dict[str, int]):
        """Update queue metrics"""
        for queue_name, length in queue_stats.items():
            self.queue_length.labels(queue_name=queue_name).set(length)


class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.storage = get_storage_backend()
        self.last_check_time = {}
        self.health_history = {}
    
    async def check_database_health(self) -> HealthStatus:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            from sqlmodel import Session, text
            
            with Session(job_manager.db_engine) as session:
                # Simple connectivity test
                result = session.exec(text("SELECT 1")).first()
                
                # Performance test - count jobs
                job_count = session.exec(text("SELECT COUNT(*) FROM ttv_jobs")).first()
                
                response_time = (time.time() - start_time) * 1000
                
                if response_time > 1000:  # 1 second threshold
                    status = "degraded"
                    message = f"Database slow response: {response_time:.2f}ms"
                else:
                    status = "healthy"
                    message = f"Database operational, {job_count} jobs in history"
                
                return HealthStatus(
                    service="database",
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    details={"job_count": job_count}
                )
                
        except Exception as e:
            return HealthStatus(
                service="database",
                status="unhealthy",
                message=f"Database error: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_redis_health(self) -> HealthStatus:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        
        try:
            from redis import Redis
            redis_client = Redis.from_url(settings.redis_url)
            
            # Ping test
            redis_client.ping()
            
            # Performance test
            redis_client.set("health_check", "test", ex=10)
            value = redis_client.get("health_check")
            
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = redis_client.info()
            memory_usage = info.get('used_memory_human', 'unknown')
            connected_clients = info.get('connected_clients', 0)
            
            status = "healthy" if response_time < 100 else "degraded"
            message = f"Redis operational, {connected_clients} clients, {memory_usage} memory"
            
            return HealthStatus(
                service="redis",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "memory_usage": memory_usage,
                    "connected_clients": connected_clients
                }
            )
            
        except Exception as e:
            return HealthStatus(
                service="redis",
                status="unhealthy",
                message=f"Redis error: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_storage_health(self) -> HealthStatus:
        """Check storage backend health"""
        start_time = time.time()
        
        try:
            # Test storage connectivity
            health_result = await self.storage.health_check()
            response_time = (time.time() - start_time) * 1000
            
            if health_result["status"] == "healthy":
                status = "healthy"
                message = f"Storage ({health_result['backend']}) operational"
            else:
                status = "unhealthy"
                message = f"Storage error: {health_result.get('error', 'Unknown error')}"
            
            return HealthStatus(
                service="storage",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details=health_result
            )
            
        except Exception as e:
            return HealthStatus(
                service="storage",
                status="unhealthy",
                message=f"Storage error: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_worker_health(self) -> HealthStatus:
        """Check Celery worker health and status"""
        start_time = time.time()
        
        try:
            from celery import Celery
            from ttv_service.job_manager import celery_app
            
            # Get worker stats
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            active = inspect.active() or {}
            
            if not stats:
                return HealthStatus(
                    service="workers",
                    status="unhealthy",
                    message="No workers available",
                    timestamp=datetime.utcnow(),
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            worker_count = len(stats)
            active_tasks = sum(len(tasks) for tasks in active.values())
            
            # Check for stuck workers
            stuck_workers = []
            for worker_name, tasks in active.items():
                for task in tasks:
                    # Check if task has been running too long
                    if time.time() - task.get('time_start', time.time()) > 3600:  # 1 hour
                        stuck_workers.append(worker_name)
            
            if stuck_workers:
                status = "degraded"
                message = f"{worker_count} workers, {active_tasks} active tasks, {len(stuck_workers)} stuck"
            else:
                status = "healthy"
                message = f"{worker_count} workers, {active_tasks} active tasks"
            
            return HealthStatus(
                service="workers",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    "worker_count": worker_count,
                    "active_tasks": active_tasks,
                    "stuck_workers": stuck_workers
                }
            )
            
        except Exception as e:
            return HealthStatus(
                service="workers",
                status="unhealthy",
                message=f"Worker check error: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_gpu_health(self) -> Optional[HealthStatus]:
        """Check GPU health and availability"""
        start_time = time.time()
        
        try:
            import GPUtil
            
            gpus = GPUtil.getGPUs()
            if not gpus:
                return HealthStatus(
                    service="gpu",
                    status="unhealthy",
                    message="No GPUs available",
                    timestamp=datetime.utcnow(),
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            gpu_info = []
            unhealthy_gpus = 0
            
            for gpu in gpus:
                gpu_data = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "temperature": gpu.temperature
                }
                gpu_info.append(gpu_data)
                
                # Check for issues
                if gpu.temperature > 85 or gpu.memoryUtil > 0.95:
                    unhealthy_gpus += 1
            
            if unhealthy_gpus > 0:
                status = "degraded"
                message = f"{len(gpus)} GPUs, {unhealthy_gpus} with issues"
            else:
                status = "healthy"
                message = f"{len(gpus)} GPUs operational"
            
            return HealthStatus(
                service="gpu",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={"gpus": gpu_info}
            )
            
        except ImportError:
            # GPUtil not available, skip GPU check
            return None
        except Exception as e:
            return HealthStatus(
                service="gpu",
                status="unhealthy",
                message=f"GPU check error: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all services"""
        checks = await asyncio.gather(
            self.check_database_health(),
            self.check_redis_health(),
            self.check_storage_health(),
            self.check_worker_health(),
            return_exceptions=True
        )
        
        # Add GPU check if available
        gpu_check = await self.check_gpu_health()
        if gpu_check:
            checks.append(gpu_check)
        
        # Process results
        health_results = {}
        overall_status = "healthy"
        
        for check in checks:
            if isinstance(check, Exception):
                logger.error(f"Health check failed: {str(check)}")
                continue
            
            if isinstance(check, HealthStatus):
                health_results[check.service] = asdict(check)
                
                # Determine overall status
                if check.status == "unhealthy":
                    overall_status = "unhealthy"
                elif check.status == "degraded" and overall_status != "unhealthy":
                    overall_status = "degraded"
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": health_results,
            "system": asdict(system_metrics)
        }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Job metrics
            active_jobs = 0
            queue_length = 0
            worker_count = 0
            
            try:
                # Get job statistics
                with Session(job_manager.db_engine) as session:
                    from sqlmodel import text
                    active_count = session.exec(
                        text("SELECT COUNT(*) FROM ttv_jobs WHERE status = 'processing'")
                    ).first()
                    active_jobs = active_count or 0
                
                # Get queue statistics
                queue_stats = asyncio.create_task(job_manager.get_queue_stats())
                if queue_stats:
                    queue_length = sum(queue_stats.result().get('queue_lengths', {}).values())
                    worker_count = queue_stats.result().get('worker_stats', {}).get('total_workers', 0)
            
            except Exception as e:
                logger.error(f"Error getting job metrics: {str(e)}")
            
            # GPU metrics
            gpu_utilization = None
            gpu_memory_percent = None
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Average GPU utilization
                    gpu_utilization = sum(gpu.load for gpu in gpus) / len(gpus) * 100
                    gpu_memory_percent = sum(gpu.memoryUtil for gpu in gpus) / len(gpus) * 100
            except ImportError:
                pass
            except Exception as e:
                logger.error(f"Error getting GPU metrics: {str(e)}")
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_percent=gpu_memory_percent,
                active_jobs=active_jobs,
                queue_length=queue_length,
                worker_count=worker_count
            )
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return SystemMetrics(cpu_percent=0, memory_percent=0, disk_percent=0)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for request monitoring"""
    
    def __init__(self, app: FastAPI, metrics: PrometheusMetrics):
        super().__init__(app)
        self.metrics = metrics
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request info
        method = request.method
        path = request.url.path
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        self.metrics.record_request(method, path, response.status_code, duration)
        
        # Add headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response


# Global instances
sentry_manager = SentryManager()
prometheus_metrics = PrometheusMetrics()
health_checker = HealthChecker()


# Background task for periodic health checks
async def periodic_health_check():
    """Background task for periodic health monitoring"""
    while True:
        try:
            # Perform health check
            health_result = await health_checker.comprehensive_health_check()
            
            # Update Prometheus metrics
            system_metrics = SystemMetrics(**health_result["system"])
            prometheus_metrics.update_system_metrics(system_metrics)
            
            # Emit health event
            await emit_event(
                EventType.SYSTEM_HEALTH,
                health_result
            )
            
            # Log critical issues
            if health_result["overall_status"] == "unhealthy":
                logger.error(f"System health check failed: {health_result}")
                sentry_manager.capture_message(
                    "TTV System Health Alert",
                    level="error",
                    extra=health_result
                )
            
            # Sleep until next check
            await asyncio.sleep(settings.health_check_interval)
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            sentry_manager.capture_exception(e)
            await asyncio.sleep(60)  # Retry in 1 minute


# Endpoint handlers
async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    return await health_checker.comprehensive_health_check()


async def get_metrics() -> Response:
    """Get Prometheus metrics"""
    # Update system metrics before serving
    system_metrics = health_checker.get_system_metrics()
    prometheus_metrics.update_system_metrics(system_metrics)
    
    # Generate Prometheus format
    output = generate_latest()
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)


def setup_monitoring(app: FastAPI):
    """Setup monitoring for FastAPI application"""
    # Add monitoring middleware
    app.add_middleware(MonitoringMiddleware, metrics=prometheus_metrics)
    
    # Start background health check
    asyncio.create_task(periodic_health_check())
    
    logger.info("Monitoring system initialized")