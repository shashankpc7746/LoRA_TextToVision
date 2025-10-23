"""
TTV Service Package
Production-ready Text-to-Vision service integration for BHIV ecosystem
"""

__version__ = "1.0.0"
__author__ = "TTV Service Team"
__description__ = "FastAPI microservice for integrating LoRA_TextToVision with BHIV platform"

# Import core components for easy access (avoid importing main app to prevent dependency issues)
from .config import settings

# Import classes without initializing instances to avoid database connections
from .job_manager import JobStatus
from .events import EventType

# Functions that can be imported safely
def get_job_manager():
    """Lazy import of JobManager to avoid database connection at import time"""
    from .job_manager import JobManager
    return JobManager()

def get_storage_backend():
    """Get storage backend"""
    from .storage import get_storage_backend as _get_storage_backend
    return _get_storage_backend()

def emit_event(event_type, data):
    """Emit event"""
    from .events import emit_event as _emit_event
    return _emit_event(event_type, data)

def get_current_user():
    """Get current user"""
    from .security import get_current_user as _get_current_user
    return _get_current_user()

def health_checker():
    """Health check"""
    from .monitoring import health_checker as _health_checker
    return _health_checker()

def get_app():
    """Lazy import of the main FastAPI app to avoid import-time dependency issues"""
    from .main import app
    return app

__all__ = [
    "get_app",
    "settings", 
    "get_job_manager",
    "JobStatus",
    "get_storage_backend",
    "emit_event",
    "EventType",
    "get_current_user",
    "health_checker"
]