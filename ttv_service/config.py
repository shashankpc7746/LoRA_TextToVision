"""
TTV Service Configuration Module
Handles environment variables and service configuration for Task 8 integration
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator
from functools import lru_cache


class TTVServiceConfig(BaseSettings):
    """Configuration for TTV Service production deployment"""
    
    # Service Configuration
    host: str = "0.0.0.0"
    port: int = 8002
    workers: int = 4
    debug: bool = False
    
    # BHIV Integration
    bhiv_backend_url: str = "http://192.168.0.121:8001"
    bhiv_api_key: Optional[str] = None
    bhiv_webhook_secret: Optional[str] = None
    
    # Database Configuration
    database_url: str = "postgresql://postgres:password@localhost:5432/ai_agent"
    redis_url: str = "redis://localhost:6379/0"
    
    # Supabase Configuration
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None
    
    # Storage Configuration
    storage_backend: str = "supabase"  # local, s3, supabase, bhiv_bucket
    s3_bucket_name: Optional[str] = None
    s3_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # BHIV Bucket Integration
    bhiv_bucket_path: str = "bucket"
    bhiv_storage_backend: str = "supabase"
    
    # Security Configuration
    jwt_secret_key: str = "development-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Content Moderation
    content_moderation_enabled: bool = True
    max_script_length: int = 10000
    forbidden_keywords: List[str] = ["violence", "hate", "explicit", "illegal", "harmful"]
    
    # Monitoring and Observability
    sentry_dsn: Optional[str] = None
    environment: str = "development"
    log_level: str = "INFO"
    
    # GPU and Processing Configuration
    gpu_memory_limit: int = 8192  # MB
    max_concurrent_jobs: int = 3
    job_timeout_minutes: int = 15
    cleanup_temp_files: bool = True
    
    # Video Generation Settings
    default_video_style: str = "realistic"
    default_quality: str = "balanced"
    default_fps: int = 12
    max_video_duration: int = 120  # seconds
    max_scenes_per_video: int = 10
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 10
    rate_limit_burst: int = 5
    
    # Health Check Configuration
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5    # seconds
    
    @validator('forbidden_keywords', pre=True)
    def parse_forbidden_keywords(cls, v):
        if isinstance(v, str):
            return [keyword.strip().lower() for keyword in v.split(',')]
        return v
    
    @validator('storage_backend')
    def validate_storage_backend(cls, v):
        valid_backends = ['local', 's3', 'supabase', 'bhiv_bucket']
        if v not in valid_backends:
            raise ValueError(f'storage_backend must be one of {valid_backends}')
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_environments = ['development', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f'environment must be one of {valid_environments}')
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    @property
    def redis_config(self) -> dict:
        """Redis configuration for Celery"""
        return {
            'broker_url': self.redis_url,
            'result_backend': self.redis_url,
            'task_serializer': 'json',
            'accept_content': ['json'],
            'result_serializer': 'json',
            'timezone': 'UTC',
            'enable_utc': True,
        }
    
    @property
    def database_config(self) -> dict:
        """Database configuration"""
        return {
            'url': self.database_url,
            'echo': not self.is_production,
            'pool_pre_ping': True,
            'pool_recycle': 300,
        }
    
    @property
    def cors_config(self) -> dict:
        """CORS configuration"""
        if self.is_production:
            return {
                'allow_origins': [self.bhiv_backend_url],
                'allow_credentials': True,
                'allow_methods': ["GET", "POST", "PUT", "DELETE"],
                'allow_headers': ["*"],
            }
        else:
            return {
                'allow_origins': ["*"],
                'allow_credentials': True,
                'allow_methods': ["*"],
                'allow_headers': ["*"],
            }
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "TTV_SERVICE_",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra environment variables to be ignored
    }


@lru_cache()
def get_settings() -> TTVServiceConfig:
    """Get cached settings instance"""
    return TTVServiceConfig()


# Global settings instance
settings = get_settings()