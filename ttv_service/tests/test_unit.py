"""
TTV Service Unit Tests
Unit tests for individual components
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from ttv_service.config import TTVServiceConfig
from ttv_service.job_manager import JobStatus, TTVJob
from ttv_service.security import ContentModerator, JWTValidator
from ttv_service.storage import LocalStorageBackend
from ttv_service.events import Event, EventType
from ttv_service.monitoring import SystemMetrics, HealthStatus


class TestConfiguration:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TTVServiceConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8002
        assert config.debug is False
        assert config.content_moderation_enabled is True
    
    def test_environment_override(self):
        """Test environment variable override"""
        with patch.dict('os.environ', {'TTV_SERVICE_PORT': '9000'}):
            config = TTVServiceConfig()
            assert config.port == 9000
    
    def test_redis_config_property(self):
        """Test Redis configuration generation"""
        config = TTVServiceConfig()
        redis_config = config.redis_config
        
        assert 'broker_url' in redis_config
        assert 'result_backend' in redis_config
        assert redis_config['task_serializer'] == 'json'


class TestJobModels:
    """Test job-related models"""
    
    def test_ttv_job_creation(self):
        """Test TTVJob model creation"""
        job = TTVJob(
            id="test_job_123",
            user_id="user_123",
            request_data='{"script": "test"}',
            status=JobStatus.PENDING
        )
        
        assert job.id == "test_job_123"
        assert job.user_id == "user_123"
        assert job.status == JobStatus.PENDING
        assert job.progress == 0
        assert job.retry_count == 0
    
    def test_job_status_enum(self):
        """Test JobStatus enum values"""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.PROCESSING == "processing"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"


class TestContentModerator:
    """Test content moderation logic"""
    
    def setup_method(self):
        """Setup test content moderator"""
        self.moderator = ContentModerator()
    
    @pytest.mark.asyncio
    async def test_clean_content(self):
        """Test clean content passes moderation"""
        content = {"script": "A peaceful garden with flowers blooming"}
        result = await self.moderator.moderate_content(content)
        
        assert result["approved"] is True
        assert len(result["violations"]) == 0
        assert result["score"] == 1.0
    
    @pytest.mark.asyncio
    async def test_length_check(self):
        """Test script length validation"""
        long_content = {"script": "A" * (self.moderator.max_script_length + 1)}
        result = await self.moderator.moderate_content(long_content)
        
        assert result["approved"] is False
        violations = [v for v in result["violations"] if v["type"] == "length_violation"]
        assert len(violations) > 0
    
    @pytest.mark.asyncio
    async def test_keyword_detection(self):
        """Test forbidden keyword detection"""
        harmful_content = {"script": "This video shows violence and harmful behavior"}
        result = await self.moderator.moderate_content(harmful_content)
        
        assert result["approved"] is False
        violations = [v for v in result["violations"] if v["type"] == "content_violation"]
        assert len(violations) > 0
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self):
        """Test harmful pattern detection"""
        pattern_content = {"script": "How to kill someone in a video game"}
        result = await self.moderator.moderate_content(pattern_content)
        
        # Should have warning violations for patterns
        violations = [v for v in result["violations"] if v["severity"] == "warning"]
        assert len(violations) > 0
    
    def test_safety_score_calculation(self):
        """Test safety score calculation"""
        # No violations
        score = self.moderator._calculate_safety_score([])
        assert score == 1.0
        
        # One error violation
        violations = [{"severity": "error"}]
        score = self.moderator._calculate_safety_score(violations)
        assert score == 0.5
        
        # One warning violation
        violations = [{"severity": "warning"}]
        score = self.moderator._calculate_safety_score(violations)
        assert score == 0.8


class TestEventSystem:
    """Test event system components"""
    
    def test_event_creation(self):
        """Test event object creation"""
        event = Event(
            event_type=EventType.JOB_CREATED,
            data={"job_id": "test_123"},
            user_id="user_123",
            job_id="test_123"
        )
        
        assert event.event_type == EventType.JOB_CREATED
        assert event.data["job_id"] == "test_123"
        assert event.user_id == "user_123"
        assert event.service == "ttv"
        assert event.id is not None
    
    def test_event_serialization(self):
        """Test event serialization"""
        event = Event(
            event_type=EventType.JOB_COMPLETED,
            data={"result": "success"},
            user_id="user_123"
        )
        
        event_dict = event.to_dict()
        assert "id" in event_dict
        assert "event_type" in event_dict
        assert "timestamp" in event_dict
        assert event_dict["service"] == "ttv"
        
        event_json = event.to_json()
        parsed = json.loads(event_json)
        assert parsed["event_type"] == EventType.JOB_COMPLETED


class TestStorageBackend:
    """Test storage backend functionality"""
    
    def setup_method(self):
        """Setup test storage backend"""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorageBackend(base_path=self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_file_upload(self):
        """Test file upload functionality"""
        import tempfile
        import os
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            test_file = f.name
        
        try:
            # Upload file
            url = await self.storage.upload_file(test_file, "test/file.txt")
            
            assert "test/file.txt" in url
            assert await self.storage.file_exists("test/file.txt")
            
        finally:
            os.unlink(test_file)
    
    @pytest.mark.asyncio
    async def test_json_upload(self):
        """Test JSON data upload"""
        test_data = {
            "job_id": "test_123",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        url = await self.storage.upload_json(test_data, "test/metadata.json")
        
        assert "test/metadata.json" in url
        assert await self.storage.file_exists("test/metadata.json")
    
    @pytest.mark.asyncio
    async def test_file_deletion(self):
        """Test file deletion"""
        # Create and upload a file first
        test_data = {"test": "data"}
        await self.storage.upload_json(test_data, "test/delete_me.json")
        
        # Verify it exists
        assert await self.storage.file_exists("test/delete_me.json")
        
        # Delete it
        result = await self.storage.delete_file("test/delete_me.json")
        assert result is True
        
        # Verify it's gone
        assert not await self.storage.file_exists("test/delete_me.json")
    
    @pytest.mark.asyncio
    async def test_presigned_url(self):
        """Test presigned URL generation"""
        # For local storage, this just returns the regular URL
        url = await self.storage.get_presigned_url("test/file.txt", expires_in=3600)
        assert "test/file.txt" in url


class TestMonitoring:
    """Test monitoring components"""
    
    def test_health_status_creation(self):
        """Test HealthStatus creation"""
        status = HealthStatus(
            service="test_service",
            status="healthy",
            message="All good",
            timestamp=datetime.utcnow(),
            response_time_ms=150.5
        )
        
        assert status.service == "test_service"
        assert status.status == "healthy"
        assert status.response_time_ms == 150.5
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation"""
        metrics = SystemMetrics(
            cpu_percent=45.2,
            memory_percent=67.8,
            disk_percent=23.1,
            gpu_utilization=78.5,
            active_jobs=3,
            queue_length=12
        )
        
        assert metrics.cpu_percent == 45.2
        assert metrics.memory_percent == 67.8
        assert metrics.active_jobs == 3
        assert metrics.gpu_utilization == 78.5


class TestUtilities:
    """Test utility functions"""
    
    def test_duration_estimation(self):
        """Test job duration estimation"""
        from ttv_service.job_manager import JobManager
        
        job_manager = JobManager()
        
        # Test basic request
        request_data = {
            "script": "Short script",
            "quality": "balanced",
            "video_style": "realistic"
        }
        
        duration = job_manager._estimate_duration(request_data)
        assert duration > 0
        assert duration <= 900  # Max 15 minutes
    
    def test_duration_estimation_with_quality(self):
        """Test duration estimation with different quality levels"""
        from ttv_service.job_manager import JobManager
        
        job_manager = JobManager()
        
        base_request = {
            "script": "Test script for duration estimation",
            "video_style": "realistic"
        }
        
        # Test different quality levels
        low_quality = {**base_request, "quality": "low"}
        high_quality = {**base_request, "quality": "high"}
        
        low_duration = job_manager._estimate_duration(low_quality)
        high_duration = job_manager._estimate_duration(high_quality)
        
        assert high_duration > low_duration
    
    def test_script_length_impact(self):
        """Test script length impact on duration estimation"""
        from ttv_service.job_manager import JobManager
        
        job_manager = JobManager()
        
        short_script = {"script": "Short", "quality": "balanced"}
        long_script = {"script": "A" * 1000, "quality": "balanced"}
        
        short_duration = job_manager._estimate_duration(short_script)
        long_duration = job_manager._estimate_duration(long_script)
        
        assert long_duration > short_duration


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_job_status(self):
        """Test handling of invalid job status"""
        with pytest.raises(ValueError):
            JobStatus("invalid_status")
    
    def test_invalid_event_type(self):
        """Test handling of invalid event type"""
        with pytest.raises(ValueError):
            EventType("invalid_event_type")
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self):
        """Test storage error handling"""
        storage = LocalStorageBackend(base_path="/nonexistent/path")
        
        # This should raise an exception or return an error
        with pytest.raises(Exception):
            await storage.upload_json({"test": "data"}, "test.json")


class TestValidation:
    """Test input validation"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid storage backend
        with pytest.raises(ValueError):
            config = TTVServiceConfig(storage_backend="invalid_backend")
    
    def test_rate_limit_validation(self):
        """Test rate limiting configuration"""
        config = TTVServiceConfig()
        
        assert config.rate_limit_requests_per_minute > 0
        assert config.rate_limit_burst > 0
    
    def test_content_length_limits(self):
        """Test content length validation"""
        moderator = ContentModerator()
        
        assert moderator.max_script_length > 0
        assert len(moderator.forbidden_keywords) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])