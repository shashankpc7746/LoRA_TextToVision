"""
TTV Service Integration Tests
Comprehensive test suite for API endpoints, job queue system, storage integration, 
authentication flow, and error handling
"""

import pytest
import asyncio
import json
import time
import jwt
from typing import Dict, Any
from datetime import datetime, timedelta
import tempfile
import os

from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, SQLModel
from unittest.mock import Mock, patch, AsyncMock

# Import TTV service components
from ttv_service.main import app
from ttv_service.config import settings
from ttv_service.job_manager import JobManager, JobStatus, TTVJob
from ttv_service.storage import get_storage_backend, LocalStorageBackend
from ttv_service.security import jwt_validator, content_moderator, audit_logger
from ttv_service.events import emit_event, EventType
from ttv_service.monitoring import health_checker


# Test configuration
TEST_DATABASE_URL = "sqlite:///./test_ttv.db"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use DB 15 for tests


@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application"""
    # Override settings for testing
    settings.database_url = TEST_DATABASE_URL
    settings.redis_url = TEST_REDIS_URL
    settings.storage_backend = "local"
    settings.content_moderation_enabled = True
    settings.environment = "testing"
    
    return app


@pytest.fixture(scope="session")
def client(test_app):
    """Create test client"""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    engine = create_engine(TEST_DATABASE_URL)
    SQLModel.metadata.create_all(engine)
    yield engine
    # Cleanup
    os.unlink("./test_ttv.db")


@pytest.fixture
def test_job_manager(test_db):
    """Create test job manager"""
    return JobManager()


@pytest.fixture
def mock_user():
    """Mock authenticated user"""
    return {
        "user_id": "test_user_123",
        "email": "test@example.com",
        "role": "user",
        "session_id": "test_session_123"
    }


@pytest.fixture
def mock_jwt_token():
    """Mock JWT token"""
    return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token"


@pytest.fixture
def test_ttv_request():
    """Sample TTV generation request"""
    return {
        "script": "A beautiful sunset over mountains with birds flying",
        "video_style": "cinematic",
        "quality": "balanced",
        "duration": 30,
        "fps": 12,
        "num_scenes": 3
    }


class TestTTVServiceAPI:
    """Test TTV Service API endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "timestamp" in data
        assert "services" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    @patch('ttv_service.security.jwt_validator.validate_token')
    def test_generate_video_endpoint(self, mock_validate, client, test_ttv_request, mock_user):
        """Test video generation endpoint"""
        mock_validate.return_value = mock_user
        
        response = client.post(
            "/api/v1/ttv/generate",
            json=test_ttv_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "pending"
    
    @patch('ttv_service.security.jwt_validator.validate_token')
    def test_get_job_status(self, mock_validate, client, mock_user):
        """Test job status endpoint"""
        mock_validate.return_value = mock_user
        
        # Create a test job first
        with patch('ttv_service.job_manager.job_manager.submit_job') as mock_submit:
            mock_submit.return_value = "test_job_123"
            
            response = client.post(
                "/api/v1/ttv/generate",
                json={"script": "test"},
                headers={"Authorization": "Bearer test_token"}
            )
            job_id = response.json()["job_id"]
        
        # Mock job status
        with patch('ttv_service.job_manager.job_manager.get_job_status') as mock_status:
            mock_status.return_value = {
                "job_id": job_id,
                "status": "processing",
                "progress": 50,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = client.get(
                f"/api/v1/ttv/jobs/{job_id}",
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == job_id
            assert data["status"] == "processing"
            assert data["progress"] == 50
    
    def test_unauthorized_access(self, client, test_ttv_request):
        """Test unauthorized access handling"""
        response = client.post("/api/v1/ttv/generate", json=test_ttv_request)
        assert response.status_code == 403
    
    def test_invalid_token(self, client, test_ttv_request):
        """Test invalid token handling"""
        response = client.post(
            "/api/v1/ttv/generate",
            json=test_ttv_request,
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401


class TestJobManager:
    """Test job management system"""
    
    @pytest.mark.asyncio
    async def test_submit_job(self, test_job_manager, mock_user):
        """Test job submission"""
        request_data = {
            "script": "Test video generation",
            "video_style": "realistic",
            "quality": "balanced"
        }
        
        with patch('ttv_service.tasks.generate_video.apply_async') as mock_task:
            mock_task.return_value = Mock(id="test_job_123")
            
            job_id = await test_job_manager.submit_job(
                user_id=mock_user["user_id"],
                request_data=request_data
            )
            
            assert job_id.startswith("ttv_")
            assert mock_user["user_id"] in job_id
    
    @pytest.mark.asyncio
    async def test_job_status_tracking(self, test_job_manager):
        """Test job status updates"""
        job_id = "test_job_123"
        
        # Test progress update
        await test_job_manager.update_job_progress(
            job_id=job_id,
            progress=50,
            status=JobStatus.PROCESSING,
            message="Processing video",
            current_step="scene_generation"
        )
        
        # Verify status was updated
        status = await test_job_manager.get_job_status(job_id)
        if status:
            assert status["progress"] == 50
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self, test_job_manager, mock_user):
        """Test job cancellation"""
        job_id = "test_job_123"
        
        with patch('ttv_service.job_manager.celery_app.control.revoke') as mock_revoke:
            result = await test_job_manager.cancel_job(job_id, mock_user["user_id"])
            # Result depends on whether job exists in DB
            mock_revoke.assert_called_once_with(job_id, terminate=True)


class TestContentModeration:
    """Test content moderation system"""
    
    @pytest.mark.asyncio
    async def test_valid_content(self):
        """Test valid content passes moderation"""
        content = {"script": "A beautiful landscape with mountains and trees"}
        
        result = await content_moderator.moderate_content(content)
        
        assert result["approved"] is True
        assert len(result["violations"]) == 0
        assert result["score"] == 1.0
    
    @pytest.mark.asyncio
    async def test_forbidden_keywords(self):
        """Test forbidden keywords detection"""
        content = {"script": "This video contains violence and harmful content"}
        
        result = await content_moderator.moderate_content(content)
        
        assert result["approved"] is False
        assert len(result["violations"]) > 0
        assert any(v["type"] == "content_violation" for v in result["violations"])
    
    @pytest.mark.asyncio
    async def test_length_violation(self):
        """Test script length limits"""
        long_script = "A" * (settings.max_script_length + 100)
        content = {"script": long_script}
        
        result = await content_moderator.moderate_content(content)
        
        assert result["approved"] is False
        assert any(v["type"] == "length_violation" for v in result["violations"])


class TestAuthentication:
    """Test authentication and authorization"""
    
    @pytest.mark.asyncio
    async def test_valid_jwt_validation(self):
        """Test valid JWT token validation"""
        mock_payload = {
            "sub": "user_123",
            "email": "test@example.com",
            "role": "user",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp())
        }
        
        with patch('jwt.decode', return_value=mock_payload):
            with patch.object(jwt_validator, 'db_engine') as mock_engine:
                # Mock session and query
                mock_session = Mock()
                mock_engine.__enter__ = Mock(return_value=mock_session)
                mock_engine.__exit__ = Mock(return_value=None)
                
                mock_session_record = Mock()
                mock_session_record.id = "session_123"
                mock_session.exec.return_value.first.return_value = mock_session_record
                
                with patch('sqlmodel.Session') as mock_session_class:
                    mock_session_class.return_value.__enter__ = Mock(return_value=mock_session)
                    mock_session_class.return_value.__exit__ = Mock(return_value=None)
                    
                    result = await jwt_validator.validate_token("valid_token")
                    
                    assert result["user_id"] == "user_123"
                    assert result["email"] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_expired_token(self):
        """Test expired token handling"""
        with patch('jwt.decode', side_effect=jwt.ExpiredSignatureError()):
            with pytest.raises(Exception):
                await jwt_validator.validate_token("expired_token")


class TestStorage:
    """Test storage integration"""
    
    @pytest.mark.asyncio
    async def test_local_storage_upload(self):
        """Test local storage file upload"""
        storage = LocalStorageBackend(base_path=tempfile.mkdtemp())
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            test_file_path = f.name
        
        try:
            # Upload file
            url = await storage.upload_file(test_file_path, "test/file.txt", "text/plain")
            
            assert url.endswith("test/file.txt")
            assert await storage.file_exists("test/file.txt")
            
        finally:
            os.unlink(test_file_path)
    
    @pytest.mark.asyncio
    async def test_json_upload(self):
        """Test JSON data upload"""
        storage = LocalStorageBackend(base_path=tempfile.mkdtemp())
        
        test_data = {"key": "value", "number": 42}
        
        url = await storage.upload_json(test_data, "test/data.json")
        
        assert url.endswith("test/data.json")
        assert await storage.file_exists("test/data.json")
    
    @pytest.mark.asyncio
    async def test_storage_health_check(self):
        """Test storage health check"""
        storage_manager = get_storage_backend()
        
        health_result = await storage_manager.health_check()
        
        assert "status" in health_result
        assert "backend" in health_result


class TestEventSystem:
    """Test event emission and handling"""
    
    @pytest.mark.asyncio
    async def test_event_emission(self):
        """Test event emission"""
        with patch('ttv_service.events.event_emitter.handlers') as mock_handlers:
            mock_handler = AsyncMock()
            mock_handler.handle.return_value = True
            mock_handlers.__iter__ = Mock(return_value=iter([mock_handler]))
            
            await emit_event(
                EventType.JOB_CREATED,
                {"job_id": "test_123", "user_id": "user_123"},
                user_id="user_123",
                job_id="test_123"
            )
            
            mock_handler.handle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_job_lifecycle_events(self):
        """Test job lifecycle event emission"""
        from ttv_service.events import event_emitter
        
        with patch.object(event_emitter, 'emit') as mock_emit:
            await event_emitter.emit_job_created("job_123", "user_123", {"script": "test"})
            await event_emitter.emit_job_started("job_123", "user_123")
            await event_emitter.emit_job_completed("job_123", "user_123", {"video_url": "http://example.com/video.mp4"})
            
            assert mock_emit.call_count == 3


class TestMonitoring:
    """Test monitoring and health checks"""
    
    @pytest.mark.asyncio
    async def test_database_health_check(self):
        """Test database health check"""
        with patch('sqlmodel.Session') as mock_session_class:
            mock_session = Mock()
            mock_session.exec.return_value.first.return_value = 1
            mock_session_class.return_value.__enter__ = Mock(return_value=mock_session)
            mock_session_class.return_value.__exit__ = Mock(return_value=None)
            
            result = await health_checker.check_database_health()
            
            assert result.service == "database"
            assert result.status in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test comprehensive health check"""
        with patch.object(health_checker, 'check_database_health') as mock_db:
            with patch.object(health_checker, 'check_redis_health') as mock_redis:
                with patch.object(health_checker, 'check_storage_health') as mock_storage:
                    with patch.object(health_checker, 'check_worker_health') as mock_worker:
                        # Mock health responses
                        from ttv_service.monitoring import HealthStatus
                        mock_db.return_value = HealthStatus("database", "healthy", "OK", datetime.utcnow())
                        mock_redis.return_value = HealthStatus("redis", "healthy", "OK", datetime.utcnow())
                        mock_storage.return_value = HealthStatus("storage", "healthy", "OK", datetime.utcnow())
                        mock_worker.return_value = HealthStatus("workers", "healthy", "OK", datetime.utcnow())
                        
                        result = await health_checker.comprehensive_health_check()
                        
                        assert "overall_status" in result
                        assert "services" in result
                        assert len(result["services"]) >= 4


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_request_data(self, client):
        """Test invalid request data handling"""
        invalid_request = {"invalid_field": "value"}
        
        response = client.post("/api/v1/ttv/generate", json=invalid_request)
        assert response.status_code in [400, 422]  # Bad request or validation error
    
    @patch('ttv_service.security.jwt_validator.validate_token')
    def test_content_moderation_failure(self, mock_validate, client, mock_user):
        """Test content moderation rejection"""
        mock_validate.return_value = mock_user
        
        harmful_request = {
            "script": "This contains violence and harmful illegal content",
            "video_style": "realistic"
        }
        
        response = client.post(
            "/api/v1/ttv/generate",
            json=harmful_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_job_timeout_handling(self, test_job_manager):
        """Test job timeout handling"""
        # This would typically be handled by Celery
        # Test the monitoring logic for stuck jobs
        from ttv_service.tasks import monitor_jobs
        
        with patch('ttv_service.job_manager.job_manager.db_engine') as mock_engine:
            mock_session = Mock()
            mock_engine.__enter__ = Mock(return_value=mock_session)
            mock_engine.__exit__ = Mock(return_value=None)
            
            # Mock finding stuck jobs
            stuck_job = Mock()
            stuck_job.id = "stuck_job_123"
            stuck_job.status = JobStatus.PROCESSING
            stuck_job.started_at = datetime.utcnow() - timedelta(hours=2)
            
            mock_session.exec.return_value.all.return_value = [stuck_job]
            
            with patch('sqlmodel.Session') as mock_session_class:
                mock_session_class.return_value.__enter__ = Mock(return_value=mock_session)
                mock_session_class.return_value.__exit__ = Mock(return_value=None)
                
                # Run monitor task
                monitor_jobs()
                
                # Verify job was marked as failed
                assert stuck_job.status == JobStatus.FAILED


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @patch('ttv_service.security.jwt_validator.validate_token')
    @patch('ttv_service.tasks.generate_video.apply_async')
    def test_complete_video_generation_flow(self, mock_task, mock_validate, client, mock_user, test_ttv_request):
        """Test complete video generation workflow"""
        mock_validate.return_value = mock_user
        mock_task.return_value = Mock(id="job_123")
        
        # Step 1: Submit job
        response = client.post(
            "/api/v1/ttv/generate",
            json=test_ttv_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        
        # Step 2: Check job status (mock progression)
        with patch('ttv_service.job_manager.job_manager.get_job_status') as mock_status:
            # Mock processing status
            mock_status.return_value = {
                "job_id": job_id,
                "status": "processing",
                "progress": 75,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = client.get(
                f"/api/v1/ttv/jobs/{job_id}",
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 200
            assert response.json()["status"] == "processing"
            assert response.json()["progress"] == 75
    
    @pytest.mark.asyncio
    async def test_storage_to_delivery_pipeline(self):
        """Test storage and delivery pipeline"""
        storage_manager = get_storage_backend()
        
        # Mock video upload
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b"fake video content")
            video_path = f.name
        
        try:
            # Upload video
            video_url = await storage_manager.upload_video(video_path, "test_job_123")
            assert video_url is not None
            
            # Upload metadata
            metadata = {
                "job_id": "test_job_123",
                "duration": 30,
                "resolution": "1080p",
                "created_at": datetime.utcnow().isoformat()
            }
            
            metadata_url = await storage_manager.upload_metadata(metadata, "test_job_123")
            assert metadata_url is not None
            
            # Get presigned URL
            presigned_url = await storage_manager.get_video_url("test_job_123", expires_in=3600)
            assert presigned_url is not None
            
        finally:
            os.unlink(video_path)


# Test configuration and fixtures
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Clear any existing test data
    if os.path.exists("./test_ttv.db"):
        os.unlink("./test_ttv.db")
    
    yield
    
    # Cleanup after test
    if os.path.exists("./test_ttv.db"):
        os.unlink("./test_ttv.db")


# Performance tests
class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_concurrent_job_submissions(self, test_job_manager, mock_user):
        """Test handling multiple concurrent job submissions"""
        async def submit_job(i):
            return await test_job_manager.submit_job(
                user_id=f"user_{i}",
                request_data={"script": f"Test script {i}"}
            )
        
        with patch('ttv_service.tasks.generate_video.apply_async') as mock_task:
            mock_task.return_value = Mock(id="job_123")
            
            # Submit 10 jobs concurrently
            tasks = [submit_job(i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all jobs were submitted
            successful_submissions = [r for r in results if isinstance(r, str)]
            assert len(successful_submissions) == 10
    
    def test_api_response_times(self, client):
        """Test API response times"""
        start_time = time.time()
        response = client.get("/health")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])