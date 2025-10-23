"""
TTV Service Storage Integration
Handles file storage for bhiv_bucket compatibility and multi-backend support
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import mimetypes
import uuid

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .config import settings


logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    async def upload_file(self, file_path: str, key: str, content_type: str = None) -> str:
        """Upload a file and return its URL"""
        pass
    
    @abstractmethod
    async def upload_json(self, data: Dict[str, Any], key: str) -> str:
        """Upload JSON data and return its URL"""
        pass
    
    @abstractmethod
    async def delete_file(self, key: str) -> bool:
        """Delete a file"""
        pass
    
    @abstractmethod
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Get a pre-signed URL for file access"""
        pass
    
    @abstractmethod
    async def file_exists(self, key: str) -> bool:
        """Check if a file exists"""
        pass


class LocalStorageBackend(StorageBackend):
    """Local file system storage backend"""
    
    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.base_url = f"http://{settings.host}:{settings.port}/storage"
    
    async def upload_file(self, file_path: str, key: str, content_type: str = None) -> str:
        """Upload file to local storage"""
        try:
            dest_path = self.base_path / key
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            import shutil
            shutil.copy2(file_path, dest_path)
            
            # Return public URL
            return f"{self.base_url}/{key}"
            
        except Exception as e:
            logger.error(f"Error uploading file to local storage: {str(e)}")
            raise
    
    async def upload_json(self, data: Dict[str, Any], key: str) -> str:
        """Upload JSON to local storage"""
        try:
            dest_path = self.base_path / key
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return f"{self.base_url}/{key}"
            
        except Exception as e:
            logger.error(f"Error uploading JSON to local storage: {str(e)}")
            raise
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from local storage"""
        try:
            file_path = self.base_path / key
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting file from local storage: {str(e)}")
            return False
    
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Get URL for local file (no pre-signing needed)"""
        return f"{self.base_url}/{key}"
    
    async def file_exists(self, key: str) -> bool:
        """Check if file exists in local storage"""
        return (self.base_path / key).exists()


class S3StorageBackend(StorageBackend):
    """Amazon S3 storage backend"""
    
    def __init__(self):
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.s3_region
            )
            self.bucket_name = settings.s3_bucket_name
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Error initializing S3 backend: {str(e)}")
            raise
    
    async def upload_file(self, file_path: str, key: str, content_type: str = None) -> str:
        """Upload file to S3"""
        try:
            if not content_type:
                content_type, _ = mimetypes.guess_type(file_path)
                content_type = content_type or 'application/octet-stream'
            
            extra_args = {'ContentType': content_type}
            
            self.s3_client.upload_file(
                file_path, 
                self.bucket_name, 
                key, 
                ExtraArgs=extra_args
            )
            
            # Return public URL
            return f"https://{self.bucket_name}.s3.{settings.s3_region}.amazonaws.com/{key}"
            
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise
    
    async def upload_json(self, data: Dict[str, Any], key: str) -> str:
        """Upload JSON to S3"""
        try:
            json_content = json.dumps(data, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_content,
                ContentType='application/json'
            )
            
            return f"https://{self.bucket_name}.s3.{settings.s3_region}.amazonaws.com/{key}"
            
        except Exception as e:
            logger.error(f"Error uploading JSON to S3: {str(e)}")
            raise
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False
    
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Get pre-signed URL for S3 object"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
            
        except Exception as e:
            logger.error(f"Error generating pre-signed URL: {str(e)}")
            raise
    
    async def file_exists(self, key: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise


class SupabaseStorageBackend(StorageBackend):
    """Supabase storage backend"""
    
    def __init__(self):
        try:
            from supabase import create_client, Client
            
            if not settings.supabase_url or not settings.supabase_key:
                raise ValueError("Supabase URL and key must be configured")
            
            self.supabase: Client = create_client(
                settings.supabase_url, 
                settings.supabase_key
            )
            self.bucket_name = "ttv-storage"
            
            # Create bucket if it doesn't exist
            try:
                self.supabase.storage.create_bucket(self.bucket_name)
            except Exception:
                # Bucket might already exist
                pass
                
        except Exception as e:
            logger.error(f"Error initializing Supabase backend: {str(e)}")
            raise
    
    async def upload_file(self, file_path: str, key: str, content_type: str = None) -> str:
        """Upload file to Supabase Storage"""
        try:
            if not content_type:
                content_type, _ = mimetypes.guess_type(file_path)
                content_type = content_type or 'application/octet-stream'
            
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            result = self.supabase.storage.from_(self.bucket_name).upload(
                key, 
                file_content,
                file_options={"content-type": content_type}
            )
            
            if result.error:
                raise Exception(f"Supabase upload error: {result.error}")
            
            # Get public URL
            public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(key)
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading file to Supabase: {str(e)}")
            raise
    
    async def upload_json(self, data: Dict[str, Any], key: str) -> str:
        """Upload JSON to Supabase Storage"""
        try:
            json_content = json.dumps(data, indent=2).encode('utf-8')
            
            result = self.supabase.storage.from_(self.bucket_name).upload(
                key, 
                json_content,
                file_options={"content-type": "application/json"}
            )
            
            if result.error:
                raise Exception(f"Supabase upload error: {result.error}")
            
            public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(key)
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading JSON to Supabase: {str(e)}")
            raise
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from Supabase Storage"""
        try:
            result = self.supabase.storage.from_(self.bucket_name).remove([key])
            return not result.error
            
        except Exception as e:
            logger.error(f"Error deleting file from Supabase: {str(e)}")
            return False
    
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Get signed URL from Supabase (expires_in in seconds)"""
        try:
            result = self.supabase.storage.from_(self.bucket_name).create_signed_url(
                key, 
                expires_in
            )
            
            if result.error:
                raise Exception(f"Supabase signed URL error: {result.error}")
            
            return result.signed_url
            
        except Exception as e:
            logger.error(f"Error generating signed URL: {str(e)}")
            raise
    
    async def file_exists(self, key: str) -> bool:
        """Check if file exists in Supabase Storage"""
        try:
            result = self.supabase.storage.from_(self.bucket_name).list(key)
            return len(result) > 0 and not result.error
        except Exception:
            return False


class BHIVBucketStorageBackend(StorageBackend):
    """
    BHIV Bucket storage backend
    Adapts to the existing bhiv_bucket system from Ashmit's repository
    """
    
    def __init__(self):
        # Initialize based on BHIV storage configuration
        self.backend_type = settings.bhiv_storage_backend
        
        if self.backend_type == "supabase":
            self.backend = SupabaseStorageBackend()
        elif self.backend_type == "s3":
            self.backend = S3StorageBackend()
        else:
            self.backend = LocalStorageBackend(settings.bhiv_bucket_path)
        
        logger.info(f"Initialized BHIV bucket with {self.backend_type} backend")
    
    async def upload_file(self, file_path: str, key: str, content_type: str = None) -> str:
        """Upload file using BHIV bucket patterns"""
        # Add BHIV-specific path structure
        bhiv_key = f"ttv/{datetime.utcnow().strftime('%Y/%m/%d')}/{key}"
        
        url = await self.backend.upload_file(file_path, bhiv_key, content_type)
        
        # Log to BHIV audit system (if available)
        await self._log_storage_action("upload", bhiv_key, {"url": url})
        
        return url
    
    async def upload_json(self, data: Dict[str, Any], key: str) -> str:
        """Upload JSON using BHIV bucket patterns"""
        # Add BHIV metadata
        bhiv_data = {
            **data,
            "bhiv_metadata": {
                "service": "ttv",
                "uploaded_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
        }
        
        bhiv_key = f"ttv/{datetime.utcnow().strftime('%Y/%m/%d')}/{key}"
        
        url = await self.backend.upload_json(bhiv_data, bhiv_key)
        
        await self._log_storage_action("upload_json", bhiv_key, {"url": url})
        
        return url
    
    async def delete_file(self, key: str) -> bool:
        """Delete file using BHIV bucket patterns"""
        result = await self.backend.delete_file(key)
        
        await self._log_storage_action("delete", key, {"success": result})
        
        return result
    
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Get pre-signed URL using BHIV bucket patterns"""
        url = await self.backend.get_presigned_url(key, expires_in)
        
        await self._log_storage_action("presigned_url", key, {"expires_in": expires_in})
        
        return url
    
    async def file_exists(self, key: str) -> bool:
        """Check if file exists using BHIV bucket patterns"""
        return await self.backend.file_exists(key)
    
    async def _log_storage_action(self, action: str, key: str, metadata: Dict[str, Any]):
        """Log storage actions to BHIV audit system"""
        try:
            # This would integrate with BHIV's audit logging system
            # For now, just log locally
            logger.info(f"BHIV Storage Action: {action} on {key}", extra={
                "action": action,
                "key": key,
                "metadata": metadata,
                "service": "ttv",
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Error logging storage action: {str(e)}")


class StorageManager:
    """Manages storage operations with automatic backend selection"""
    
    def __init__(self):
        self.backend = self._get_backend()
        logger.info(f"Initialized storage manager with {settings.storage_backend} backend")
    
    def _get_backend(self) -> StorageBackend:
        """Get the configured storage backend"""
        backend_type = settings.storage_backend.lower()
        
        if backend_type == "local":
            return LocalStorageBackend()
        elif backend_type == "s3":
            return S3StorageBackend()
        elif backend_type == "supabase":
            return SupabaseStorageBackend()
        elif backend_type == "bhiv_bucket":
            return BHIVBucketStorageBackend()
        else:
            logger.warning(f"Unknown storage backend {backend_type}, falling back to local")
            return LocalStorageBackend()
    
    async def upload_video(self, file_path: str, job_id: str) -> str:
        """Upload a video file with TTV-specific naming"""
        key = f"{job_id}/video.mp4"
        return await self.backend.upload_file(file_path, key, "video/mp4")
    
    async def upload_metadata(self, metadata: Dict[str, Any], job_id: str) -> str:
        """Upload job metadata"""
        key = f"{job_id}/metadata.json"
        return await self.backend.upload_json(metadata, key)
    
    async def upload_thumbnail(self, file_path: str, job_id: str) -> str:
        """Upload video thumbnail"""
        key = f"{job_id}/thumbnail.jpg"
        return await self.backend.upload_file(file_path, key, "image/jpeg")
    
    async def get_video_url(self, job_id: str, expires_in: int = 3600) -> str:
        """Get pre-signed URL for video access"""
        key = f"{job_id}/video.mp4"
        return await self.backend.get_presigned_url(key, expires_in)
    
    async def cleanup_job_files(self, job_id: str) -> bool:
        """Clean up all files for a job"""
        success = True
        
        for file_type in ["video.mp4", "metadata.json", "thumbnail.jpg"]:
            key = f"{job_id}/{file_type}"
            if await self.backend.file_exists(key):
                result = await self.backend.delete_file(key)
                success = success and result
        
        return success
    
    async def health_check(self) -> Dict[str, Any]:
        """Check storage backend health"""
        try:
            # Try to upload a small test file
            test_key = f"health_check/{uuid.uuid4().hex}.txt"
            test_data = {"health_check": True, "timestamp": datetime.utcnow().isoformat()}
            
            url = await self.backend.upload_json(test_data, test_key)
            
            # Clean up test file
            await self.backend.delete_file(test_key)
            
            return {
                "status": "healthy",
                "backend": settings.storage_backend,
                "test_upload_success": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": settings.storage_backend,
                "error": str(e)
            }


# Global storage manager instance
def get_storage_backend() -> StorageManager:
    """Get the configured storage manager"""
    return StorageManager()


# Convenience functions for backward compatibility
async def upload_file(file_path: str, key: str, content_type: str = None) -> str:
    """Upload a file using the configured storage backend"""
    storage = get_storage_backend()
    return await storage.backend.upload_file(file_path, key, content_type)


async def upload_json(data: Dict[str, Any], key: str) -> str:
    """Upload JSON data using the configured storage backend"""
    storage = get_storage_backend()
    return await storage.backend.upload_json(data, key)