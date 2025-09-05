"""
NAS Storage Manager for Task 4 Day 3
Handles NAS write/read operations and signed URL generation
"""

import os
import shutil
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from urllib.parse import quote
import tempfile
import socket


class NASStorageManager:
    """Manages NAS storage operations with signed URL support"""

    def __init__(self, nas_path: str = r"\\192.168.0.94\Shashank\Cached_TTV",
                 secret_key: str = "shashank_ttv_secret_2024",
                 url_expiry_seconds: int = 3600):
        self.nas_path = Path(nas_path)
        self.secret_key = secret_key
        self.url_expiry_seconds = url_expiry_seconds

        # Create local cache directory for faster access
        self.local_cache_dir = Path("cache/nas_cache")
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Test NAS connectivity
        self.nas_available = self._test_nas_connectivity()

    def _test_nas_connectivity(self) -> bool:
        """Test if NAS is accessible"""
        try:
            if self.nas_path.exists():
                # Try to create a test file
                test_file = self.nas_path / "test_connection.tmp"
                test_file.write_text("test")
                test_file.unlink()
                return True
            else:
                print(f"Warning: NAS path {self.nas_path} not accessible")
                return False
        except Exception as e:
            print(f"Warning: NAS connectivity test failed: {e}")
            return False

    def _generate_signature(self, file_path: str, expiry: int) -> str:
        """Generate HMAC signature for signed URL"""
        message = f"{file_path}:{expiry}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_local_cache_path(self, nas_path: Path) -> Path:
        """Get local cache path for NAS file"""
        # Create a hash of the NAS path for local cache
        path_hash = hashlib.md5(str(nas_path).encode()).hexdigest()[:8]
        return self.local_cache_dir / f"{path_hash}_{nas_path.name}"

    def write_file(self, local_path: str, nas_filename: str,
                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Write file to NAS with metadata

        Args:
            local_path: Path to local file
            nas_filename: Filename on NAS
            metadata: Optional metadata to store

        Returns:
            Dict with operation results
        """
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")

            # Create NAS destination path
            nas_file_path = self.nas_path / nas_filename

            # Copy file to NAS
            shutil.copy2(local_file, nas_file_path)

            # Store metadata if provided
            if metadata:
                metadata_file = nas_file_path.with_suffix('.meta.json')
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            # Also cache locally for faster future access
            local_cache_path = self._get_local_cache_path(nas_file_path)
            shutil.copy2(local_file, local_cache_path)

            # Generate signed URL
            signed_url = self.generate_signed_url(nas_filename)

            return {
                "success": True,
                "nas_path": str(nas_file_path),
                "signed_url": signed_url,
                "file_size": local_file.stat().st_size,
                "metadata": metadata
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "local_path": local_path,
                "nas_filename": nas_filename
            }

    def read_file(self, nas_filename: str, local_destination: Optional[str] = None) -> Dict[str, Any]:
        """
        Read file from NAS to local destination

        Args:
            nas_filename: Filename on NAS
            local_destination: Local path to save file (optional)

        Returns:
            Dict with operation results
        """
        try:
            nas_file_path = self.nas_path / nas_filename

            # Check local cache first
            local_cache_path = self._get_local_cache_path(nas_file_path)
            if local_cache_path.exists():
                # Use cached version
                if local_destination:
                    shutil.copy2(local_cache_path, local_destination)
                return {
                    "success": True,
                    "source": "cache",
                    "nas_path": str(nas_file_path),
                    "local_path": local_destination or str(local_cache_path),
                    "file_size": local_cache_path.stat().st_size
                }

            # Read from NAS
            if not nas_file_path.exists():
                raise FileNotFoundError(f"NAS file not found: {nas_filename}")

            # Determine local destination
            if not local_destination:
                local_destination = str(self.local_cache_dir / nas_filename)

            # Copy from NAS to local
            shutil.copy2(nas_file_path, local_destination)

            # Update local cache
            shutil.copy2(nas_file_path, local_cache_path)

            return {
                "success": True,
                "source": "nas",
                "nas_path": str(nas_file_path),
                "local_path": local_destination,
                "file_size": nas_file_path.stat().st_size
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "nas_filename": nas_filename,
                "local_destination": local_destination
            }

    def generate_signed_url(self, nas_filename: str, expiry_seconds: Optional[int] = None) -> str:
        """
        Generate signed URL for secure NAS file access

        Args:
            nas_filename: Filename on NAS
            expiry_seconds: URL expiry time (optional)

        Returns:
            Signed URL string
        """
        if expiry_seconds is None:
            expiry_seconds = self.url_expiry_seconds

        expiry = int(time.time()) + expiry_seconds

        # Create the URL path
        file_path = f"/nas/{quote(nas_filename)}"

        # Generate signature
        signature = self._generate_signature(file_path, expiry)

        # Create signed URL
        signed_url = f"http://localhost:8001{nas_filename}?expiry={expiry}&signature={signature}"

        return signed_url

    def validate_signed_url(self, nas_filename: str, expiry: int, signature: str) -> bool:
        """
        Validate signed URL signature

        Args:
            nas_filename: Filename on NAS
            expiry: Expiry timestamp
            signature: Provided signature

        Returns:
            True if signature is valid
        """
        # Check if URL has expired
        if int(time.time()) > expiry:
            return False

        # Recreate signature
        file_path = f"/nas/{quote(nas_filename)}"
        expected_signature = self._generate_signature(file_path, expiry)

        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)

    def list_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """
        List files on NAS matching pattern

        Args:
            pattern: File pattern to match

        Returns:
            List of file information
        """
        try:
            files_info = []
            for file_path in self.nas_path.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    files_info.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "nas_path": str(file_path)
                    })
            return files_info
        except Exception as e:
            print(f"Warning: Failed to list NAS files: {e}")
            return []

    def get_file_metadata(self, nas_filename: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for NAS file

        Args:
            nas_filename: Filename on NAS

        Returns:
            Metadata dict or None
        """
        try:
            metadata_file = self.nas_path / f"{nas_filename}.meta.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read metadata for {nas_filename}: {e}")

        return None

    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """
        Clean up old files from local cache

        Args:
            max_age_hours: Maximum age of files to keep

        Returns:
            Number of files cleaned up
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0

            for cache_file in self.local_cache_dir.glob("*"):
                if cache_file.is_file():
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        cache_file.unlink()
                        cleaned_count += 1

            return cleaned_count
        except Exception as e:
            print(f"Warning: Cache cleanup failed: {e}")
            return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get NAS storage statistics"""
        try:
            total_files = 0
            total_size = 0

            for file_path in self.nas_path.glob("*"):
                if file_path.is_file() and not file_path.name.endswith('.meta.json'):
                    total_files += 1
                    total_size += file_path.stat().st_size

            cache_files = len(list(self.local_cache_dir.glob("*")))
            cache_size = sum(f.stat().st_size for f in self.local_cache_dir.glob("*") if f.is_file())

            return {
                "nas_available": self.nas_available,
                "nas_path": str(self.nas_path),
                "total_files": total_files,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_files": cache_files,
                "cache_size_mb": cache_size / (1024 * 1024)
            }
        except Exception as e:
            return {
                "error": str(e),
                "nas_available": self.nas_available,
                "nas_path": str(self.nas_path)
            }


# Global NAS storage instance
_nas_storage = None

def get_nas_storage() -> NASStorageManager:
    """Get global NAS storage instance"""
    global _nas_storage
    if _nas_storage is None:
        _nas_storage = NASStorageManager()
    return _nas_storage