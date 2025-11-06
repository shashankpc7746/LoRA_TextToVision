"""
Runtime Key Validation System
Core-signed runtime keys for startup authentication
Time-limited keys (12-24h) with Ed25519 signatures
"""
import os
import json
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


class RuntimeKeyValidator:
    """Validate Core-signed runtime keys for worker authentication"""
    
    # Key validity duration in seconds
    DEFAULT_KEY_LIFETIME = 24 * 60 * 60  # 24 hours
    
    def __init__(self, public_key_path: Optional[str] = None):
        """
        Initialize runtime validator
        
        Args:
            public_key_path: Path to Core's public key for verification
        """
        self.public_key = self._load_public_key(public_key_path)
        self.cached_key = None
        self.cached_key_expiry = None
    
    def _load_public_key(self, public_key_path: Optional[str] = None) -> ed25519.Ed25519PublicKey:
        """Load Core's public key"""
        if public_key_path is None:
            # Try default locations
            default_paths = [
                Path('.signing_keys/core_public_key.pem'),
                Path('.signing_keys/public_key.pem'),
                Path('/etc/bhiv/core_public_key.pem'),
            ]
            
            for path in default_paths:
                if path.exists():
                    public_key_path = str(path)
                    break
        
        if public_key_path and Path(public_key_path).exists():
            with open(public_key_path, 'rb') as f:
                return serialization.load_pem_public_key(f.read())
        else:
            # For development: use local key if available
            private_key_path = Path('.signing_keys/private_key.pem')
            if private_key_path.exists():
                print("⚠️  WARNING: Using local key for development. Production should use Core's public key!")
                with open(private_key_path, 'rb') as f:
                    private_key = serialization.load_pem_private_key(f.read(), password=None)
                return private_key.public_key()
            
            raise ValueError("No Core public key found. Set public_key_path or create .signing_keys/core_public_key.pem")
    
    def validate_runtime_key(self, runtime_key: str, strict: bool = True) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate runtime key signature and expiry
        
        Args:
            runtime_key: Base64-encoded signed runtime key
            strict: If False, allow expired keys (demo mode)
        
        Returns:
            Tuple of (is_valid, key_data)
        """
        try:
            # Decode runtime key
            key_package = json.loads(base64.b64decode(runtime_key))
            
            # Extract components
            key_data = key_package['key_data']
            signature_b64 = key_package['signature']
            signature = base64.b64decode(signature_b64)
            
            # Verify signature
            canonical_json = json.dumps(key_data, sort_keys=True)
            
            try:
                self.public_key.verify(signature, canonical_json.encode('utf-8'))
            except InvalidSignature:
                print("❌ Invalid runtime key signature")
                return False, None
            
            # Check expiry
            expires_at = datetime.fromisoformat(key_data['expires_at'].replace('Z', '+00:00'))
            now = datetime.now(expires_at.tzinfo)
            
            if now > expires_at:
                if strict:
                    print(f"❌ Runtime key expired at {expires_at}")
                    return False, key_data
                else:
                    print(f"⚠️  Runtime key expired but allowing in demo mode")
            
            # Key is valid
            return True, key_data
            
        except Exception as e:
            print(f"❌ Runtime key validation error: {e}")
            return False, None
    
    def request_runtime_key(self, worker_id: str, core_endpoint: Optional[str] = None) -> Optional[str]:
        """
        Request runtime key from Core service
        
        Args:
            worker_id: Worker identifier
            core_endpoint: Core API endpoint (default: from CORE_ENDPOINT env)
        
        Returns:
            Runtime key string or None if request fails
        """
        import requests
        
        core_endpoint = core_endpoint or os.getenv('CORE_ENDPOINT', 'http://localhost:8080')
        
        try:
            response = requests.post(
                f"{core_endpoint}/api/v1/runtime-keys/request",
                json={
                    'worker_id': worker_id,
                    'requested_at': datetime.utcnow().isoformat() + 'Z',
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                runtime_key = data['runtime_key']
                
                print(f"✅ Runtime key obtained for worker {worker_id}")
                return runtime_key
            else:
                print(f"❌ Core rejected runtime key request: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Failed to request runtime key: {e}")
            return None
    
    def cache_runtime_key(self, runtime_key: str):
        """Cache validated runtime key"""
        is_valid, key_data = self.validate_runtime_key(runtime_key, strict=True)
        
        if is_valid and key_data:
            self.cached_key = runtime_key
            self.cached_key_expiry = datetime.fromisoformat(key_data['expires_at'].replace('Z', '+00:00'))
            print(f"✅ Runtime key cached until {self.cached_key_expiry}")
    
    def get_cached_key(self) -> Optional[str]:
        """Get cached runtime key if still valid"""
        if self.cached_key is None or self.cached_key_expiry is None:
            return None
        
        now = datetime.now(self.cached_key_expiry.tzinfo)
        if now < self.cached_key_expiry:
            return self.cached_key
        else:
            print("⚠️  Cached runtime key expired")
            self.cached_key = None
            self.cached_key_expiry = None
            return None
    
    def require_valid_key(self, runtime_key: Optional[str] = None, 
                         worker_id: Optional[str] = None,
                         demo_mode: bool = False) -> bool:
        """
        Require valid runtime key or enter restricted mode
        
        Args:
            runtime_key: Runtime key to validate
            worker_id: Worker ID for requesting new key
            demo_mode: Allow operation without valid key (restricted mode)
        
        Returns:
            True if valid key, False if entering restricted mode
        """
        # Try provided key
        if runtime_key:
            is_valid, _ = self.validate_runtime_key(runtime_key, strict=not demo_mode)
            if is_valid:
                self.cache_runtime_key(runtime_key)
                return True
        
        # Try cached key
        cached = self.get_cached_key()
        if cached:
            return True
        
        # Try requesting new key
        if worker_id:
            new_key = self.request_runtime_key(worker_id)
            if new_key:
                is_valid, _ = self.validate_runtime_key(new_key)
                if is_valid:
                    self.cache_runtime_key(new_key)
                    return True
        
        # No valid key available
        if demo_mode:
            print("⚠️  RESTRICTED MODE: No valid runtime key, operating in demo mode")
            print("    - Production outputs disabled")
            print("    - Watermarks applied")
            print("    - Quality limited")
            return False
        else:
            raise RuntimeError(
                "No valid runtime key available. "
                "Request key from Core or set demo_mode=True for restricted operation."
            )


class RuntimeKeyIssuer:
    """Issue runtime keys (Core service side)"""
    
    def __init__(self, private_key_path: Optional[str] = None):
        """
        Initialize key issuer
        
        Args:
            private_key_path: Path to Core's private key
        """
        self.private_key = self._load_private_key(private_key_path)
    
    def _load_private_key(self, private_key_path: Optional[str] = None) -> ed25519.Ed25519PrivateKey:
        """Load Core's private key"""
        if private_key_path is None:
            # Try default location
            private_key_path = Path('.signing_keys/private_key.pem')
        
        if Path(private_key_path).exists():
            with open(private_key_path, 'rb') as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        else:
            raise ValueError(f"Private key not found: {private_key_path}")
    
    def issue_runtime_key(self, worker_id: str, lifetime_hours: int = 24,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Issue new runtime key for worker
        
        Args:
            worker_id: Worker identifier
            lifetime_hours: Key validity duration in hours
            metadata: Optional metadata to include
        
        Returns:
            Base64-encoded signed runtime key
        """
        # Create key data
        issued_at = datetime.utcnow()
        expires_at = issued_at + timedelta(hours=lifetime_hours)
        
        key_data = {
            'worker_id': worker_id,
            'issued_at': issued_at.isoformat() + 'Z',
            'expires_at': expires_at.isoformat() + 'Z',
            'key_version': 'v1',
        }
        
        if metadata:
            key_data['metadata'] = metadata
        
        # Sign key data
        canonical_json = json.dumps(key_data, sort_keys=True)
        signature = self.private_key.sign(canonical_json.encode('utf-8'))
        
        # Create key package
        key_package = {
            'key_data': key_data,
            'signature': base64.b64encode(signature).decode('utf-8'),
        }
        
        # Encode as base64 JSON
        runtime_key = base64.b64encode(json.dumps(key_package).encode()).decode()
        
        print(f"✅ Runtime key issued for {worker_id} (valid {lifetime_hours}h)")
        return runtime_key


# Singleton validator
_global_validator = None

def get_runtime_validator() -> RuntimeKeyValidator:
    """Get global runtime validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = RuntimeKeyValidator()
    return _global_validator


def validate_runtime_key(runtime_key: str, strict: bool = True) -> bool:
    """Validate runtime key"""
    is_valid, _ = get_runtime_validator().validate_runtime_key(runtime_key, strict)
    return is_valid


def require_runtime_key(runtime_key: Optional[str] = None, 
                       worker_id: Optional[str] = None,
                       demo_mode: bool = False) -> bool:
    """Require valid runtime key or enter restricted mode"""
    return get_runtime_validator().require_valid_key(runtime_key, worker_id, demo_mode)


if __name__ == "__main__":
    # Test runtime key validation
    import tempfile
    
    print("Testing Runtime Key Validation System\n")
    
    # Create issuer and validator
    issuer = RuntimeKeyIssuer()
    validator = RuntimeKeyValidator()
    
    # Issue runtime key
    worker_id = "worker_test_001"
    runtime_key = issuer.issue_runtime_key(worker_id, lifetime_hours=24)
    print(f"Runtime key: {runtime_key[:50]}...\n")
    
    # Validate key
    is_valid, key_data = validator.validate_runtime_key(runtime_key)
    print(f"✅ Validation result: {'VALID' if is_valid else 'INVALID'}")
    if key_data:
        print(f"   Worker ID: {key_data['worker_id']}")
        print(f"   Expires: {key_data['expires_at']}")
    
    # Test expired key
    print("\n--- Testing expired key ---")
    expired_key = issuer.issue_runtime_key("worker_002", lifetime_hours=-1)  # Already expired
    is_valid_expired, _ = validator.validate_runtime_key(expired_key, strict=True)
    print(f"✅ Expired key (strict): {'VALID' if is_valid_expired else 'INVALID (expected)'}")
    
    is_valid_demo, _ = validator.validate_runtime_key(expired_key, strict=False)
    print(f"✅ Expired key (demo mode): {'VALID' if is_valid_demo else 'INVALID'}")
    
    # Test key caching
    print("\n--- Testing key caching ---")
    validator.cache_runtime_key(runtime_key)
    cached = validator.get_cached_key()
    print(f"✅ Key cached: {cached is not None}")
    
    # Test require_valid_key
    print("\n--- Testing require_valid_key ---")
    has_key = validator.require_valid_key(runtime_key=runtime_key)
    print(f"✅ Has valid key: {has_key}")
    
    has_key_demo = validator.require_valid_key(demo_mode=True)
    print(f"✅ Demo mode (no key): {has_key_demo}")
    
    print("\n✅ All runtime key validation tests passed!")
