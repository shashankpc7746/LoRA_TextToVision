"""
Artifact Signing Infrastructure
Cryptographic signing for models, checkpoints, and other artifacts
Uses Ed25519 for fast, secure signatures
"""
import os
import json
import hashlib
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


class ArtifactSigner:
    """Sign and verify artifacts with Ed25519 cryptographic signatures"""
    
    def __init__(self, private_key_path: Optional[str] = None, public_key_path: Optional[str] = None):
        """
        Initialize artifact signer
        
        Args:
            private_key_path: Path to Ed25519 private key (for signing)
            public_key_path: Path to Ed25519 public key (for verification)
        """
        self.private_key = None
        self.public_key = None
        
        # Load or generate keys
        if private_key_path and Path(private_key_path).exists():
            self.private_key = self._load_private_key(private_key_path)
        elif not private_key_path:
            # Try default location
            default_private = Path('.signing_keys/private_key.pem')
            if default_private.exists():
                self.private_key = self._load_private_key(str(default_private))
        
        if public_key_path and Path(public_key_path).exists():
            self.public_key = self._load_public_key(public_key_path)
        elif not public_key_path:
            # Try default location
            default_public = Path('.signing_keys/public_key.pem')
            if default_public.exists():
                self.public_key = self._load_public_key(str(default_public))
        
        # If no keys found, generate new pair (development only)
        if self.private_key is None and self.public_key is None:
            self._generate_key_pair()
    
    def _generate_key_pair(self):
        """Generate new Ed25519 key pair (development only)"""
        print("⚠️  WARNING: Generating new signing keys. Use secure key storage in production!")
        
        # Generate keys
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
        # Create .signing_keys directory
        key_dir = Path('.signing_keys')
        key_dir.mkdir(exist_ok=True)
        
        # Save private key (encrypted would be better)
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(key_dir / 'private_key.pem', 'wb') as f:
            f.write(private_pem)
        
        # Save public key
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(key_dir / 'public_key.pem', 'wb') as f:
            f.write(public_pem)
        
        # Add to .gitignore
        gitignore = Path('.gitignore')
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                content = f.read()
            if '.signing_keys' not in content:
                with open(gitignore, 'a') as f:
                    f.write('\n# Artifact signing keys (DO NOT COMMIT)\n.signing_keys/\n')
        
        print(f"✅ Generated new key pair in {key_dir}/")
    
    def _load_private_key(self, path: str) -> ed25519.Ed25519PrivateKey:
        """Load Ed25519 private key from PEM file"""
        with open(path, 'rb') as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    
    def _load_public_key(self, path: str) -> ed25519.Ed25519PublicKey:
        """Load Ed25519 public key from PEM file"""
        with open(path, 'rb') as f:
            return serialization.load_pem_public_key(f.read())
    
    def compute_artifact_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        """
        Compute cryptographic hash of artifact
        
        Args:
            file_path: Path to artifact file
            algorithm: Hash algorithm (sha256, sha512)
        
        Returns:
            Hexadecimal hash digest
        """
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            while chunk := f.read(8192):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def sign_artifact(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sign artifact with private key
        
        Args:
            file_path: Path to artifact to sign
            metadata: Optional metadata to include in signature
        
        Returns:
            Signature package with hash, signature, and metadata
        """
        if self.private_key is None:
            raise ValueError("No private key available for signing")
        
        # Compute artifact hash
        artifact_hash = self.compute_artifact_hash(file_path)
        
        # Prepare signing data
        signing_data = {
            'artifact_path': str(Path(file_path).name),
            'artifact_hash': artifact_hash,
            'hash_algorithm': 'sha256',
            'signed_at': datetime.utcnow().isoformat() + 'Z',
        }
        
        if metadata:
            signing_data['metadata'] = metadata
        
        # Create canonical JSON for signing
        canonical_json = json.dumps(signing_data, sort_keys=True)
        
        # Sign
        signature_bytes = self.private_key.sign(canonical_json.encode('utf-8'))
        
        # Create signature package
        signature_package = {
            **signing_data,
            'signature': base64.b64encode(signature_bytes).decode('utf-8'),
            'signature_algorithm': 'Ed25519',
        }
        
        return signature_package
    
    def verify_signature(self, file_path: str, signature_package: Dict[str, Any]) -> bool:
        """
        Verify artifact signature
        
        Args:
            file_path: Path to artifact to verify
            signature_package: Signature package from sign_artifact()
        
        Returns:
            True if signature is valid, False otherwise
        """
        if self.public_key is None:
            raise ValueError("No public key available for verification")
        
        try:
            # Compute current artifact hash
            current_hash = self.compute_artifact_hash(file_path)
            
            # Check hash matches
            if current_hash != signature_package['artifact_hash']:
                print(f"❌ Hash mismatch: {current_hash} != {signature_package['artifact_hash']}")
                return False
            
            # Reconstruct signing data (without signature field)
            signing_data = {k: v for k, v in signature_package.items() 
                          if k not in ['signature', 'signature_algorithm']}
            
            canonical_json = json.dumps(signing_data, sort_keys=True)
            
            # Decode signature
            signature_bytes = base64.b64decode(signature_package['signature'])
            
            # Verify
            self.public_key.verify(signature_bytes, canonical_json.encode('utf-8'))
            
            return True
            
        except InvalidSignature:
            print("❌ Invalid signature")
            return False
        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False
    
    def sign_and_save(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Sign artifact and save signature to .sig file
        
        Args:
            file_path: Path to artifact to sign
            metadata: Optional metadata
        
        Returns:
            Path to signature file
        """
        signature_package = self.sign_artifact(file_path, metadata)
        
        sig_path = file_path + '.sig'
        with open(sig_path, 'w') as f:
            json.dump(signature_package, f, indent=2)
        
        return sig_path
    
    def verify_from_file(self, file_path: str, sig_path: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify artifact using .sig file
        
        Args:
            file_path: Path to artifact
            sig_path: Path to signature file (default: file_path + '.sig')
        
        Returns:
            Tuple of (is_valid, signature_package)
        """
        if sig_path is None:
            sig_path = file_path + '.sig'
        
        if not Path(sig_path).exists():
            print(f"❌ Signature file not found: {sig_path}")
            return False, None
        
        # Load signature
        with open(sig_path, 'r') as f:
            signature_package = json.load(f)
        
        # Verify
        is_valid = self.verify_signature(file_path, signature_package)
        
        return is_valid, signature_package
    
    def batch_sign_directory(self, directory: str, pattern: str = "*.pt",
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Sign all matching files in directory
        
        Args:
            directory: Directory to scan
            pattern: Glob pattern for files to sign
            metadata: Optional metadata for all signatures
        
        Returns:
            Dictionary mapping file paths to signature file paths
        """
        results = {}
        
        for file_path in Path(directory).rglob(pattern):
            try:
                sig_path = self.sign_and_save(str(file_path), metadata)
                results[str(file_path)] = sig_path
                print(f"✅ Signed: {file_path.name}")
            except Exception as e:
                print(f"❌ Failed to sign {file_path.name}: {e}")
        
        return results
    
    def export_public_key(self, output_path: str) -> str:
        """
        Export public key for distribution
        
        Args:
            output_path: Path to save public key
        
        Returns:
            Path to exported key
        """
        if self.public_key is None:
            raise ValueError("No public key available")
        
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        with open(output_path, 'wb') as f:
            f.write(public_pem)
        
        return output_path


# Singleton instance for convenience
_global_signer = None

def get_artifact_signer() -> ArtifactSigner:
    """Get global artifact signer instance"""
    global _global_signer
    if _global_signer is None:
        _global_signer = ArtifactSigner()
    return _global_signer


def sign_artifact(file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Sign artifact and save signature file"""
    return get_artifact_signer().sign_and_save(file_path, metadata)


def verify_artifact(file_path: str, sig_path: Optional[str] = None) -> bool:
    """Verify artifact signature"""
    is_valid, _ = get_artifact_signer().verify_from_file(file_path, sig_path)
    return is_valid


if __name__ == "__main__":
    # Test artifact signing
    import tempfile
    
    signer = ArtifactSigner()
    
    # Create test artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False) as f:
        f.write("Test model weights")
        test_file = f.name
    
    print(f"Test artifact: {test_file}")
    
    # Sign artifact
    metadata = {
        'model_type': 'gurukul_lora',
        'version': '1.0.0',
        'build_id': 'test_build_123'
    }
    
    sig_package = signer.sign_artifact(test_file, metadata)
    print(f"\n✅ Signed artifact")
    print(f"Hash: {sig_package['artifact_hash'][:16]}...")
    print(f"Signature: {sig_package['signature'][:50]}...")
    
    # Verify signature
    is_valid = signer.verify_signature(test_file, sig_package)
    print(f"\n✅ Signature verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Save and verify from file
    sig_path = signer.sign_and_save(test_file, metadata)
    is_valid_from_file, loaded_package = signer.verify_from_file(test_file)
    print(f"✅ File-based verification: {'PASSED' if is_valid_from_file else 'FAILED'}")
    
    # Test tamper detection
    with open(test_file, 'a') as f:
        f.write("TAMPERED")
    
    is_valid_tampered = signer.verify_signature(test_file, sig_package)
    print(f"✅ Tamper detection: {'PASSED' if not is_valid_tampered else 'FAILED'}")
    
    # Cleanup
    os.unlink(test_file)
    os.unlink(sig_path)
    
    print("\n✅ All artifact signing tests passed!")
